import os.path as osp
from tqdm import tqdm
import numpy as np
import math

import time

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from datasets.data_helpers import get_template

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        # self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts): # (C,L,D) (C,L) / (N,C,L,D) (C,L) -> (C,D) / (N,C,D)
        if prompts.dim() == 3:
            x = prompts + self.positional_embedding.type(self.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x).type(self.dtype)

            return x

        elif prompts.dim() == 4:
            N, C, L, D = prompts.shape
            x = prompts + self.positional_embedding.type(self.dtype)[None,None,:,:]
            x = x.flatten(0, 1)
            
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x).type(self.dtype)
            
            x = x.reshape(N, C, x.shape[-2], x.shape[-1])

            # cls_eos_idx = tokenized_prompts.argmax(dim=-1)
            # cls_eos_idx = cls_eos_idx[None, :].expand(N, -1).flatten()
            # x = x[torch.arange(x.shape[0]), cls_eos_idx]
            # x = x.reshape(N, C, D)
            # x = x @ self.text_projection

            return x

        else:
            return

def cumu_mul(shape_):
    ret = 1
    for i in shape_:
        ret = ret * i
    return ret

optim_idx = -1
optim_idx_max = 256  # q
epsilon_ = 0.001

cat_nums = 100
text_batchsize = 24000 # 3600 # 32000 # 6000

estim_type = "gauss" # "prob_partgrad" "partgrad"  "loss<"  "pca"

def rademacher(shape, dir_nums=1, device='cpu'):
    global estim_type
    global optim_idx

    if estim_type == "gauss":  # uni_word PCA dir proj
        # last_cktm = time.time()
        x = torch.stack([sub_rand(shape, device) for i in range(dir_nums)], dim=0)
        # print("rademacher once time:", time.time() - last_cktm)
        return x
    
        # x = torch.zeros((dir_nums, cumu_mul(shape)), device=device).float()
        # nn.init.normal_(x)
        # x = x / x.norm(p=2, dim=-1, keepdim=True)
        # x = x.reshape(dir_nums, *shape)
        # return x
    else:
        return

def sub_rand(shape, device='cpu'):
    x = torch.zeros(cumu_mul(shape), device=device).float()
    nn.init.normal_(x)
    x = x / x.norm(p=2, dim=0)
    x = x.reshape(shape)
    return x


class grad_modifier():
    def __init__(self, alpha=0.4) -> None:
        self.grad_hist = None
        self.alpha = alpha
    def onehot_momentum(self, grad_cur):
        flatten_grad = grad_cur.reshape(-1)
        if self.grad_hist is None:
            self.grad_hist = torch.zeros_like(flatten_grad)
        
        _, dir_idx = flatten_grad.max(0)
        if self.grad_hist[dir_idx] > 1e-6:
            flatten_grad[dir_idx] = self.alpha * self.grad_hist[dir_idx] + (1-self.alpha) * flatten_grad[dir_idx]
            self.grad_hist[dir_idx] = flatten_grad[dir_idx]
        else:
            self.grad_hist[dir_idx] = flatten_grad[dir_idx]

        return flatten_grad.reshape(grad_cur.shape)
    


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        
        global cat_nums
        cat_nums = n_cls
        
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # ### init linear proj with uniword PCA
        # # compress word embedding: 512 -> intrinsic_dim
        if cfg.TRAIN.grad_dir_pca:
            # pca_vec = torch.load("datasets/T5_w2s_eurosat_1000s_uniwords_pca{}_sklearn.pkl".format(cfg.TRAIN.grad_dir_pca_dim))
            pca_vec = torch.load("datasets/Vocabulary_pca{}_sklearn.pkl".format(cfg.TRAIN.grad_dir_pca_dim))
            self.ctx_proj = pca_vec  # intrinsic_dim*512
        else:
            self.ctx_proj = None
        
        global optim_idx_max
        optim_idx_max = cfg.TRAIN.grad_dir_pca_dim
        global text_batchsize
        text_batchsize = cfg.TRAIN.text_batchsize
        
        ### find main words with PCA results
        # if cfg.TRAIN.grad_dir_pca:
        #     pca_vec = torch.load("datasets/T5_w2s_eurosat_1000s_uniwords_pca{}_word.pkl".format(cfg.TRAIN.grad_dir_pca_dim))
        #     self.ctx_proj = torch.from_numpy(pca_vec)  # 512 * intrinsic_word_num
        # else:
        #     self.ctx_proj = None
        
        self.base_ctx = None
        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            # self.base_ctx = ctx_vectors
            if self.ctx_proj is not None:
                ctx_vectors = self.pca_proj(ctx_vectors)
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        self.base_cat_ctx = None
        # self.ctx = nn.Parameter(ctx_vectors[3:4])  # to be optimized
        # self.base_cat_ctx = ctx_vectors[:3]

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self, if_permute=0, permute_scale=True):
        global epsilon_
        global estim_type
        global optim_idx_max
        if estim_type == "partgrad":
            epsilon_ = 0.001
        elif estim_type == "pca":
            epsilon_ = 0.05
        elif estim_type == "gauss":
            epsilon_ = max(1 / cumu_mul(self.ctx.shape), 0.001)
        else:
            epsilon_ = 0.001
            
        permute_ = None
        flag_unzqu = False
        if if_permute:
            permute_ = rademacher(self.ctx.shape, if_permute, device=self.ctx.device)
            if permute_scale:
                permute_ = epsilon_ * permute_
                if optim_idx_max <= 1:
                    ctx = self.ctx + permute_[0]
            else:
                if optim_idx_max <= 1:
                    ctx = self.ctx + epsilon_ * permute_[0]
                else:
                    ctx = self.ctx.unsqueeze(0).expand(permute_.shape[0], -1, -1)
                    ctx = ctx + epsilon_ * permute_  # N, L, D
                    # ctx = torch.cat([self.ctx.unsqueeze(0), ctx], dim=0)
                    flag_unzqu = True
        else:
            ctx = self.ctx

        ## proj
        if self.ctx_proj is not None:
            ctx = self.pca_proj_inv(ctx)

        if self.base_ctx is not None:
            self.base_ctx.requires_grad_(False)
            ctx = ctx + self.base_ctx.to(ctx.device)
        if self.base_cat_ctx is not None:
            self.base_cat_ctx.requires_grad_(False)
            ctx = torch.cat([self.base_cat_ctx.to(ctx.device), ctx], dim=0)

        if not flag_unzqu:
            if ctx.dim() == 2:
                ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        else:
            if ctx.dim() == 3:
                ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            if not flag_unzqu:
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=1,
                )
            else:
                prompts = torch.cat(
                    [
                        prefix.unsqueeze(0).expand(ctx.shape[0], -1, -1, -1),  # (N, n_cls, 1, dim)
                        ctx,     # (N, n_cls, n_ctx, dim)
                        suffix.unsqueeze(0).expand(ctx.shape[0], -1, -1, -1),  # (N, n_cls, *, dim)
                    ],
                    dim=2,
                )
        else:
            raise ValueError

        if if_permute:
            return prompts, permute_
        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.cur_imgs = None
        self.grad_manage = grad_modifier()
        self.text_projection = clip_model.text_projection

        self.reg_text_feat = None
        if cfg.TRAINER.COOP.IF_TEXT_GUIDE:
            dataset_prompt, eurosat_object_categories = get_template(cfg.DATASET.NAME)
            cat_prompt = [dataset_prompt.format(i) for i in eurosat_object_categories]
            
            tokenized_cat_prompt = torch.cat([clip.tokenize(p) for p in cat_prompt])
            with torch.no_grad():
                embedding_cat_prompt = clip_model.token_embedding(tokenized_cat_prompt).type(self.dtype)

            self.reg_text_feat = self.text_encoder(embedding_cat_prompt, tokenized_cat_prompt)
            print("**guide text prompts", cat_prompt)
            print("**self.reg_text_feat.shape", self.reg_text_feat.shape)

    def forward(self, image, if_permute=0, use_cache_img=False, permute_scale=True):
        if if_permute:
            prompts, mutation = self.prompt_learner(if_permute=if_permute, permute_scale=permute_scale) # -> (N+1,C,L=77,D) (N,L=1,D)
        else:
            prompts = self.prompt_learner(if_permute=if_permute)
        tokenized_prompts = self.tokenized_prompts  # (C,77)

        # image first
        if use_cache_img:
            image_features = self.cur_imgs
        else:
            image_features = self.image_encoder(image.type(self.dtype))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            self.cur_imgs = image_features
        
        # last_cktm = time.time()
        x = self.text_encoder(prompts) # (C,L,D) (C,L) / (N,C,L,D) (C,L) -> (C,D) / (N,C,D)
        # print("===== text_encoder once time:", time.time() - last_cktm)
        
        if x.dim() == 4:  # train
            x = torch.stack([x_[torch.arange(x_.shape[0]), tokenized_prompts.argmax(dim=-1)] for x_ in x], dim=0)
            text_features = x @ self.text_projection
            del x

            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
            # N,C,_ = text_features.shape
            # B,_ = image_features.shape
            logits = image_features @ text_features.permute(0,2,1) # (B,D) (N,D,C) -> (N,B,C)
            # assert logits.shape[0] == N
            # assert logits.shape[1] == B
            # assert logits.shape[2] == C
        else:
            text_features = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
            del x
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logits = image_features @ text_features.permute(1,0)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * logits

        if if_permute:
            return logits, mutation
        else:
            return logits
        
    def set_grad(self, optim_dir, pos):
        if pos:
            self.prompt_learner.ctx.grad = optim_dir
        else:
            self.prompt_learner.ctx.grad = (optim_dir * -1.0)
        torch.nn.utils.clip_grad_norm(parameters=self.prompt_learner.parameters(), max_norm=5, norm_type=2)  
            
@TRAINER_REGISTRY.register()
class CBBT(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model.text_encoder = nn.DataParallel(self.model.text_encoder)
        
    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.COOP.PREC

        if self.cfg.TRAIN.grad_estimate:
            ################### clean forward, pertubation forward
            with torch.no_grad():
                output = self.model(image)
                loss = F.cross_entropy(output, label)

            loss_summary = {
                "loss": loss.item(),
                "acc": compute_accuracy(output, label)[0].item(),
            }
            
            # update manualy
            global epsilon_
            global estim_type
            global optim_idx_max
            global text_batchsize
            global cat_nums

            self.model_zero_grad(None)

            # q estimate
            grad_alldir = torch.zeros_like(self.model.prompt_learner.ctx)
            dim_num = 1/epsilon_ # cumu_mul(grad_alldir.shape)
            
            onece_dir_num_max = int(text_batchsize / cat_nums)
            
            # loop_st = time.time()
            # last_cktm = time.time()
            for dir_bidx in tqdm(range(math.ceil(optim_idx_max/onece_dir_num_max))):
                if optim_idx_max <= onece_dir_num_max:
                    cur_dir_nums = optim_idx_max
                else:
                    cur_dir_nums = min(optim_idx_max - dir_bidx * onece_dir_num_max, onece_dir_num_max)
                
                with torch.no_grad():
                    # last_cktm_ = time.time()
                    outputs, permute_dirs = self.model(image, if_permute=cur_dir_nums, use_cache_img=True, permute_scale=False) # -> (N,B,C) (N,L,D)
                    # print("Forward onece time:", time.time() - last_cktm_)
                    # last_cktm = time.time()
                    
                    for dir in range(outputs.shape[0]):
                        output_, permute_dir = outputs[dir], permute_dirs[dir]
                        loss_ = F.cross_entropy(output_, label)
                        grad_dir = (loss_ - loss) / epsilon_
                        # permute_dir = permute_dir / epsilon_  # permute_scale=False makes this line useless
                        grad_alldir = grad_alldir + permute_dir * grad_dir
                    # print("proc once time:", time.time() - last_cktm)
                    # last_cktm = time.time()
            # print("loop time:", time.time() - loop_st)
            # last_cktm = time.time()
            
            grad_alldir = grad_alldir / optim_idx_max * dim_num
            self.model.set_grad(grad_alldir, True)

            self.model_update(None)

            if (self.batch_idx + 1) == self.num_batches:
                self.update_lr()

        else:
            image, label = self.parse_batch_train(batch)
            
            prec = self.cfg.TRAINER.COOP.PREC
            if prec == "amp":
                with autocast():
                    output = self.model(image)
                    loss = F.cross_entropy(output, label)
                self.optim.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
            else:
                output = self.model(image)
                loss = F.cross_entropy(output, label)
                self.model_backward_and_update(loss)

            loss_summary = {
                "loss": loss.item(),
                "acc": compute_accuracy(output, label)[0].item(),
            }

            if (self.batch_idx + 1) == self.num_batches:
                self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def after_epoch(self):
        global epsilon_
        global estim_type
        global optim_idx_max
        global text_batchsize
        print("** Justify params **", "epsilon_",epsilon_,"estim_type",estim_type,"optim_idx_max",optim_idx_max,'text_batchsize',text_batchsize)
        
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            self.epoch + 1
        ) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0 if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False

        if do_test and self.cfg.TEST.FINAL_MODEL == 'best_val':
            curr_result = self.test(split='test')
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    model_name='model-best.pth.tar'
                )

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
