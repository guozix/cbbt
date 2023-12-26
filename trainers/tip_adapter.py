import os
import os.path as osp
import random
from tqdm import tqdm
from PIL import Image
import numpy as np
import math

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

# from datasets import build_dataset
# from datasets.utils import build_data_loader
import clip

# from datasets.data_helpers import get_template


def build_cache_model(cfg, clip_model, train_loader_cache):
    clip_model = clip_model.to(torch.device("cuda:0"))
    
    cache_dir = os.path.join(cfg.cache_dir, cfg.backbone, cfg.dataset)
    if_have_cache = os.path.exists(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    if (cfg['load_cache'] == False) or (not if_have_cache):    
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []

                print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for i, (batch) in enumerate(tqdm(train_loader_cache)):
                    images = batch["img"]
                    target = batch["label"]
                    images = images.to(torch.device("cuda:0"))
                    target = target.to(torch.device("cuda:0"))
                    image_features = clip_model.encode_image(images)
                    train_features.append(image_features)
                    if augment_idx == 0:
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
            
        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)
        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).float()

        print("======== saving Tip caches ======== ")
        print('cache_keys.shape, cache_values.shape', cache_keys.shape, cache_values.shape)
        torch.save(cache_keys, cache_dir + '/keys_' + str(cfg['shots']) + "shots.pt")
        torch.save(cache_values, cache_dir + '/values_' + str(cfg['shots']) + "shots.pt")

    else:
        cache_keys = torch.load(cache_dir + '/keys_' + str(cfg['shots']) + "shots.pt")
        cache_values = torch.load(cache_dir + '/values_' + str(cfg['shots']) + "shots.pt")

    return cache_keys, cache_values


def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts_ = [t.format(classname) for t in template]
            texts = clip.tokenize(texts_)
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)
        print("Created clip text classifiers by:", texts_)
        
        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights


class Tip_Adapter(nn.Module):
    def __init__(self, cfg, classnames, clip_model, tip_dataloader, if_clip_score=False, logits_scale=60):
        super(Tip_Adapter, self).__init__()
        train_loader_cache = tip_dataloader
        
        self.if_clip_score = if_clip_score
        if self.if_clip_score:
            self.clip_weights = clip_classifier(classnames, cfg.dataset_prompt_template, clip_model)

        # Construct the cache model by few-shot training set
        print("\nConstructing cache model by few-shot visual features and labels.")
        cache_keys, cache_values = build_cache_model(cfg, clip_model, train_loader_cache)
        
        # Enable the cached keys to be learnable
        self.adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(clip_model.dtype)
        self.adapter.weight = nn.Parameter(cache_keys.t())
        
        self.W2 = cache_values
        
        self.beta, self.alpha = cfg['init_beta'], cfg['init_alpha']
        self.logits_scale = logits_scale

    def forward(self, x, if_search_hp=False):
        if not if_search_hp:
            beta, alpha = self.beta, self.alpha
        else:
            raise NotImplementedError()
            # beta, alpha = search_hp()
        image_features = x
        affinity = self.adapter(image_features)
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ self.W2
        
        if self.if_clip_score:
            clip_logits = self.logits_scale * image_features @ self.clip_weights
            tip_logits = clip_logits + cache_logits * alpha
        else:
            tip_logits = cache_logits * alpha
        
        return tip_logits
