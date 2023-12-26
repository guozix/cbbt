import argparse
import torch

import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# custom
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet

import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r

import datasets.cifar10

import trainers.coop
import trainers.cocoop
import trainers.zsclip
import trainers.cooppara
import trainers.coopadapter
import trainers.cooptip


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head

    if args.model_dir:
        cfg.MODEL.INIT_WEIGHTS = args.model_dir


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp32"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    
    cfg.INPUT.random_resized_crop_scale = (0.8, 1.0)
    cfg.INPUT.cutout_proportion = 0.4
    cfg.INPUT.TRANSFORMS_TEST = ("resize", "center_crop", "normalize")
    
    cfg.TRAIN.grad_estimate = False
    cfg.TRAIN.grad_dir_pca = False
    cfg.TRAIN.grad_dir_pca_dim = 100
    cfg.TRAIN.text_batchsize = 5000
    
    cfg.TRAIN.cma_popsize = 10
    cfg.TRAIN.cma_budget = 1000000 # 8000
    cfg.TRAIN.cma_bound = 0
    cfg.TRAIN.cma_sigma = 1
    cfg.TRAIN.cma_alpha = 1
    cfg.TRAIN.cma_init_startp = False
    
    cfg.TRAINER.COOP.IF_CTX_INIT = False
    cfg.TRAINER.COOP.IF_TEXT_GUIDE = False
    cfg.TRAINER.COOP.TEXT_GUIDE_SCALE = 0.5
    
    cfg.TRAINER.COOP.DeFo_query_nums = 0
    # cfg.TRAINER.COOP.DeFo_init_clsname = True
    cfg.TRAINER.COOP.DeFo_fix_kweight = True
    
    cfg.TRAINER.COOP.adapter_ratio = 0.2
    cfg.TRAINER.COOP.adapter_startepoch = 10
    cfg.TRAINER.COOP.only_adapter = False
    
    ### Tip configs
    cfg.TIP = CN()
    cfg.TIP.load_cache = False
    cfg.TIP.load_pre_feat = False
    # cfg.TIP.load_cache = True
    # cfg.TIP.load_pre_feat = True

    # ------ Hyperparamters ------
    # cfg.TIP.search_hp = True
    cfg.TIP.search_hp = False

    cfg.TIP.search_scale = [12, 5]
    cfg.TIP.search_step = [200, 20]

    cfg.TIP.init_beta = 1.0
    cfg.TIP.init_alpha = 3.0
    
    cfg.TIP.dataset = cfg.DATASET.NAME
    cfg.TIP.shots = 16
    cfg.TIP.backbone = ''

    cfg.TIP.lr = 0.001
    cfg.TIP.augment_epoch = 10
    cfg.TIP.train_epoch = 20
    
    cfg.TIP.cache_dir = './tip_cache'
    cfg.TIP.dataset_prompt_template = ["a photo of a {}."]
    
    ### TIP_OPTIM
    cfg.TIP_OPTIM = CN()
    cfg.TIP_OPTIM.NAME = "adamw"
    cfg.TIP_OPTIM.LR = 0.001
    cfg.TIP_OPTIM.WEIGHT_DECAY = 0.01
    cfg.TIP_OPTIM.MOMENTUM = 0.9
    cfg.TIP_OPTIM.SGD_DAMPNING = 0.0
    cfg.TIP_OPTIM.SGD_NESTEROV = False
    cfg.TIP_OPTIM.RMSPROP_ALPHA = 0.99
    cfg.TIP_OPTIM.ADAM_BETA1 = 0.9
    cfg.TIP_OPTIM.ADAM_BETA2 = 0.999

    cfg.TIP_OPTIM.STAGED_LR = False
    cfg.TIP_OPTIM.NEW_LAYERS = ()
    cfg.TIP_OPTIM.BASE_LR_MULT = 0.1
    # Learning rate scheduler
    cfg.TIP_OPTIM.LR_SCHEDULER = "cosine"
    # -1 or 0 means the stepsize is equal to max_epoch
    cfg.TIP_OPTIM.STEPSIZE = (-1, )
    cfg.TIP_OPTIM.GAMMA = 0.1
    cfg.TIP_OPTIM.MAX_EPOCH = 20
    # Set WARMUP_EPOCH larger than 0 to activate warmup training
    cfg.TIP_OPTIM.WARMUP_EPOCH = -1
    # Either linear or constant
    cfg.TIP_OPTIM.WARMUP_TYPE = "constant"
    # Constant learning rate when type=constant
    cfg.TIP_OPTIM.WARMUP_CONS_LR = 1e-5
    # Minimum learning rate when type=linear
    cfg.TIP_OPTIM.WARMUP_MIN_LR = 1e-5
    # Recount epoch for the next scheduler (last_epoch=-1)
    # Otherwise last_epoch=warmup_epoch
    cfg.TIP_OPTIM.WARMUP_RECOUNT = True
    
    ### SPSA
    cfg.SPSA = CN()
    cfg.SPSA.p_eps = 1.0
    cfg.SPSA.MOMS = 0.9
    cfg.SPSA.OPT_TYPE = "spsa"
    cfg.SPSA.SPSA_PARAMS = [1.0,0.005,0.01,0.4,0.2]
    cfg.SPSA.SP_AVG = 5
    # cfg.SPSA.spsa_os=1.0
    # cfg.SPSA.spsa_gamma = 0.2
    # cfg.SPSA.spsa_a=0.01
    # cfg.SPSA.spsa_c = 0.005
    # cfg.SPSA.alpha=0.4
    

def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)
        
    # 2. From the Tip-Adapter config file
    if args.tip_config_file:
        cfg.merge_from_file(args.tip_config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)
    
    cfg.TIP_OPTIM.MAX_EPOCH = cfg.TIP.train_epoch
    cfg.TIP.backbone = cfg.MODEL.BACKBONE.NAME

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)
    
    if cfg.TRAINER.COOP.IF_CTX_INIT:
        if cfg.TRAINER.COOP.N_CTX == 1:
            cfg.TRAINER.COOP.CTX_INIT = "a"
        elif cfg.TRAINER.COOP.N_CTX == 2:
            cfg.TRAINER.COOP.CTX_INIT = "a photo"
        elif cfg.TRAINER.COOP.N_CTX == 8:
            cfg.TRAINER.COOP.CTX_INIT = "a photo of a a photo of a"
        elif cfg.TRAINER.COOP.N_CTX == 16:
            cfg.TRAINER.COOP.CTX_INIT = "a photo of a a photo of a a photo of a a photo of a" # "a a a a a a a a a a a a a a a a" 
        else:
            cfg.TRAINER.COOP.CTX_INIT = "a photo of a"

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument(
        "--tip-config-file",
        type=str,
        default="",
        help="path to config file for Tip-Adapter",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)
