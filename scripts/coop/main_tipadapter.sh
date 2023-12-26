#!/bin/bash

# custom config
DATA=/path/to/datasets
TRAINER=CBBTtip

DATASET=$1
CFG=$2  # config file
CTP=$3  # class token position (end or middle)
NCTX=$4  # number of context tokens
SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
CSC=$6  # class-specific context (False or True)

run_ID=$7
export CUDA_VISIBLE_DEVICES=$8

grad_estimate_=$9
init_ctx=${10}
uniword_pca=${11}
pca_dim=${12}
maxepoch=${13}  # 200
text_batch=${14}

for SEED in 1 # 2 # 3
do
    DIR=output/${run_ID}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --tip-config-file configs/tip_configs/${DATASET}.yaml \
        --output-dir ${DIR} \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS} \
        TRAIN.grad_estimate ${grad_estimate_} \
        TRAINER.COOP.IF_CTX_INIT ${init_ctx} \
        TRAIN.grad_dir_pca ${uniword_pca} \
        TRAIN.grad_dir_pca_dim ${pca_dim} \
        TRAIN.text_batchsize ${text_batch} \
        OPTIM.MAX_EPOCH ${maxepoch} \
        TRAINER.COOP.adapter_startepoch ${15} \
        TRAINER.COOP.only_adapter ${16}
    fi
done


# bash scripts/coop/main_tipadapter.sh eurosat rn50 end 1 16 False eurosat_OURS_TIP_rn50_1ctx_haug 0 True True False 256 150 6000 10 False
# bash scripts/coop/main_tipadapter.sh fgvc_aircraft rn50 end 1 16 False fgvc_OURS_TIP_rn50_1ctx_haug 1 True True False 256 100 6000 10 False
# bash scripts/coop/main_tipadapter.sh oxford_pets rn50 end 1 16 False oxford_pets_OURS_TIP_rn50_1ctx_haug 7 True True False 256 100 6000 10 False
# bash scripts/coop/main_tipadapter.sh oxford_flowers rn50 end 1 16 False oxford_flowers_OURS_TIP_rn50_1ctx_haug 6 True True False 256 100 6000 10 False
# bash scripts/coop/main_tipadapter.sh dtd rn50 end 1 16 False dtd_OURS_TIP_rn50_1ctx_haug 5 True True False 256 100 6000 10 False
# bash scripts/coop/main_tipadapter.sh stanford_cars rn50 end 1 16 False stanford_cars_OURS_TIP_rn50_1ctx_haug 2 True True False 256 80 5000 10 False
# bash scripts/coop/main_tipadapter.sh food101 rn50 end 1 16 False food101_OURS_TIP_rn50_1ctx_haug 4 True True False 256 150 6000 10 False
# bash scripts/coop/main_tipadapter.sh sun397 rn50 end 1 16 False sun397_OURS_TIP_rn50_1ctx_haug 5 True True False 256 40 5500 10 False
# bash scripts/coop/main_tipadapter.sh caltech101 rn50 end 1 16 False caltech101_OURS_TIP_rn50_1ctx_haug 6 True True False 256 150 6000 10 False
# bash scripts/coop/main_tipadapter.sh ucf101 rn50 end 1 16 False ucf101_OURS_TIP_rn50_1ctx_haug 4 True True False 256 150 6000 10 False
# bash scripts/coop/main_tipadapter.sh imagenet rn50 end 1 16 False imagenet_OURS_TIP_rn50_1ctx_haug 4,5,6,7 True True False 256 12 24000


# bash scripts/coop/main_tipadapter.sh eurosat vit_b16 end 1 16 False eurosat_OURS_TIP_vitb16_1ctx_haug 0 True True False 256 150 6000 10 False
# bash scripts/coop/main_tipadapter.sh fgvc_aircraft vit_b16 end 1 16 False fgvc_OURS_TIP_vitb16_1ctx_haug 2 True True False 256 100 6000 10 False
# bash scripts/coop/main_tipadapter.sh oxford_pets vit_b16 end 1 16 False oxford_pets_OURS_TIP_vitb16_1ctx_haug 4 True True False 256 100 6000 10 False
# bash scripts/coop/main_tipadapter.sh oxford_flowers vit_b16 end 1 16 False oxford_flowers_OURS_TIP_vitb16_1ctx_haug 5 True True False 256 100 6000 10 False
# bash scripts/coop/main_tipadapter.sh dtd vit_b16 end 1 16 False dtd_OURS_TIP_vitb16_1ctx_haug 3 True True False 256 100 6000 10 False
# bash scripts/coop/main_tipadapter.sh stanford_cars vit_b16 end 1 16 False stanford_cars_OURS_TIP_vitb16_1ctx_haug 4 True True False 256 80 5500 10 False
# bash scripts/coop/main_tipadapter.sh food101 vit_b16 end 1 16 False food101_OURS_TIP_vitb16_1ctx_haug 6 True True False 256 100 6000 10 False
# bash scripts/coop/main_tipadapter.sh sun397 vit_b16 end 1 16 False sun397_OURS_TIP_vitb16_1ctx_haug 7 True True False 256 40 5500 10 False
# bash scripts/coop/main_tipadapter.sh caltech101 vit_b16 end 1 16 False caltech101_OURS_TIP_vitb16_1ctx_haug 7 True True False 256 100 6000 10 False
# bash scripts/coop/main_tipadapter.sh ucf101 vit_b16 end 1 16 False ucf101_OURS_TIP_vitb16_1ctx_haug 1 True True False 256 150 6000 10 False
# bash scripts/coop/main_tipadapter.sh imagenet vit_b16 end 1 16 False imagenet_OURS_TIP_vitb16_1ctx_haug 4,5,6,7 True True False 256 12 24000

