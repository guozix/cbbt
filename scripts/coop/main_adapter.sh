#!/bin/bash

# custom config
DATA=/path/to/datasets
TRAINER=CBBTadapter

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
        TRAINER.COOP.adapter_ratio ${15}
    fi
done

# bash scripts/coop/main_adapter.sh eurosat rn50 end 1 16 False eurosat_OURS_rn50_1ctx_haug 4 True True False 256 150 6000 0.2
# bash scripts/coop/main_adapter.sh fgvc_aircraft rn50 end 1 16 False fgvc_OURS_rn50_1ctx_haug 5 True True False 256 150 6000 0.2
# bash scripts/coop/main_adapter.sh oxford_pets rn50 end 1 16 False oxford_pets_OURS_rn50_1ctx_haug 0 True True False 256 150 6000 0.2
# bash scripts/coop/main_adapter.sh oxford_flowers rn50 end 1 16 False oxford_flowers_OURS_rn50_1ctx_haug 7 True True False 256 150 6000 0.2
# bash scripts/coop/main_adapter.sh dtd rn50 end 1 16 False dtd_OURS_rn50_1ctx_haug 0 True True False 256 150 6000 0.2
# bash scripts/coop/main_adapter.sh stanford_cars rn50 end 1 16 False stanford_cars_OURS_rn50_1ctx_haug 2 True True False 256 80 5000 0.2
# bash scripts/coop/main_adapter.sh food101 rn50 end 1 16 False food101_OURS_rn50_1ctx_haug 4 True True False 256 150 6000 0.2
# bash scripts/coop/main_adapter.sh sun397 rn50 end 1 16 False sun397_OURS_rn50_1ctx_haug 5 True True False 256 40 5500 0.2
# bash scripts/coop/main_adapter.sh caltech101 rn50 end 1 16 False caltech101_OURS_rn50_1ctx_haug 6 True True False 256 150 6000 0.2
# bash scripts/coop/main_adapter.sh ucf101 rn50 end 1 16 False ucf101_OURS_rn50_1ctx_haug 7 True True False 256 150 6000 0.2
# bash scripts/coop/main_adapter.sh imagenet rn50 end 1 16 False imagenet_OURS_rn50_1ctx_haug 4,5,6,7 True True False 256 12 24000

# bash scripts/coop/main_adapter.sh eurosat vit_b16 end 1 16 False eurosat_OURS_vitb16_1ctx_haug 6 True True False 256 150 6000 0.2
# bash scripts/coop/main_adapter.sh fgvc_aircraft vit_b16 end 1 16 False fgvc_OURS_vitb16_1ctx_haug 7 True True False 256 150 6000 0.2
# bash scripts/coop/main_adapter.sh oxford_pets vit_b16 end 1 16 False oxford_pets_OURS_vitb16_1ctx_haug 0 True True False 256 150 6000 0.2
# bash scripts/coop/main_adapter.sh oxford_flowers vit_b16 end 1 16 False oxford_flowers_OURS_vitb16_1ctx_haug 1 True True False 256 150 6000 0.2
# bash scripts/coop/main_adapter.sh dtd vit_b16 end 1 16 False dtd_OURS_vitb16_1ctx_haug 2 True True False 256 150 6000 0.2
# bash scripts/coop/main_adapter.sh stanford_cars vit_b16 end 1 16 False stanford_cars_OURS_vitb16_1ctx_haug 4 True True False 256 80 5500 0.2
# bash scripts/coop/main_adapter.sh food101 vit_b16 end 1 16 False food101_OURS_vitb16_1ctx_haug 6 True True False 256 150 6000 0.2
# bash scripts/coop/main_adapter.sh sun397 vit_b16 end 1 16 False sun397_OURS_vitb16_1ctx_haug 7 True True False 256 40 5500 0.2
# bash scripts/coop/main_adapter.sh caltech101 vit_b16 end 1 16 False caltech101_OURS_vitb16_1ctx_haug 0 True True False 256 150 6000 0.2
# bash scripts/coop/main_adapter.sh ucf101 vit_b16 end 1 16 False ucf101_OURS_vitb16_1ctx_haug 1 True True False 256 150 6000 0.2
# bash scripts/coop/main_adapter.sh imagenet vit_b16 end 1 16 False imagenet_OURS_vitb16_1ctx_haug 4,5,6,7 True True False 256 20 24000 0.2

