#!/bin/bash

# custom config
DATA=/root/gzx/coop_datasets
TRAINER=CBBT

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
        OPTIM.MAX_EPOCH ${maxepoch}
    fi
done

# bash scripts/coop/main.sh eurosat rn50 end 1 16 False eurosat_rn50_1ctx 1 True True False 256 150 6000