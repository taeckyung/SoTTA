#!/bin/bash

SRC_PREFIX="reproduce_src"
LOG_PREFIX="eval_results"

BASE_DATASETS=("cifar10noisy") # ("cifar10noisy" "cifar100noisy" "imagenetnoisy")
METHODS=("SoTTA") # ("Src" "BN_Stats" "PseudoLabel" "TENT" "CoTTA" "SAR" "RoTTA" "EATA" "LAME" "SoTTA")
SEEDS=(0 1 2)
NOISY_TYPES=("--noisy_type mnist")
# Guidelines for NOISY_TYPES
#
# CIFAR10-C (cifar10noisy)
# Benign: "--noisy_type original"
# Near:   "--noisy_type cifar100 --noisy_size 10000"
# Far:    "--noisy_type mnist"
# Attack: "--noisy_type repeat --tta_attack_type indiscriminate --tta_attack_step 10 --tta_attack_eps 0.1"
# Noise:  "--noisy_type uniform --noisy_size 10000"
#
# CIFAR100-C (cifar100noisy)
# Benign: "--noisy_type original"
# Near:   "--noisy_type imagenet --noisy_size 10000"
# Far:    "--noisy_type mnist"
# Attack: "--noisy_type repeat --tta_attack_type indiscriminate --tta_attack_step 10 --tta_attack_eps 0.1"
# Noise:  "--noisy_type uniform --noisy_size 10000"
#
# ImageNet-C (imagenetnoisy)
# Benign: "--noisy_type original"
# Near:   "--noisy_type cifar100"
# Far:    "--noisy_type mnist --noisy_size 50000"
# Attack: "--noisy_type repeat --tta_attack_type indiscriminate --tta_attack_step 1 --tta_attack_eps 0.2"
# Noise:  "--noisy_type uniform --noisy_size 50000"

echo BASE_DATASETS: "${BASE_DATASETS[@]}"
echo METHODS: "${METHODS[@]}"
echo SEEDS: "${SEEDS[@]}"
echo NOISY_TYPE: "${NOISY_TYPES[@]}"

GPUS=(0 1 2 3 4 5 6 7) #available gpus
NUM_GPUS=${#GPUS[@]}

sleep 1 # prevent mistake
mkdir raw_logs # save console outputs here




#### Useful functions
wait_n() {
  #limit the max number of jobs as NUM_MAX_JOB and wait
  background=($(jobs -p))
  local default_num_jobs=8  #num concurrent jobs
  local num_max_jobs=${1:-$default_num_jobs}
  if ((${#background[@]} >= num_max_jobs)); then
    wait -n
  fi
}


test_time_adaptation() {
  ###############################################################
  ###### Run Baselines & Ours; Evaluation: Target domains  ######
  ###############################################################

  i=0

  for NOISY_TYPE in "${NOISY_TYPES[@]}"; do
    for DATASET in "${BASE_DATASETS[@]}"; do
      for METHOD in "${METHODS[@]}"; do

        update_every_x="64"
        memory_size="64"
        SEED="0"
        lr="0.001" #other baselines
        validation="--dummy"
        weight_decay="0"

        if [ "${DATASET}" = "cifar10" ] || [ "${DATASET}" = "cifar10noisy" ]; then
          MODEL="resnet18"
          CP_base="log/cifar10/Src/tgt_test/"${SRC_PREFIX}

          #              TGTS="test"
          TGTS="gaussian_noise-5
              shot_noise-5
              impulse_noise-5
              defocus_blur-5
              glass_blur-5
              motion_blur-5
              zoom_blur-5
              snow-5
              frost-5
              fog-5
              brightness-5
              contrast-5
              elastic_transform-5
              pixelate-5
              jpeg_compression-5"
        elif [ "${DATASET}" = "cifar100" ] || [ "${DATASET}" = "cifar100noisy" ]; then
            MODEL="resnet18"
            CP_base="log/cifar100/Src/tgt_test/"${SRC_PREFIX}

            #              TGTS="test"
            TGTS="gaussian_noise-5
                shot_noise-5
                impulse_noise-5
                defocus_blur-5
                glass_blur-5
                motion_blur-5
                zoom_blur-5
                snow-5
                frost-5
                fog-5
                brightness-5
                contrast-5
                elastic_transform-5
                pixelate-5
                jpeg_compression-5"
        elif [ "${DATASET}" = "imagenetnoisy" ]; then
          MODEL="resnet18_pretrained"
          CP_base="log/imagenet/Src/tgt_test/"${SRC_PREFIX}
          #              TGTS="test"
          TGTS="gaussian_noise-5
              shot_noise-5
              impulse_noise-5
              defocus_blur-5
              glass_blur-5
              motion_blur-5
              zoom_blur-5
              snow-5
              frost-5
              fog-5
              brightness-5
              contrast-5
              elastic_transform-5
              pixelate-5
              jpeg_compression-5"
        fi

        for SEED in "${SEEDS[@]}"; do #multiple seeds

          if [ "${DATASET}" = "cifar10noisy" ] || [ "${DATASET}" = "cifar100noisy" ]; then
            CP="--load_checkpoint_path ${CP_base}_${SEED}/cp/cp_last.pth.tar"
          else
            CP="" # for imagenet
          fi


            if [ "${METHOD}" = "Src" ]; then
              EPOCH=0
              #### Train with BN
  #            CP=${CP_base}_${SEED}/cp/cp_last.pth.tar
  #            CP=timm
              for TGT in $TGTS; do
              python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method ${METHOD} --tgt ${TGT} --model $MODEL --epoch $EPOCH ${CP} --update_every_x ${update_every_x} --seed $SEED \
                --log_prefix ${LOG_PREFIX}_${SEED} \
                ${NOISY_TYPE} \
                ${validation}  \
                2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &

              i=$((i + 1))
              wait_n
            done
            
            elif [ "${METHOD}" = "SRC_FOR_TTA_ATTACK" ]; then
              EPOCH=0
              #### Train with BN
  #            CP=${CP_base}_${SEED}/cp/cp_last.pth.tar
  #            CP=timm
              for TGT in $TGTS; do
              python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method ${METHOD} --tgt ${TGT} --model $MODEL --epoch $EPOCH ${CP} --update_every_x ${update_every_x} --seed $SEED \
                --log_prefix ${LOG_PREFIX}_${SEED} --online\
                ${NOISY_TYPE} \
                ${validation}  \
                2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &

              i=$((i + 1))
              wait_n
            done


          elif [ "${METHOD}" = "SoTTA" ]; then

            lr="0.001"
            EPOCH=1
            loss_scaler=0
            bn_momentum=0.2

            if [ "${DATASET}" = "cifar10" ] || [ "${DATASET}" = "cifar10noisy" ]; then
              high_threshold=0.99
            elif [ "${DATASET}" = "cifar100" ] || [ "${DATASET}" = "cifar100noisy" ]; then
              high_threshold=0.66
            elif [ "${DATASET}" = "imagenetnoisy" ]; then
              high_threshold=0.33
            fi
            #### Train with BN

            for dist in 1; do #dist 0: non-i.i.d. / dist 1: i.i.d.
              for memory_type in "HUS"; do #
                for TGT in $TGTS; do
                  python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method SoTTA --tgt ${TGT} --model $MODEL --epoch $EPOCH ${CP} --seed $SEED \
                    --remove_cp --online --use_learned_stats --lr ${lr} --weight_decay ${weight_decay} --tgt_train_dist ${dist} --update_every_x ${update_every_x} --memory_size ${memory_size} --memory_type ${memory_type} --bn_momentum ${bn_momentum} \
                    --log_prefix ${LOG_PREFIX}_${SEED}_dist${dist}_mt${bn_momentum}_${memory_type}_ht${high_threshold}_lr${lr} \
                    --loss_scaler ${loss_scaler} \
                    ${NOISY_TYPE} --esm \
                    ${validation} \
                    --high_threshold ${high_threshold} \
                    2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &

                  i=$((i + 1))
                  wait_n
                done
              done
          done


          elif [ "${METHOD}" = "RoTTA" ]; then
            EPOCH=1
            loss_scaler=0
            lr="0.001"
            bn_momentum=0.05
            #### Train with BN

            for dist in 1; do #dist 0: non-i.i.d. / dist 1: i.i.d.

              for memory_type in "CSTU"; do #
                for TGT in $TGTS; do
                  python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method "RoTTA" --tgt ${TGT} --model $MODEL --epoch $EPOCH ${CP} --seed $SEED \
                    --remove_cp --online --use_learned_stats --lr ${lr} --weight_decay ${weight_decay} --tgt_train_dist ${dist} --update_every_x ${update_every_x} --memory_size ${memory_size} --memory_type ${memory_type} --bn_momentum "0.05" \
                    --log_prefix "${LOG_PREFIX}_${SEED}_dist${dist}_mt0.05_${memory_type}" \
                    --loss_scaler ${loss_scaler} \
                    ${NOISY_TYPE} \
                    ${validation} \
                    2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &

                  i=$((i + 1))
                  wait_n
                done
              done
            done

          elif [ "${METHOD}" = "BN_Stats" ]; then
            EPOCH=1

            #### Train with BN
            for dist in 1; do
#              CP=${CP_base}_${SEED}/cp/cp_last.pth.tar
              for TGT in $TGTS; do

                python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method ${METHOD} --tgt ${TGT} --model $MODEL --epoch $EPOCH ${CP} --seed $SEED \
                  --remove_cp --online --tgt_train_dist ${dist} --update_every_x ${update_every_x} --memory_size ${memory_size} \
                  --log_prefix ${LOG_PREFIX}_${SEED}_dist${dist} \
                  ${NOISY_TYPE} \
                  ${validation} \
                  2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &

                i=$((i + 1))
                wait_n
              done

            done
          elif [ "${METHOD}" = "PseudoLabel" ]; then
            EPOCH=1
            lr=0.001
            #### Train with BN
            for dist in 1; do
#              CP=${CP_base}_${SEED}/cp/cp_last.pth.tar
              for TGT in $TGTS; do

                python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method ${METHOD} --tgt ${TGT} --model $MODEL --epoch $EPOCH ${CP} --seed $SEED \
                  --remove_cp --online --tgt_train_dist ${dist} --update_every_x ${update_every_x} --memory_size ${memory_size} \
                  --lr ${lr} --weight_decay ${weight_decay} \
                  --log_prefix ${LOG_PREFIX}_${SEED}_dist${dist} \
                  ${validation} \
                  ${NOISY_TYPE} \
                  2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &

                i=$((i + 1))
                wait_n
              done
            done
          elif [ "${METHOD}" = "TENT" ]; then
            EPOCH=1
            if [ "${DATASET}" = "imagenetnoisy" ]; then
              lr=0.00025 #referred to the paper
            else
              lr=0.001
            fi
            #### Train with BN
            for dist in 1; do
              for TGT in $TGTS; do

                python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method ${METHOD} --tgt ${TGT} --model $MODEL --epoch $EPOCH ${CP} --seed $SEED \
                  --remove_cp --online --tgt_train_dist ${dist} --update_every_x ${update_every_x} --memory_size ${memory_size} \
                  --lr ${lr} --weight_decay ${weight_decay} \
                  --log_prefix ${LOG_PREFIX}_${SEED}_dist${dist} \
                  ${NOISY_TYPE} \
                  ${validation} \
                  2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &

                i=$((i + 1))
                wait_n
              done
            done
          elif [ "${METHOD}" = "LAME" ]; then
            EPOCH=1
            #### Train with BN
            for dist in 1; do
              for TGT in $TGTS; do

                python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method ${METHOD} --tgt ${TGT} --model $MODEL --epoch $EPOCH ${CP} --seed $SEED \
                  --remove_cp --online --tgt_train_dist ${dist} --update_every_x ${update_every_x} --memory_size ${memory_size} \
                  --lr ${lr} --weight_decay ${weight_decay} \
                  --log_prefix ${LOG_PREFIX}_${SEED}_dist${dist} \
                  ${NOISY_TYPE} \
                  ${validation} \
                  2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &

                i=$((i + 1))
                wait_n
              done
            done
          elif [ "${METHOD}" = "CoTTA" ]; then
            lr=0.001
            EPOCH=1

            if [ "${DATASET}" = "cifar10" ] || [ "${DATASET}" = "cifar10noisy" ]; then
              aug_threshold=0.92 #value reported from the official code
            elif [ "${DATASET}" = "cifar100" ] || [ "${DATASET}" = "cifar100noisy" ]; then
              aug_threshold=0.72 #value reported from the official code
            elif [ "${DATASET}" = "imagenetnoisy" ]; then
              aug_threshold=0.1 #value reported from the official code
            fi

            for dist in 1; do
#              CP=${CP_base}_${SEED}/cp/cp_last.pth.tar
              for TGT in $TGTS; do

                python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method ${METHOD} --tgt ${TGT} --model $MODEL --epoch $EPOCH ${CP} --seed $SEED \
                  --remove_cp --online --tgt_train_dist ${dist} --update_every_x ${update_every_x} --memory_size ${memory_size} \
                  --lr ${lr} --weight_decay ${weight_decay} \
                  --aug_threshold ${aug_threshold} \
                  --log_prefix ${LOG_PREFIX}_${SEED}_dist${dist} \
                  ${NOISY_TYPE} \
                  ${validation} \
                  2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &

                i=$((i + 1))
                wait_n
              done
            done
          elif [ "${METHOD}" = "SAR" ]; then
            EPOCH=1

            BATCH_SIZE=64 # For ImageNetC
            lr=0.00025 # For ImageNetC; from SAR paper: args.lr = (0.00025 / 64) * bs * 2 if bs < 32 else 0.00025

            #### Train with BN
            for dist in 1; do  # 1 for i.i.d; 6 for SAR imagenetc
              for TGT in $TGTS; do
                # --use_learned_stats --bn_momentum 0.01 should be turned off for original SAR.
                #  --batch-size ${BATCH_SIZE} --load_checkpoint_path ${CP}
                python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method ${METHOD} --tgt ${TGT} --model $MODEL ${CP} --epoch $EPOCH --seed $SEED \
                  --remove_cp --online --tgt_train_dist ${dist} --update_every_x ${update_every_x} --memory_size ${memory_size} \
                  --lr ${lr} --weight_decay ${weight_decay} \
                  --log_prefix "${LOG_PREFIX}_${SEED}_dist${dist}" \
                  ${NOISY_TYPE} \
                  ${validation} \
                  2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &

                i=$((i + 1))
                wait_n
              done
            done
          elif [ "${METHOD}" = "EATA" ]; then
            EPOCH=1

            if [ "${DATASET}" = "cifar10" ] || [ "${DATASET}" = "cifar10noisy" ]; then
              lr=0.005
              e_margin=0.92103 # 0.4*ln(10)
              d_margin=0.4
              fisher_alpha=1
            elif [ "${DATASET}" = "cifar100" ] || [ "${DATASET}" = "cifar100noisy" ]; then
              lr=0.005
              e_margin=1.84207 # 0.4*ln(100)
              d_margin=0.4
              fisher_alpha=1
            elif [ "${DATASET}" = "imagenetnoisy" ]; then
              lr=0.00025
              e_margin=2.76310 # 0.4*ln(1000)
              d_margin=0.05
              fisher_alpha=2000
            fi

            #### Train with BN
            for dist in 1; do  # 1 for i.i.d
              for TGT in $TGTS; do
                python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method ${METHOD} --tgt ${TGT} --model $MODEL ${CP} --epoch $EPOCH --seed $SEED \
                  --remove_cp --online --tgt_train_dist ${dist} --update_every_x ${update_every_x} --memory_size ${memory_size} \
                  --lr ${lr} --weight_decay ${weight_decay} \
                  --log_prefix "${LOG_PREFIX}_${SEED}_dist${dist}" \
                  --e_margin ${e_margin} --d_margin ${d_margin} --fisher_alpha ${fisher_alpha} \
                  ${NOISY_TYPE} \
                  ${validation} \
                  2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &

                i=$((i + 1))
                wait_n
              done
            done
          fi

        done
      done
    done
  done

  wait
}

test_time_adaptation
