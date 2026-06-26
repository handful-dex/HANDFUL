#!/bin/bash

# experiments
seeds=(666)

for seed in "${seeds[@]}"
do
    python train.py \
        --env_id="xArm7-v1" \
        --num_envs=32 \
        --utd=0.5 \
        --buffer_size=500000 \
        --total_timesteps=5000000 \
        --eval_freq=50000 \
        --control-mode="pd_joint_delta_pos" \
        --seed=$seed
done
