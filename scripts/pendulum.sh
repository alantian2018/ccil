
#!/bin/bash
TASK='pendulum_disc'
SEED='41'
NOISE_BC='0.0001'

./scripts/train_ccil.sh "${TASK}" "${SEED}" "${NOISE_BC}"

# Add ablation to train dynamics model without enforcing Lipschitz continuity
MODEL='none'
for task in $TASK
    do
    for seed in $SEED
        do
            for model in $MODEL
                do
                    echo "Training dynamics model without enforcing Lipschitz continuity for ${task} with seed ${seed}"
                    python correct_il/train_dynamics_model.py config/${task}.yml output.location output/${task} seed ${seed} dynamics.lipschitz_type ${model}
                    echo "Generating augmentation labels for ${task} with seed ${seed}"
                    python correct_il/gen_aug_label.py config/${task}.yml output.location output/${task} seed ${seed} dynamics.lipschitz_type ${model}
                    echo "Training BC policy for ${task} with seed ${seed}"
                    python correct_il/train_bc_policy.py config/${task}.yml output.location output/${task} seed ${seed} dynamics.lipschitz_type ${model}
                    echo "Evaluating BC policy for ${task} with seed ${seed}"
                    python correct_il/eval_bc_policy.py config/${task}.yml output.location output/${task} seed ${seed} dynamics.lipschitz_type ${model}
            done
        done
    done