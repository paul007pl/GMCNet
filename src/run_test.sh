partition=ntu
job_name=test_reg
gpus=1
gpus_per_node=$((${gpus}<8?${gpus}:8))
#g=$((${gpus}<8?${gpus}:8))

export TORCH_EXTENSIONS_DIR=/mnt/lustre/ntu004/tmp
srun -u --partition=${partition} --job-name=${job_name} -w SH-IDC1-10-142-16-18 \
    --gres=gpu:${gpus_per_node} --ntasks=${gpus} --ntasks-per-node=${gpus_per_node} \
    python test.py --config cfgs/$1.yaml