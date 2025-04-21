device_num=0
port_num=29500

CUDA_VISIBLE_DEVICES=$device_num python -m torch.distributed.launch --nproc_per_node=1 --master_port $port_num main.py --config_exp './configs/baseline/nyud.yml' --run_mode train
# CUDA_VISIBLE_DEVICES=$device_num python -m torch.distributed.launch --nproc_per_node=1 --master_port $port_num main.py --config_exp './configs/baseline/pascal.yml' --run_mode train
# CUDA_VISIBLE_DEVICES=$device_num python -m torch.distributed.launch --nproc_per_node=1 --master_port $port_num main.py --config_exp './configs/baseline/taskonomy.yml' --run_mode train