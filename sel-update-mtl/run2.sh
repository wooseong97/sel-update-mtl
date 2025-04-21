device_num=1
port_num=29501

CUDA_VISIBLE_DEVICES=$device_num python -m torch.distributed.launch --nproc_per_node=1 --master_port $port_num main.py --config_exp './configs/sel/nyud.yml' --run_mode train
# CUDA_VISIBLE_DEVICES=$device_num python -m torch.distributed.launch --nproc_per_node=1 --master_port $port_num main.py --config_exp './configs/sel/pascal.yml' --run_mode train
# CUDA_VISIBLE_DEVICES=$device_num python -m torch.distributed.launch --nproc_per_node=1 --master_port $port_num main.py --config_exp './configs/sel/taskonomy.yml' --run_mode train