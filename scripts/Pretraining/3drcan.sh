CUDA_VISIBLE_DEVICES=0 \
OPENAI_LOGDIR="/xxx/checkpoints/3DRCAN-All_256_64_3_1000_linear_1e-4_8" \
python pretrain.py \
--data_dir datasets/3drcan/gt \
--image_size 256 \
--num_channels 64 \
--num_res_blocks 3 \
--diffusion_steps 1000 \
--noise_schedule linear \
--lr 1e-4 \
--batch_size 64 \
--microbatch 8 \
--save_interval 1000 \
--max_steps 10000 \
--dataset_type TIFF