CUDA_VISIBLE_DEVICES=0 \
OPENAI_LOGDIR="/xxx/checkpoints/CARE-All_256_64_3_1000_linear_1e-4_8_norm-patch" \
python pretrain.py \
--data_dir datasets/CARE/gt \
--image_size 256 \
--num_channels 64 \
--num_res_blocks 3 \
--diffusion_steps 1000 \
--noise_schedule linear \
--lr 1e-4 \
--batch_size 64 \
--microbatch 8 \
--save_interval 500 \
--max_steps 2000 \
--dataset_type TIFF