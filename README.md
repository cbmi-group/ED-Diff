# ED-Diff

![framwork](./figs/framwork.jpg)

Implementation for the paper  "Diffusion Prior-based Zero-Shot Denoising for Real-World Fluorescence Microscopy Images via Encoder-Decoder Guided Sampling".

We sincerely thank the project [improved-diffusion](https://github.com/openai/improved-diffusion). The portion of ED-Diff related to diffusion models is based on modifications to the code from this project. Consequently, ED-Diff also supports direct usage of models pre-trained using improved-diffusion (or other projects built upon improved-diffusion).

## Environment

* python >= 3.6
* pytorch >= 1.8.0

## Preparation

We provide several pre-trained models and image samples for testing purposes, which can be accessed via the [link](https://drive.google.com/drive/folders/17_3fs9Ratmy_tObAUgrcpFO8jheeidch?usp=drive_link). Additionally, we include the original links to the open-source datasets used in the paper to facilitate convenient downloading:

* [CARE](https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.FDFZOF)
* [3dRCAN](https://zenodo.org/record/4624364#.Y6pszdVByyB)
* [FMDD](https://gigadb.org/dataset/100888)

For pre-training and inference on custom datasets, the following recommendations are suggested:

- Ensure that the noisy images and their corresponding reference images are stored in separate folders while maintaining identical filenames.
- All files default to the `.tif` extension; if other image formats are desired, you should refer to `datasets.py` to construct the Dataset accordingly.

## Usage

### Pretraining

If you wish to pre-train a diffusion model on a custom dataset, you can refer to the following format to write a `.sh` script:

```shell
SCRIPT_DIR=$(dirname "$(realpath "$0")")

CUDA_VISIBLE_DEVICES=0 \
OPENAI_LOGDIR="$SCRIPT_DIR/checkpoints/Custom_256_64_3_1000_1e-4" \
python pretrain.py \
--data_dir datasets/Trib/custom_dataset \
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
```

The key parameters are as follows:

- **CUDA_VISIBLE_DEVICES**: GPU index
- **OPENAI_LOGDIR**: Path to save checkpoints
- **data_dir**: High Signal-to-Noise Ratio Image Folder for Pre-training

### Inference

To perform inference on a specific image, you can refer to the following format to write a `.sh` script:

```shell
python inference.py \
--device 0 \
--model_path "checkpoints/care_model.pt" \
--data_pth "datasets/raw/care_trib.tif" \
--gt_pth "datasets/gt/care_trib.tif" \
--save_dir "results/exp" \
--ae_iters 20 \
--ae_lr 0.01 \
--sigma 5 \
--omega 0.005 \
--cropped_size 256 \
--num_channels 64 \
--num_res_blocks 3 \
--diffusion_steps 1000 \
--noise_schedule linear \
--timestep_respacing ddim50 \
--use_ddim
```

The key parameters are as follows:

- **device**: GPU index
- **model_path**: Path to the pre-trained model
- **data_pth**: Path to the target noisy image
- **gt_pth**: Path to the reference image of the target image (delete if no reference image is available)
- **save_dir**: Path to save the denoised result
- **ae_iters**: Maximum number of iterations for NiTM
- **ae_lr**: Maximum learning rate for NiTM
- **cropped_size**: Image crop size during NiTM training

Additionally, we also support batch inference. Simply invoke `batch_inference.py` and adjust the parameters `data_path `and `gt_path `to `data_dir `and `gt_dir`, respectively. Refer to the following example:

```shell
python batch_inference.py \
--device 0 \
--model_path "checkpoints/all_model.pt" \
--data_dir "datasets/raw" \
--gt_dir "datasets/gt"
--save_dir "results/exp_batch" \
--ae_iters 20 \
--ae_lr 0.01 \
--sigma 5 \
--omega 0.005 \
--cropped_size 256 \
--num_channels 64 \
--num_res_blocks 3 \
--diffusion_steps 1000 \
--noise_schedule linear \
--timestep_respacing ddim50 \
--use_ddim
```

### Result reproduction

To intuitively display the parameters used in the experiments and facilitate reproduction, we have provided the scripts for execution within the scripts folder. Please refer to the corresponding [guide](scripts/README.md) for usage instructions.

## Results

![results](./figs/results.jpg)
