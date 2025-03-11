# CUDA_VISIBLE_DEVICES=0 python train.py -c ZO.yml
# CUDA_VISIBLE_DEVICES=0 nohup python train.py -c ZO.yml >/dev/null 2>&1 &

# CUDA_VISIBLE_DEVICES=0 python train.py -c ZO.yml --eval

##### Different Corruption Types
corruption_types=(gaussian_noise shot_noise impulse_noise speckle_noise gaussian_blur defocus_blur glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression saturate spatter)
# corruption_types=(gaussian_noise shot_noise impulse_noise speckle_noise gaussian_blur defocus_blur glass_blur motion_blur zoom_blur) 
# corruption_types=(snow frost fog brightness contrast elastic_transform pixelate jpeg_compression saturate spatter)

# Initialize a counter for the GPU index
# gpu_index=0
gpu=0

# Iterate through the list of corruption types
for corruption in "${corruption_types[@]}"
do
  # Set the GPU to be used for the current experiment
  # gpu=$((gpu_index % 6))
  # gpu=$((gpu_index % 3 + 3))
  
  # Run the experiment with the specified GPU and corruption type
  CUDA_VISIBLE_DEVICES=$gpu python train.py -c ZO.yml --corruption_type $corruption
  # CUDA_VISIBLE_DEVICES=$gpu nohup python train.py -c ZO.yml --corruption_type $corruption >/dev/null 2>&1 &
  
  # Increment the GPU index for the next experiment
  gpu_index=$((gpu_index + 1))
done