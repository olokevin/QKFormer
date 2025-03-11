# List of .yml config files
config_files=("configs/BP_full.yml" "configs/ZO_full.yml" "configs/BP_last_layer.yml" "configs/BP_first_layer.yml" "configs/ZO_ours.yml")

# Initialize a counter for the GPU index
gpu=0

# Iterate through the list of config files
for config in "${config_files[@]}"
do
  # Run the experiment with the specified GPU, config file, and corruption type
  CUDA_VISIBLE_DEVICES=$gpu python train.py -c $config
done