# Improved Local-Feature-Based Few-Shot Learning with Sinkhorn Metrics

This is the code repository for the paper under review "Improved Local-Feature-Based Few-Shot Learning with Sinkhorn Metrics" in PyTorch.

Currently we only provide the trained model parameters and evaluation code with the Sinkhorn Metrics. Code for training will be released soon.

## Single-domain Few-shot Learning Results

For single-domain FSL, we present the results on 3 FSL datasets with ResNet-12 (Same as [this repo](https://github.com/kjunelee/MetaOptNet)). Average accuracy over 2000 random tasks and 95% confidence intervals are given.

|    Datasets    | 1-Shot 5-Way | 5-Shot 5-Way |
|  miniImagenet  | 69.48+-0.46  | 84.51+-0.30  |
| tieredImagenet | 71.71+-0.49  | 85.74+-0.33  |
|      cub       | 84.11+-0.39  | 93.61+-0.19  |

## Cross-domain Few-shot Learning Results

For cross-domain FSL, we present results of training on miniImagenet and testing on 2 FSL datasets with ResNet-10 (Same as [this repo](https://github.com/mileyan/simple_shot)). Average accuracy over 2000 random tasks and 95% confidence intervals are given.

|   Datasets  | 1-Shot 5-Way | 5-Shot 5-Way |
|   mini2cub  | 50.56+-0.45  | 71.04+-0.39  |
|   mini2cars | 37.95+-0.38  | 53.59+-0.42  |

## Prerequisites

The following packages are required to run the scripts:

- PyTorch-1.2

- torchvision-0.4

- numpy-1.18.1

- PIL-7.1.2

- tqdm-4.46.0

- json-2.0.9

## Dataset
For the miniImagenet, cub, cars, places365, and plantae datasets, we follow the [CrossDomainFewShot repo](https://github.com/hytseng0509/CrossDomainFewShot) to download the data.
For the tieredImagenet datasets, we follow the [DeepEMD repo](https://github.com/icoz69/DeepEMD) to download, which provides images of size 224x224.
All the datasets are pre-processed following the [CrossDomainFewShot repo](https://github.com/hytseng0509/CrossDomainFewShot). One json file of each split(base, val, novel) is generated in /root_path/dataset_name, containing the paths of all the images and labels. The images are stored in /root_path/dataset_name/source. 

## Trained weights
Please download the trained weights from the [this link](https://drive.google.com/file/d/1csSTAtTtWK3jiiRUzJPuZJoLpoaU5GoW/view?usp=sharing) , each of which is a .pth file. Please provide the model path to the argument `--trained_model_path` when running the test.py file.

## Code Structures
 - `test.py`: The main file of the code.
 - `FewShotLearning/methods`: Codes related to the model, including the SM based FSL model and the backbone.
 - `FewShotLearning/dataloader`: Codes related to the data.

## Arguments
The 'test.py' takes the following arguments from command line (for details please check `test.py`):

**Arguments related to FSL task**
- `test_n_way`: The number of classes in a few-shot task during meta-test, default to `5`.

- `test_n_shot`: The number of labeled data of each class in a few-shot task during meta-test, default to `1`.

- `test_n_query`: The number of instances in each class to evaluate during meta-test, default to `15`.

**Arguments related to data loading**
- `base_data_path`: Root path to save the dataset.

- `test_dataset_name`: Option for the dataset (`miniImagenet/cub/tieredImagenet/cars`), default to `miniImagenet`.

- `image_size`: The size of the loaded image (`84/112`),  default to `112`.

- `test_seed`: Random seed to control the data loading process, default to `111`.

- `test_epoch_size`: The number of tasks for evaluation, default to `2000`.

- `fewshot_test_batch`: The number of tasks loaded in one iteration, defualt to `4`.

**Arguments related to model**
- `trained_model_path`: The path to save the trained model.

- `model`: The types of backbone (only `ResNet12/10` supported in this version, ResNet12 for single-domain FSL, ResNet10 for cross-domain FSL), defualt to `ResNet12`.

- `feature_scale`: Indicate the number of local features (`1/2`, `1` means smaller number and `2` means larger number), default to `2`.

- `seed`: Random seed to control the model, default to `0`.

- `drop_rate`: The dropout rate, default to `0.0`.

- `drop_block`: Whether to use the drop block instead of dropout (`True/False`), default to `False`.

- `dropblock_size`: The size of drop block, default to `5`. When `drop_block` is set to `False`, this argument has no effect.

**Arguments related to Sinkhorn Metrics** 
- `thresh`: The stop condition of the SM iteration, default to `0.01`.

- `maxiters`: The maximum number of the SM iteration, default to `1000`.

- `lam`: The scale factor of regularization term for the ROT (`100` for 1-shot, `80` for 5-shot), default to `100`.

**Other Arguments** 
- `gpu_devices`: The index of GPU(s) to use (`0/1/0,1`), default to `0`.

## Input and output scripts

#########################
To test the single-domain 5-way 1-shot model on miniImagenet:

    $ python test.py  --model ResNet12 --test_dataset_name miniImagenet --test_seed 111 --image_size 112 --feature_scale 2 --test_n_way 5 --test_n_shot 1 --test_n_query 15 --test_epoch_size 2000 --fewshot_test_batch 4 --thresh 0.01 --maxiters 1000 --lam 100 --trained_model_path /data_disk/deeplearning/trained_model/single_domain/miniImagenet/1-shot/ResNet12-miniImagenet-1shot.pth --gpu_devices 1

which will give:

    $ 20 cats, 12000 images.
    $ Running model ProtoNet_Sinkhorn
    $ testing: 100%|████████████████| 500/500 [08:55<00:00,  1.07s/it, curr_acc=0.695]
    $ ###########Test accuracy 69.48%, std 0.46%#############
#########################

#########################
To test the single-domain 5-way 5-shot model on miniImagenet:
    
    $ python test.py  --model ResNet12 --test_dataset_name miniImagenet --test_seed 111 --image_size 112 --feature_scale 2 --test_n_way 5 --test_n_shot 5 --test_n_query 15 --test_epoch_size 2000 --fewshot_test_batch 4 --thresh 0.01 --maxiters 1000 --lam 80  --trained_model_path /data_disk/deeplearning/trained_model/single_domain/miniImagenet/5-shot/ResNet12-miniImagenet-5shot.pth --gpu_devices 1

which will give:

    $ 20 cats, 12000 images.
    $ Running model ProtoNet_Sinkhorn
    $ testing: 100%|████████████████| 500/500 [08:32<00:00,  1.03s/it, curr_acc=0.845]
    $ ###########Test accuracy 84.51%, std 0.30%#############
#########################

#########################
To test the single-domain 5-way 1-shot model on tieredImagenet:

    $ python test.py  --model ResNet12 --test_dataset_name tieredImagenet --test_seed 126 --image_size 112 --feature_scale 2 --test_n_way 5 --test_n_shot 1 --test_n_query 15 --test_epoch_size 2000 --fewshot_test_batch 4 --thresh 0.01 --maxiters 1000 --lam 100 --trained_model_path /data_disk/deeplearning/trained_model/single_domain/tieredImagenet/1-shot/ResNet12-tieredImagenet-1shot.pth --gpu_devices 1

which will give:

    $ 160 cats, 206209 images.
    $ Running model ProtoNet_Sinkhorn
    $ testing: 100%|████████████████| 500/500 [09:32<00:00,  1.15s/it, curr_acc=0.717]
    $ ###########Test accuracy 71.71%, std 0.49%#############
#########################    

#########################
To test the single-domain 5-way 5-shot model on tieredImagenet:

    $ python test.py  --model ResNet12 --test_dataset_name tieredImagenet --test_seed 135 --image_size 112 --feature_scale 2 --test_n_way 5 --test_n_shot 5 --test_n_query 15 --test_epoch_size 2000 --fewshot_test_batch 4 --thresh 0.01 --maxiters 1000 --lam 80  --trained_model_path /data_disk/deeplearning/trained_model/single_domain/tieredImagenet/5-shot/ResNet12-tieredImagenet-5shot.pth --gpu_devices 0

which will give:

    $ 160 cats, 206209 images.
    $ Running model ProtoNet_Sinkhorn
    $ testing: 100%|████████████████| 500/500 [07:57<00:00,  1.05it/s, curr_acc=0.857]
    $ ###########Test accuracy 85.74%, std 0.33%#############
#########################  

######################### 
To test the single-domain 5-way 1-shot model on cub:

    $ python test.py  --model ResNet12 --test_dataset_name cub --test_seed 111 --image_size 112 --feature_scale 1 --test_n_way 5 --test_n_shot 1 --test_n_query 15 --test_epoch_size 2000 --fewshot_test_batch 4 --thresh 0.01 --maxiters 1000 --lam 100 --trained_model_path /data_disk/deeplearning/trained_model/single_domain/cub/1-shot/ResNet12-cub-1shot.pth --gpu_devices 1

which will give:

    $ 50 cats, 2953 images.
    $ Running model ProtoNet_Sinkhorn
    $ testing: 100%|████████████████| 500/500 [05:11<00:00,  1.61it/s, curr_acc=0.841]
    $ ###########Test accuracy 84.11%, std 0.39%#############
######################### 

#########################  
To test the single-domain 5-way 5-shot model on cub:

    $ python test.py  --model ResNet12 --test_dataset_name cub --test_seed 111 --image_size 112 --feature_scale 1 --test_n_way 5 --test_n_shot 5 --test_n_query 15 --test_epoch_size 2000 --fewshot_test_batch 4 --thresh 0.01 --maxiters 1000 --lam 80  --trained_model_path /data_disk/deeplearning/trained_model/single_domain/cub/5-shot/ResNet12-cub-5shot.pth --gpu_devices 0

which will give:

    $ 50 cats, 2953 images.
    $ Running model ProtoNet_Sinkhorn
    $ testing: 100%|████████████████| 500/500 [05:46<00:00,  1.44it/s, curr_acc=0.936]
    $ ###########Test accuracy 93.62%, std 0.19%#############
######################### 

#########################
To test the cross-domain 5-way 1-shot model on mini2cub:

    $ python test.py  --model ResNet10 --test_dataset_name cub --test_seed 111 --image_size 84 --feature_scale 1 --test_n_way 5 --test_n_shot 1 --test_n_query 15 --test_epoch_size 2000 --fewshot_test_batch 4 --thresh 0.01 --maxiters 1000 --lam 100 --trained_model_path /data_disk/deeplearning/trained_model/cross_domain/mini2cub/1-shot/ResNet10-mini2cub-1shot.pth --gpu_devices 0
which will give:

    $ 50 cats, 2953 images.
    $ Running model ProtoNet_Sinkhorn
    $ testing: 100%|████████████████| 500/500 [04:32<00:00,  1.83it/s, curr_acc=0.506]
    $ ##########Test accuracy 50.56%, std 0.45%#############
#########################

#########################
To test the cross-domain 5-way 5-shot model on mini2cub:

    $ python test.py  --model ResNet10 --test_dataset_name cub --test_seed 111 --image_size 84 --feature_scale 1 --test_n_way 5 --test_n_shot 5 --test_n_query 15 --test_epoch_size 2000 --fewshot_test_batch 4 --thresh 0.01 --maxiters 1000 --lam 80  --trained_model_path /data_disk/deeplearning/trained_model/cross_domain/mini2cub/5-shot/ResNet10-mini2cub-5shot.pth --gpu_devices 1

which will give:

    $ 50 cats, 2953 images.
    $ Running model ProtoNet_Sinkhorn
    $ testing: 100%|████████████████| 500/500 [03:36<00:00,  2.31it/s, curr_acc=0.711]
    $ ###########Test accuracy 71.10%, std 0.38%#############
#########################

#########################
To test the cross-domain 5-way 1-shot model on mini2cars:

    $ python test.py  --model ResNet10 --test_dataset_name cars --test_seed 111 --image_size 84 --feature_scale 1 --test_n_way 5 --test_n_shot 1 --test_n_query 15 --test_epoch_size 2000 --fewshot_test_batch 4 --thresh 0.01 --maxiters 1000 --lam 100 --trained_model_path /data_disk/deeplearning/trained_model/cross_domain/mini2cars/1-shot/ResNet10-mini2cars-1shot.pth --gpu_devices 1

which will give:

    $ 49 cats, 2027 images.
    $ Running model ProtoNet_Sinkhorn
    $ testing: 100%|████████████████| 500/500 [08:56<00:00,  1.07s/it, curr_acc=0.379]
    $ ###########Test accuracy 37.95%, std 0.38%#############
#########################

#########################
To test the cross-domain 5-way 5-shot model on mini2cars:

    $ python test.py  --model ResNet10 --test_dataset_name cars --test_seed 111 --image_size 84 --feature_scale 1 --test_n_way 5 --test_n_shot 5 --test_n_query 15 --test_epoch_size 2000 --fewshot_test_batch 4 --thresh 0.01 --maxiters 1000 --lam 80  --trained_model_path /data_disk/deeplearning/trained_model/cross_domain/mini2cars/5-shot/ResNet10-mini2cars-5shot.pth --gpu_devices 1

which will give:

    $ 49 cats, 2027 images.
    $ Running model ProtoNet_Sinkhorn
    $ testing: 100%|████████████████| 500/500 [09:26<00:00,  1.13s/it, curr_acc=0.536]
    $ ###########Test accuracy 53.58%, std 0.42%#############
#########################

## Acknowledgment
We thank the following repos providing helpful components/functions in our work.
- [CAN](https://github.com/kjunelee/MetaOptNet/)
- [CrossDomainFewShot](https://github.com/hytseng0509/CrossDomainFewShot)
- [MetaOptNet](https://github.com/kjunelee/MetaOptNet/)

