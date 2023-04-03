#!/bin/bash

#SBATCH -J bar
#SBATCH -p DGXA100
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --gres-flag=enforce-binding

#python train.py --model_name y_net_gen --dataset UMN --n_classes 2 --image_dir /hpcstor6/scratch01/y/yanankristin.qi001/ynet/UMNData
#python train.py --model_name unet --dataset Duke
#python train.py --model_name y_net_gen --dataset Duke
#python train_imagenet.py  --supervised   --epochs 1000  --pretrained --if-LARS
source att/bin/activate
#python main.py --epochs 2000 --batch-size 256
python  evaluate.py  './checkpoint/lincls/my50224resfinal__redo_norm.pth'  --epochs 250 --batch-size 256 --lr-classifier 0.3
 