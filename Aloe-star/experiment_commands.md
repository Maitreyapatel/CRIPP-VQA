# This readme contains the experiments and all the results with insights if any.

### Exp1 (CIFAR10/100 resnet18 and mobilenet_v2 training)
1. HYDRA_FULL_ERROR=1 python train.py trainer.gpus=2 +trainer.strategy=ddp logger=wandb callbacks=wandb name=resnet18_cifar10
2. HYDRA_FULL_ERROR=1 python train.py trainer.gpus=2 +trainer.strategy=ddp logger=wandb callbacks=wandb name=resnet18_cifar10 model.model_name=mobilenetv2_100
3. python test.py ckpt_path=./logs/checkpoints/effnet_b7_ns_best.ckpt model.model_name=tf_efficientnet_b7_ns trainer.gpus=1



## Varified working commands:
1. `python train.py trainer.gpus=1`
2. `python train.py trainer.gpus=2 +trainer.strategy=ddp`
3. `HYDRA_FULL_ERROR=1 python train.py trainer.gpus=2 +trainer.strategy=ddp logger=wandb callbacks=wandb`
4. `HYDRA_FULL_ERROR=1 python train.py trainer.gpus=2 +trainer.strategy=ddp logger=wandb callbacks=wandb name=resnet18_cifar10`
5. `HYDRA_FULL_ERROR=1 python train.py trainer.gpus=2 +trainer.strategy=ddp logger=wandb callbacks=wandb name=resnet18_cifar10`