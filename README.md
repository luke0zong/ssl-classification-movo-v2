# dl09
Deep Learning Final Competition Team DL09

## code structure

## how to pickout bad samples?
- use existing models to select hard samples?
- will email TAs for detail

## model

### lightly
 - https://github.com/lightly-ai/lightly

### resnet

### swav
- https://github.com/facebookresearch/swav

### moco
- https://github.com/facebookresearch/moco
- https://medium.com/analytics-vidhya/simclr-with-less-computational-constraints-moco-v2-in-pytorch-3d8f3a8f8bf2

## Transform
- use code from assignment 2

## work sep
    - Yaowei: Run scripts and train model on gcp
    - Arthur: Create models and train.py

## TODO
### Yaowei:
1. moco-dim
2. solve cuda memory bug of pretrain_continue.sbatch
3. (optional) use 2 gpus

### Arthur:
1. add convert pretrain model to train model
2. train

### Future:
1. get_model