# KSIC
Koopman system identification and control with visual data
This version works only for the planar quadrotor system.

## How to launch a learning of a model:
- run : python3 run_train_vision_model.py --dynamics linear --mode vision --seed 1 --id 1 --geom_losses --drone_dim 2 
- results and models will be recorded in the directory results
- configuration defined in the configuration file (.json)

## Main parameters to be set in the config file
- z_dim: the dimension of the observable space
- alphas: weights of the components of the losses, in the order: loss_rec, loss_pred, loss_dim
- n_epoch: number of epochs of the training,
- name_dataset: name of the dataset to be used,
