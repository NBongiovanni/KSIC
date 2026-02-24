# KSIC_v8
Koopman system identification and control with visual data
This version works only for the planar quadrotor system.

## do-mpc Sandbox
Ce dossier contient des scripts exploratoires (« sandbox ») pour tester des fonctionnalités spécifiques de la bibliothèque do-mpc.  
Ces fichiers ne font pas partie du workflow CI/CD ni du packaging.

## How to launch a learning of a model:
- run : python3 train.py --config <name_json_file>
- results and models will be recorded in the directory results
- configuration defined in the configuration file (.json)

## Main parameters to be set in the config file
- z_dim: the dimension of the observable space
- alphas: weights of the components of the losses, in the order: less_rec, loss_pred, loss_dim
- n_epoch: number of epochs of the training,
- name_dataset: name of the dataset to be used,
