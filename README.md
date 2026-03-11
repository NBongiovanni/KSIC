# KSIC
Koopman System Identification and Control with Visual Data

This repository contains the code associated with the paper on learning Koopman models from visual data.

The current implementation focuses on the **planar quadrotor system**, but the framework is designed to support different drone models and sensing modalities.

---

# Overview

The objective of this project is to learn a **Koopman representation of the system dynamics directly from visual observations**.

The learned model follows the linear latent dynamics:

```
z_{k+1} = A z_k + B u_k
```

where

- `z` is the latent observable state
- `u` is the control input
- `A` and `B` are learned matrices

The model architecture consists of:

1. **Encoder**

CNN + MLP mapping image observations to a latent state `z`.

2. **Linear Koopman dynamics**

Linear evolution of the latent state.

3. **Decoder**

MLP + transposed CNN reconstructing the image observations.

---

# Training a model

Example command to train a Koopman model from visual data:

```bash
python3 run_train_vision_model.py \
    --dynamics linear \
    --mode vision \
    --seed 1 \
    --id 1 \
    --geom_losses \
    --drone_dim 2
```

Results and trained models are stored in the `outputs/` directory.

All experiment parameters are defined in configuration files (`.yaml`).

---

# Command line arguments

## `--dynamics`

Type of dynamics model used in the latent space.

Two options are currently available:

| Value | Model |
|------|------|
| `linear` | Linear Koopman dynamics |
| `bilinear` | Bilinear Koopman dynamics |

### `linear`

The latent dynamics follow:

```
z_{k+1} = A z_k + B u_k
```

### `bilinear`

The latent dynamics include an additional bilinear interaction between the latent state and the control:

```
z_{k+1} = A z_k + B u_k + \sum_i u_i N_i z_k
```

where `A`, `B`, and `N_i` are learned matrices.
## `--mode`

Defines the input modality used for training.

Possible values:

| Mode | Description |
|-----|-----|
| `vision` | Learning from image observations |
| `sensors` | Learning from physical state measurements |

---

## `--seed`

Random seed used for reproducibility.

Example:

```
--seed 1
```

---

## `--id`

Identifier of the configuration file used for the experiment.

Example:

```
--id 1
```

This loads the configuration file:

```
configs/.../config_1.yaml
```

---

## `--geom_losses` / `--no-geom_losses`

Enable or disable geometric loss terms that enforce consistency between the latent representation and the drone state.

These losses improve the interpretability of the learned representation.

---

## `--drone_dim`

Dimension of the drone model.

| Value | System |
|-----|-----|
| `1` | 1D vertical drone |
| `2` | planar quadrotor |
| `3` | full 3D quadrotor |

Example (planar quadrotor):

```
x_dim = 6
u_dim = 2
```

---

# Configuration file

All experiment parameters are defined in YAML configuration files.

The configuration is divided into three main sections:

```
dataset_params
training_params
model_params
```

---

# Dataset parameters

## `name_dataset`

Name of the dataset used for training.

---

## `dt`

Sampling time between two consecutive frames.

---

## `dim`

Dimension of the drone model.

---

# Training parameters

## `num_epochs`

Number of training epochs.

---

## `loss_weights`

Weights applied to the different components of the loss function.

Typical loss terms include:

### Reconstruction loss

Measures how well the decoder reconstructs the input image.

### Prediction loss

Measures how well the latent dynamics predict future observations.

### Latent regularization / dimensional loss

Encourages structural constraints in the latent space.

The total loss is typically a weighted sum:

```
L = α_rec L_rec + α_pred L_pred + α_dim L_dim
```

---

## `checkpoint_every`

Frequency (in epochs) at which model checkpoints are saved.

---

## `grad_clip_max_norm`

Maximum gradient norm used for gradient clipping during training.

---

# Model parameters

## `dt`

Time discretization used for the Koopman dynamics.

---

## `z_dynamics.model`

Type of latent dynamics model.

Example:

```
linear
```

---

## `z_dynamics.z_dim`

Dimension of the latent state `z`.

This parameter controls the size of the Koopman observable space.

---

## `z_dynamics.x_dim`

Dimension of the physical state vector.

---

## `z_dynamics.u_dim`

Dimension of the control input.

---

# Dataset format

Each training sample contains:

```
(y_t, y_{t+1}, u_t)
```

where

- `y_t` : image at time `t`
- `y_{t+1}` : image at time `t+1`
- `u_t` : control input

The encoder receives the stacked images:

```
(batch_size, 2, 128, 128)
```

The network encodes the observation into the latent state:

```
z_t = encoder(y_t, y_{t+1})
```

---

# Training outputs

During training, the following directory structure is generated:

```
outputs/
    learning/
        vision/
            <run_id>/
                config.yaml
                checkpoints/
                tensorboard_logs/
```

Contents:

- `config.yaml` : configuration used for the experiment
- `checkpoints/` : saved model weights
- `tensorboard_logs/` : training metrics

---

# Visualization

Training curves can be visualized with TensorBoard:

```bash
tensorboard --logdir outputs
```

---

# Citation

If you use this repository, please cite the associated paper.