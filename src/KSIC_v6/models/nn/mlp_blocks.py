from torch import nn

def build_mlp(
        dim_in: int,
        dim_hidden: int,
        dim_out: int,
        num_hidden_layers: int,
        act: str,
) -> nn.Sequential:

    if act == "tanh":
        act_function = nn.Tanh()
    elif act == "relu":
        act_function = nn.ReLU()
    else:
        raise ValueError(f"Unknown activation function {act}")

    layers = []
    layers.append(nn.Linear(dim_in, dim_hidden))
    layers.append(act_function)

    for _ in range(num_hidden_layers):
        layers.append(nn.Linear(dim_hidden, dim_hidden))
        layers.append(act_function)

    layers.append(nn.Linear(dim_hidden, dim_out))
    return nn.Sequential(*layers)