from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

VISION_TAGS = {
    "y_pred":     "01_y_pred",
    "y_rec":      "02_y_rec",
    "z":          "03_z",
    "c":          "04_c",
    "horizontal": "05_horizontal",
    "vertical":   "06_vertical",
    "angle":      "07_angle",
    "iou":        "08_iou",
}

SENSOR_TAGS = {
    "pred_x":         "01_x_pred",
    "rec":            "02_x_rec",
    "pred_z":         "03_z",
    "pred_position":  "04_position",
}

def log_losses_console(
        modality: str,
        epoch: int,
        phase_losses: dict,
        num_val_datasets: int
) -> None:
    """
    Print the new values of the train losses and validation losses on the terminal.
    """
    if modality == "vision":
        tags = VISION_TAGS
    elif modality == "sensor":
        tags = SENSOR_TAGS
    else:
        raise ValueError("Problem in the modality name")

    phase_names = ["train"] + [f"val_{i}" for i in range(1, num_val_datasets + 1)]

    print(f"Epoch: {epoch}")
    for phase in phase_names:
        global_loss, sub = phase_losses[phase]
        lines = [f"{phase} loss: {global_loss:.2e}", "\t  pred :"]
        for key, tag in tags.items():
            val = sub[key]
            lines.append(f"\t    {tag:<9}: {val:.2e}")
        print("\n".join(lines))
    print("")


def log_losses_tensorboard(
        modality: str,
        writer: SummaryWriter,
        epoch: int,
        phase_losses: dict,
        num_val_datasets: int
) -> None:
    """
    Log all the losses using TensorBoard
    """
    if modality == "vision":
        tags = VISION_TAGS
    elif modality == "sensor":
        tags = SENSOR_TAGS
    else:
        raise ValueError("Problem in the modality name")
    phase_idx = 2
    phase_names = ["train"] + [f"val_{i}" for i in range(1, num_val_datasets + 1)]

    for phase in phase_names:
        global_loss, sub = phase_losses[phase]
        writer.add_scalar(f"01_total/{phase}", global_loss, epoch)
        tag_base = f"0{phase_idx}_{phase}/"
        for key, tag in tags.items():
            writer.add_scalar(tag_base + tag, sub[key], epoch)
        phase_idx = phase_idx + 1


def log_on_tb(
        writer: SummaryWriter,
        tag: str,
        name: str,
        quantity,
        epoch: int
) -> None:
    if writer is not None:
        writer.add_scalar(
            tag + name,
            quantity,
            epoch
        )


def log_training_state_vision(
        writer: SummaryWriter,
        optimizer: Optimizer,
        base: dict,
        effective_weight,
        total_norm: float,
        epoch: int,
        phase_index: int
) -> None:

    tag = "08_training_state/"
    quantity = effective_weight(base["c"], "c")
    log_on_tb(writer, tag, "c_effective",quantity, epoch)
    quantity = effective_weight(base["a"], "a")
    log_on_tb(writer, tag, "a_effective",quantity, epoch)
    log_on_tb(writer, tag, "phase_index",phase_index, epoch)

    # === Log du learning rate ===
    for i, pg in enumerate(optimizer.param_groups):
        # essaie de récupérer le nom si tu l'as mis dans pg["name"], sinon fallback à l'index
        group_name = pg.get("name", f"group_{i}")
        lr = pg["lr"]
        log_on_tb(writer, tag, f"lr/{group_name}", lr, epoch)

    writer.add_scalar(tag + "grad_norm", total_norm, epoch)

def log_training_state_sensors(
        writer: SummaryWriter,
        optimizer: Optimizer,
        total_norm: float,
        epoch: int
) -> None:
    tag = "08_training_state/"
    for i, pg in enumerate(optimizer.param_groups):
        group_name = pg.get("name", f"group_{i}")
        writer.add_scalar(tag + f"lr/{group_name}", pg["lr"], epoch)
    writer.add_scalar(tag + "grad_norm", total_norm, epoch)