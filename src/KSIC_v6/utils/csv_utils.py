import torch
import numpy as np


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().squeeze(0).numpy()


def save_dyn_matrices_to_csv(
        tensor: torch.Tensor,
        filename: str,
        col_names: list[str] | None = None,
        row_names: list[str] | None = None,
        include_index: bool = False,
        float_format: str = '%.2f') -> None:
    """
    Save a 2D torch.Tensor to a CSV file.

    Args:
        tensor: 2D torch.Tensor with shape [rows, cols].
        filename: Path to the CSV file to create.
        col_names: List of column names of length = tensor.shape[1].
                   If None, pandas auto-generates numeric names.
        row_names: List of row names of length = tensor.shape[0].
                   Used only if include_index is True.
        include_index: If True, write the row index to CSV.
        float_format: Format string for floats (e.g. '%.6f').
                      If None, pandas applies default formatting.

    Raises:
        ValueError: if tensor.dim() != 2, or if name lists have wrong
                    lengths.
    """
    # Vérification de la dimension
    if tensor.dim() != 2:
        raise ValueError(f"Le tenseur doit être 2D, or tensor.dim()={tensor.dim()}")
    # Transférer sur CPU si nécessaire, détacher du graphe

    # Conversion en numpy
    array = tensor.detach().cpu().numpy()
    n_rows, n_cols = array.shape

    # Vérification des noms de colonnes
    if col_names is not None:
        if len(col_names) != n_cols:
            raise ValueError(f"len(col_names)={len(col_names)} n'est pas égal au nombre de colonnes {n_cols}")
    # Vérification des noms de lignes
    if include_index and row_names is not None:
        if len(row_names) != n_rows:
            raise ValueError(f"len(row_names)={len(row_names)} n'est pas égal au nombre de lignes {n_rows}")

    # Créer le DataFrame
    df = pd.DataFrame(data=array, columns=col_names)
    df = df.round(2)

    # Gérer l’index si demandé
    if include_index:
        if row_names is not None:
            df.index = row_names
        # sinon, pandas utilisera l’index 0,1,2...
    else:
        # Si on ne veut pas inclure l’index, on écrira index=False dans to_csv
        pass

    # Créer le dossier si nécessaire
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

    # Écrire en CSV
    df.to_csv(filename, index=include_index, float_format=float_format)
    print(f"CSV généré avec succès : '{filename}' (shape={n_rows}×{n_cols})")
    return None


def save_eigenvectors_to_csv(
        tensor: torch.Tensor,
        filename: str,
        col_names: list[str] | None = None,
        row_names: list[str] | None = None,
        include_index: bool = False,
        float_format: str = '%.2f'
) -> None:
    """
    Exporte un torch.Tensor 2D dans un fichier CSV.
    Si le tenseur contient des nombres complexes, la fonction place
    la partie réelle et la partie imaginaire dans des cellules distinctes,
    générant ainsi 2 colonnes par colonne d'origine.

    Args:
        tensor: Tenseur 2D de forme [lignes, colonnes]. Peut être de type complexe.
        filename: Chemin vers le fichier CSV à créer.
        col_names: Liste de noms de colonnes de longueur tensor.shape[1].
                   Si le tenseur est complexe, chaque nom sera suffixed par '_real' et '_imag'.
                   Si None, les noms sont générés automatiquement.
        row_names: Liste de noms de lignes de longueur tensor.shape[0].
                   Utilisée uniquement si include_index est True.
        include_index: Si True, ajoute l’index des lignes dans le CSV.
        float_format: Chaîne de format pour les flottants (par exemple, '%.2f').

    Raises:
        ValueError: si tensor.dim() != 2, ou si les listes de noms n'ont pas la bonne longueur.
    """
    # Vérifier que le tenseur est bien 2D
    if tensor.dim() != 2:
        raise ValueError(f"Le tenseur doit être 2D, or tensor.dim() = {tensor.dim()}")

    # Transférer sur CPU si nécessaire et convertir en numpy
    array = tensor.cpu().numpy()
    n_rows, n_cols = array.shape

    # Vérification de la longueur de col_names
    if col_names is not None:
        if len(col_names) != n_cols:
            raise ValueError(f"len(col_names) = {len(col_names)} n'est pas égal au nombre de colonnes {n_cols}")

    # Vérification de la longueur de row_names si on souhaite inclure l'index
    if include_index and row_names is not None:
        if len(row_names) != n_rows:
            raise ValueError(f"len(row_names) = {len(row_names)} n'est pas égal au nombre de lignes {n_rows}")

    # Si le tableau est de nombres complexes, on sépare la partie réelle et imaginaire
    if np.iscomplexobj(array):
        # Extraire les parties réelle et imaginaire
        real_part = array.real
        imag_part = array.imag

        # Construire les nouveaux noms de colonnes
        new_col_names = []
        for j in range(n_cols):
            base_name = col_names[j] if col_names is not None else f"col{j}"
            new_col_names.extend([f"{base_name}_real", f"{base_name}_imag"])

        # Pour chaque colonne, empiler horizontalement la partie réelle et la partie imaginaire
        # Ceci crée une nouvelle matrice de forme (n_rows, 2*n_cols) où chaque paire correspond
        # aux valeurs réelle et imaginaire de la colonne originale.
        new_data = np.hstack([
            np.column_stack((real_part[:, j], imag_part[:, j]))
            for j in range(n_cols)
        ])

        df = pd.DataFrame(new_data, columns=new_col_names)
    else:
        # Cas classique pour un tenseur réel
        df = pd.DataFrame(data=array, columns=col_names)

    # Gestion de l'index si demandé
    if include_index:
        if row_names is not None:
            df.index = row_names
        # Sinon, pandas utilisera l’index par défaut (0, 1, 2, ...)

    # Créer le dossier parent si nécessaire
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

    # Écrire le DataFrame en CSV en appliquant le formatage spécifié pour les floats
    df.to_csv(filename, index=include_index, float_format=float_format)
    print(f"CSV généré avec succès : '{filename}' (shape={df.shape[0]}×{df.shape[1]})")


import os
import torch
import numpy as np
import pandas as pd


def save_eigenvalues_to_csv(
        tensor: torch.Tensor,
        filename: str,
        col_names: list[str] | None = None,
        row_names: list[str] | None = None,
        include_index: bool = False,
        float_format: str = '%.2f'
) -> None:
    """
    Exporte un torch.Tensor 2D dans un fichier CSV.

    Pour un tenseur réel, le CSV aura la même forme que le tenseur, avec éventuellement
    des noms de colonnes définis par `col_names`.

    Pour un tenseur complexe, la fonction aplatit le tenseur (en parcourant les éléments en
    ordre ligne par ligne) et crée un CSV à deux colonnes :
      - "Real" contenant la partie réelle
      - "Imag" contenant la partie imaginaire

    Si `include_index` est True et que `row_names` est fourni, la longueur de `row_names` doit
    correspondre à :
      - le nombre de lignes du tenseur réel, ou
      - le nombre total d'éléments (lignes × colonnes) pour un tenseur complexe.

    Args:
        tensor: Tenseur 2D de forme [lignes, colonnes]. Peut être complexe.
        filename: Chemin vers le fichier CSV à créer.
        col_names: Liste de noms pour les colonnes (non utilisé pour un tenseur complexe).
        row_names: Liste de noms de lignes.
        include_index: Si True, ajoute une colonne d'index dans le CSV.
        float_format: Format string pour les nombres à virgule (ex. '%.2f').

    Raises:
        ValueError: si tensor.dim() != 2, ou si les listes de noms ne correspondent pas aux dimensions.
    """
    # Vérification que le tenseur est bien 2D
    if tensor.dim() != 2:
        raise ValueError(f"Le tenseur doit être 2D, or tensor.dim() = {tensor.dim()}")

    # Conversion sur CPU et passage en tableau numpy
    array = tensor.cpu().numpy()
    n_rows, n_cols = array.shape

    if np.iscomplexobj(array):
        # Cas complexe : on aplatit le tableau pour obtenir un vecteur de longueur n_rows*n_cols
        real_flat = array.real.flatten()  # Partie réelle
        imag_flat = array.imag.flatten()  # Partie imaginaire

        # Vérification de row_names dans ce cas
        if include_index and row_names is not None:
            expected_length = n_rows * n_cols
            if len(row_names) != expected_length:
                raise ValueError(
                    f"len(row_names) = {len(row_names)} n'est pas égal au nombre d'éléments {expected_length} pour un tenseur complexe.")

        # Création du DataFrame avec deux colonnes
        df = pd.DataFrame({
            "Real": real_flat,
            "Imag": imag_flat
        })
    else:
        # Cas réel : vérification des longueurs si des noms de colonnes ou de lignes sont fournis
        if col_names is not None and len(col_names) != n_cols:
            raise ValueError(f"len(col_names) = {len(col_names)} n'est pas égal au nombre de colonnes {n_cols}")
        if include_index and row_names is not None and len(row_names) != n_rows:
            raise ValueError(f"len(row_names) = {len(row_names)} n'est pas égal au nombre de lignes {n_rows}")
        df = pd.DataFrame(data=array, columns=col_names)

    # Gestion de l'index, si demandé
    if include_index and row_names is not None:
        df.index = row_names

    # Création du dossier parent si nécessaire
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

    # Écriture du CSV avec le formatage des floats
    df.to_csv(filename, index=include_index, float_format=float_format)
    print(f"CSV généré avec succès : '{filename}' (shape={df.shape[0]}×{df.shape[1]})")

