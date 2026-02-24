from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from KSIC_v6.utils import build_relative_dataset_path, find_project_root
from .dataset_params import DatasetParams

class ImageProcessorMmap:
    def __init__(self, params: DatasetParams, phase: str):
        super().__init__()
        self.params = params
        self.phase = phase
        self.dataset_states = None
        self.dataset_inputs = None
        self.dataset_idx = None
        self.traj_len = self.params.train["num_steps_loaded"]

    def pipeline(self, num_simulations: int, im_size: int, num_steps_pred: int) -> None:
        root_dir = find_project_root()
        save_dir = root_dir / self._define_save_path()
        save_dir.mkdir(parents=True, exist_ok=True)
        traj_len = self.traj_len
        H = W = im_size

        # Vues selon la dimension
        if self.params.drone_dim == 3:
            views = ["left", "right"]
        else:
            views = [None]  # mono-vue
        V = len(views)
        print(f"[{self.phase}] Conversion of the PNG files in a unique numpy array")

        # 1) memmap pour les images
        im_path = save_dir / "im_dataset_memmap.dat"
        im_dataset = np.memmap(
            im_path,
            dtype=np.uint8,
            mode="w+",
            shape=(num_simulations, traj_len, V, H, W),
        )
        name_png_dir = self._define_png_dir()

        def worker_load(args):
            i, j, v_idx, path = args
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError(f"Image not found: {path}")
            img_resized = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            return i, j, v_idx, img_resized

        paths = []
        for i in range(num_simulations):
            for j in range(traj_len):
                for v_idx, v in enumerate(views):
                    if v is None:
                        p = name_png_dir / f"traj_{i}" / f"step_{j}.png"
                    else:
                        p = name_png_dir / f"traj_{i}" / f"{v}" / f"step_{j}.png"
                    paths.append((i, j, v_idx, p))

        nb_threads = min(8, os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=nb_threads) as executor:
            futures = {executor.submit(worker_load, tup): tup[:3] for tup in paths}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Chargement"):
                i, j, v_idx = futures[future]
                try:
                    _, _, _, img_proc = future.result()
                    im_dataset[i, j, v_idx, :, :] = img_proc
                except Exception as exc:
                    print(f"Erreur traitement image traj={i}, step={j}, view={v_idx} : {exc}")

        im_dataset.flush()
        del im_dataset

        # 3) Reshape + construction des paires (t, t+1)
        self.reorganise_dataset_mmap(
            im_path,
            num_simulations,
            traj_len,
            num_steps_pred,
            im_size,
            save_dir,
            num_views=V,
        )

    def _define_png_dir(self) -> Path:
        root_dir = find_project_root()
        version_dir = str(self.params.dataset_version)
        dim_dir = f"{self.params.drone_dim}d"
        return root_dir / "datasets" / "raw_images" / dim_dir / version_dir / self.phase

    def _define_save_path(self) -> Path:
        return build_relative_dataset_path(
            "vision",
            self.params.drone_dim,
            str(self.params.dataset_version),
            self.phase
        )

    def reorganise_dataset_mmap(
            self,
            im_path: Path,
            num_simulations: int,
            traj_len: int,
            seq_len: int,
            resolution: int,
            save_dir: Path,
            num_views: int = 1,
    ) -> None:
        print("Dataset reshape (memmap / par trajectoire)")
        total_pairs, num_seq, H, W = self._compute_shapes(
            num_simulations,
            traj_len,
            seq_len,
            resolution
        )
        im_dataset = np.memmap(
            im_path,
            dtype=np.uint8,
            mode="r",
            shape=(num_simulations, traj_len, num_views, H, W),
        )

        y_sliced = np.memmap(
            save_dir / "dataset_memmap.dat",
            dtype=np.uint8,
            mode="w+",
            shape=(num_seq, seq_len, 2 * num_views, H, W),
         )# sortie finale: channels = 2*n_views  (t et t+1 pour chaque vue)
        y_flat = y_sliced.reshape(total_pairs, 2 * num_views, H, W)

        pair_idx = 0
        for i in tqdm(range(num_simulations), desc="Reshape traj-by-traj"):
            traj = im_dataset[i]
            pairs = self._traj_to_pairs(traj)
            n_pairs_i = pairs.shape[0]
            y_flat[pair_idx:pair_idx + n_pairs_i] = pairs
            pair_idx += n_pairs_i
        assert pair_idx == total_pairs

        y_sliced.flush()
        self._write_metadata(
            save_dir,
            [num_seq, seq_len, 2 * num_views, H, W],
            num_views
        )
        del y_sliced, y_flat, im_dataset
        os.remove(im_path)

    @staticmethod
    def _compute_shapes(
            num_simulations: int,
            traj_len: int,
            seq_len: int,
            resolution: int
    ) -> tuple[int, int, int, int]:

        n_pairs_per_traj = traj_len - 1
        total_pairs = num_simulations * n_pairs_per_traj
        assert total_pairs % seq_len == 0, "total_pairs doit être divisible par seq_len"
        num_seq = total_pairs // seq_len
        H = W = resolution
        return total_pairs, num_seq, H, W

    @staticmethod
    def _traj_to_pairs(traj: np.ndarray) -> np.ndarray:
        """
        traj: (T, V, H, W)
        returns pairs: (T-1, 2V, H, W) where channels = [views@t, views@t+1]
        """
        t_img = traj[:-1]
        tp1_img = traj[1:]
        return np.concatenate([t_img, tp1_img], axis=1)

    @staticmethod
    def _write_metadata(save_dir: Path, y_shape, n_views: int) -> None:
        metadata = {
            "y_shape": list(y_shape),
            "dtype": "uint8",
            "n_views": int(n_views),
            "channel_layout": "concat([views@t, views@t+1])",
        }
        with open(save_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

