from pathlib import Path
from uuid import UUID
from typing import List, Tuple
import os

import numpy as np
import pandas as pd

from spectrumx import Client
from spectrumx.errors import SDSError
from dotenv import load_dotenv

from src.edge_ran_gary.config import SpectrumXDatasetConfig



load_dotenv()  

class SpectrumXDataset:
    DATASET_UUID = UUID("458c3f72-8d7e-49cc-9be3-ed0b0cd7e03d")

    def __init__(self, cfg: SpectrumXDatasetConfig):
        self.cfg = cfg
        self.root_dir = cfg.dataset_root
        self.files_dir = self.root_dir / "files"

        if self.files_dir.exists():
            self._discover_paths()

    # -------------------------
    # Download
    # -------------------------

    def download(self, overwrite: bool = False, verbose: bool = True):
        token = os.getenv("SDS_SECRET_TOKEN")
        if token is None:
            raise RuntimeError(
                "SDS_SECRET_TOKEN not found. Ensure it exists in .env."
            )

        client = Client(
            host=self.cfg.sds_host,
            env_config={"SDS_SECRET_TOKEN": token},
        )
        client.dry_run = False
        client.authenticate()

        if self.root_dir.exists() and not overwrite:
            print(f"Dataset already exists at {self.root_dir}, skipping download.")
            return

        print(f"Downloading SpectrumX dataset to {self.root_dir}")

        results = client.download_dataset(
            dataset_uuid=self.DATASET_UUID,
            to_local_path=self.root_dir,
            skip_contents=False,
            overwrite=overwrite,
            verbose=verbose,
        )

        failed = [r for r in results if not r]
        if failed:
            raise RuntimeError(f"{len(failed)} files failed to download")

        self._discover_paths()

    # -------------------------
    # Internal path discovery
    # -------------------------

    def _discover_paths(self):
        user_dirs = [d for d in self.files_dir.iterdir() if d.is_dir()]
        if not user_dirs:
            raise RuntimeError("No user directory found in dataset")

        self.user_dir = user_dirs[0]
        self.unlabeled_dir = self.user_dir / "training data"
        self.labeled_dir = self.user_dir / "VLABrutal"
        self.gt_path = self.labeled_dir / "groundtruth.csv"

    # -------------------------
    # Loaders
    # -------------------------

    def load_labeled(self) -> Tuple[List[np.ndarray], np.ndarray, List[str]]:
        gt = pd.read_csv(self.gt_path)
        X, y, filenames = [], [], []

        for _, row in gt.iterrows():
            file_path = self.labeled_dir / row["filename"]
            X.append(np.load(file_path))
            y.append(int(row["label"]))
            filenames.append(row["filename"])

        return X, np.array(y), filenames

    def load_unlabeled(self) -> Tuple[List[np.ndarray], List[str]]:
        X, filenames = [], []

        for file_path in sorted(self.unlabeled_dir.glob("*.npy")):
            X.append(np.load(file_path))
            filenames.append(file_path.name)

        return X, filenames


# -------------------------
# Manual test
# -------------------------

if __name__ == "__main__":
    cfg = SpectrumXDatasetConfig()
    ds = SpectrumXDataset(cfg)

    ds.download(overwrite=False)
    X_l, y, _ = ds.load_labeled()
    X_u, _ = ds.load_unlabeled()

    print("Loader test successful ✅")
    print(f"Labeled samples: {len(X_l)}")
    print(f"Unlabeled samples: {len(X_u)}")
