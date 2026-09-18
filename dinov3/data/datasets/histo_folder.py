# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Dataset custom pour finetuning SSL facon DinoBloom sur DINOv3.
# Lit un dossier d'images (recursif) ou un/des fichier(s) .txt de chemins.
# A placer dans : dinov3/data/datasets/histo_folder.py
"""
HistoFolder : dataset d'images sans labels pour l'entrainement auto-supervise.
Compatible avec l'interface ExtendedVisionDataset de DINOv3 :
  - get_image_data(index) -> bytes bruts de l'image (decodes par la classe de base)
  - get_target(index)     -> None (SSL, pas de label)
  - __len__               -> nombre reel d'images
Usage dans une config DINOv3 :
  dataset_path: HistoFolder:root=/home/a_claveau/patch_lists_mydata
`root` peut etre :
  - un dossier contenant des images (scan recursif), ou
  - un dossier contenant un/des fichier(s) .txt (un chemin d'image par ligne),
    comme le patch_root de DinoBloom.

NOTE MEMOIRE (important a grande echelle, ex: 15M images) :
  Les chemins ne sont PAS stockes dans une list[str] Python. Avec les
  DataLoader workers (num_workers>0, fork), l'acces a une grande liste Python
  fait grimper les refcounts objet par objet -> le copy-on-write duplique les
  pages -> la RAM monte lineairement au fil des iterations (fuite classique,
  cf. github.com/pytorch/pytorch/issues/13246).
  Solution : on encode tous les chemins dans DEUX arrays numpy contigus
  (un buffer d'octets + des offsets). Les arrays numpy n'ont pas de refcount
  par element : le fork les partage vraiment en lecture seule, sans fuite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .extended import ExtendedVisionDataset

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".JPEG"}


class HistoFolder(ExtendedVisionDataset):
    def __init__(
        self,
        *,
        root: str,
        transforms=None,
        transform=None,
        target_transform=None,
    ) -> None:
        super().__init__(
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
        )
        paths: list[str] = self._build_index(Path(root))
        if not paths:
            raise RuntimeError(f"HistoFolder: aucune image trouvee sous {root}")

        # --- Encodage des chemins en buffers numpy (pas de list[str]) ---
        # buffer : tous les chemins concatenes, encodes en UTF-8 (gere les accents)
        # offsets : position de debut de chaque chemin ; offsets[i+1] = fin du i-eme
        encoded = [p.encode("utf-8") for p in paths]
        self._n = len(encoded)
        lengths = np.fromiter((len(b) for b in encoded), dtype=np.int64, count=self._n)
        self._offsets = np.zeros(self._n + 1, dtype=np.int64)
        np.cumsum(lengths, out=self._offsets[1:])
        self._buf = np.frombuffer(b"".join(encoded), dtype=np.uint8)

        print(f"HistoFolder: {self._n} images depuis {root}")

    def _build_index(self, root: Path) -> list[str]:
        # 1) si root contient des .txt, on lit les chemins listes dedans (mode DinoBloom)
        txts = sorted(root.glob("*.txt")) if root.is_dir() else []
        if txts:
            paths: list[str] = []
            for txt in txts:
                print(f"HistoFolder: lecture de {txt}")
                with open(txt, "r", encoding="utf-8") as fh:
                    for line in fh:
                        s = line.strip()
                        if s:
                            paths.append(s)
            return paths
        # 2) sinon, scan recursif du dossier d'images
        paths = []
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix in IMAGE_EXTS:
                paths.append(str(p.resolve()))
        return paths

    def _path(self, index: int) -> str:
        # reconstruit le chemin depuis le buffer numpy (aucune list[str] en memoire)
        start = int(self._offsets[index])
        end = int(self._offsets[index + 1])
        return self._buf[start:end].tobytes().decode("utf-8")

    def get_image_data(self, index: int) -> bytes:
        self._gc_counter = getattr(self, "_gc_counter", 0) + 1
        if self._gc_counter % 5000 == 0:
            import gc

            gc.collect()
        path = self._path(index)
        with open(path, "rb") as f:
            return f.read()

    def get_target(self, index: int) -> Any | None:
        # SSL : pas de label.
        return None

    def __len__(self) -> int:
        return self._n
