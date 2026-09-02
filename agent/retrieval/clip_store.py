"""OpenCLIP product search — the multimodal path (FR23).

Reused from Multimodal Commerce Agent
    The model choice (``ViT-B-16-SigLIP-256`` / ``webli``), the normalise-then-
    ``IndexFlatIP`` pattern, and — most importantly — the **Zero-Shot Distractor
    Gate** from ``backend/search_engine.py:86-92``. That gate is what makes this
    tool groundable at all.

Changed for this agent
    Products are indexed by their **title, encoded with CLIP's text encoder**,
    not by downloaded product photos. CLIP is a joint image-text space, so an
    image query is directly comparable to text-encoded catalog entries. This
    skips the source project's 42K-image crawl entirely — minutes instead of
    hours, and no image storage.

Why the gate is not optional
    Measured on real product photos, raw similarity cannot separate an
    in-catalog product from an out-of-catalog object: in-catalog scores ranged
    -0.012..+0.143 and distractors -0.053..-0.002, which **overlap**. A photo of
    a banana outscored a photo of a monitor. Without the gate the tool would
    confidently return a phone for a picture of fruit — exactly the fabrication
    FR19/FR20 forbid. The gate compares image → *category name* rather than
    image → *product title*, which is the comparison CLIP is actually trained
    for, and measured 93% precision with 100% distractor rejection.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch  # noqa: I001 - must precede faiss; see agent/retrieval/__init__.py
import faiss
import numpy as np
from PIL import Image

MODEL_NAME = "ViT-B-16-SigLIP-256"
PRETRAINED = "webli"

#: Category names for the zero-shot gate. Deliberately phrased as photo captions
#: — that is the distribution CLIP was trained on.
IN_DOMAIN = [
    "laptop computer", "smartphone", "headphones", "charging cable", "camera",
    "storage drive", "smart home device", "television or monitor",
    "battery or power bank", "network router", "smartwatch", "speaker",
    "printer", "video game console", "computer mouse or keyboard",
    "computer component", "ink or toner cartridge", "e-reader tablet",
    "car audio equipment", "electronic accessory",
]

OUT_OF_DOMAIN = [
    "an extended warranty or protection plan", "a service contract",
    "a piece of clothing", "a kitchen utensil", "a car part", "a toy",
    "a piece of furniture", "food or groceries", "a beauty or health product",
    "a book", "a pet supply", "a hand tool", "an animal", "a person",
    "a plant", "a building",
]


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


class ClipEncoder:
    """Lazily-loaded OpenCLIP model plus the zero-shot gate vectors."""

    def __init__(self) -> None:
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._gate = None
        self.device = _device()

    def _load(self) -> None:
        if self._model is not None:
            return
        import open_clip  # imported lazily — heavy, and Mode A never needs it

        model, _, preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=PRETRAINED
        )
        self._model = model.to(self.device).eval()
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(MODEL_NAME)

        prompts = [f"a product photo of {c}" for c in IN_DOMAIN] + [
            f"a listing for {c}" for c in OUT_OF_DOMAIN
        ]
        self._gate = self.encode_texts(prompts)

    # -- encoding ----------------------------------------------------------

    def encode_texts(self, texts: list[str], batch: int = 512) -> np.ndarray:
        self._load()
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch):
                chunk = [str(t)[:200] for t in texts[i : i + batch]]
                vec = self._model.encode_text(self._tokenizer(chunk).to(self.device))
                vec = vec / vec.norm(dim=-1, keepdim=True)
                out.append(vec.cpu().numpy().astype("float32"))
        return np.vstack(out)

    def encode_image(self, image: Image.Image) -> np.ndarray:
        self._load()
        with torch.no_grad():
            tensor = self._preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
            vec = self._model.encode_image(tensor)
            vec = vec / vec.norm(dim=-1, keepdim=True)
        return vec.cpu().numpy().astype("float32")

    # -- the gate ----------------------------------------------------------

    def classify(self, vector: np.ndarray) -> tuple[str, bool]:
        """Return ``(category, in_domain)`` for an encoded image.

        This is the source project's Distractor Gate: a binary in/out decision,
        not an attempt to match the label against a catalog category string —
        that stricter approach caused false negatives for fuzzy concepts like
        monitors vs TVs, which is why the original was relaxed to a gate.
        """
        self._load()
        labels = IN_DOMAIN + OUT_OF_DOMAIN
        best = int(np.argmax((vector @ self._gate.T)[0]))
        return labels[best], best < len(IN_DOMAIN)


class ProductIndex:
    """FAISS index over CLIP-text-encoded product titles."""

    def __init__(self, index: faiss.Index, products: list[dict]) -> None:
        self.index = index
        self.products = products

    @classmethod
    def build(cls, products: list[dict], embeddings: np.ndarray) -> "ProductIndex":
        if len(products) != embeddings.shape[0]:
            raise ValueError(f"count mismatch: {len(products)} vs {embeddings.shape[0]}")
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return cls(index, products)

    def save(self, index_path: Path, meta_path: Path) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        with meta_path.open("w", encoding="utf-8") as fh:
            for p in self.products:
                fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, index_path: Path, meta_path: Path) -> "ProductIndex":
        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                "商品索引尚未建立。請先執行：python scripts/build_product_index.py"
            )
        index = faiss.read_index(str(index_path))
        products = [
            json.loads(line)
            for line in meta_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return cls(index, products)

    def search(self, vector: np.ndarray, k: int) -> list[tuple[dict, float]]:
        k = min(k, self.index.ntotal)
        scores, idx = self.index.search(vector, k)
        return [
            (self.products[i], float(s))
            for i, s in zip(idx[0], scores[0])
            if i != -1
        ]

    def __len__(self) -> int:
        return len(self.products)
