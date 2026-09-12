"""Prepare a checksum-verified, revision-pinned FashionMNIST dataset at image build time."""

import json
from pathlib import Path
import sys

from common import sha256
from torchvision.datasets import FashionMNIST

REVISION = "b2617bb6d3ffa2e429640350f613e3291e10b141"


def prepare(root):
    FashionMNIST.mirrors = [
        "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/"
        + REVISION
        + "/data/fashion/"
    ]
    # torchvision verifies the source MD5 checksums before decompressing.
    FashionMNIST(root, train=True, download=True)
    FashionMNIST(root, train=False, download=True)
    files = {
        str(path.relative_to(root)): sha256(path.read_bytes())
        for path in sorted(Path(root).glob("FashionMNIST/raw/*"))
        if path.is_file()
    }
    if len(files) != 8:
        raise ValueError("Expected four compressed and four decompressed dataset files")
    manifest = {"source_revision": REVISION, "files": files}
    Path(root, "data.json").write_text(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    prepare(sys.argv[1])
