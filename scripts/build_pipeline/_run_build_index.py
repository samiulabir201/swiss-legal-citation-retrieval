"""Run build_index locally to generate DB + cards for Drive upload."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from segment_lattice_v3 import build_index

DATA_DIR = ROOT / "data"
INSIGHTS_DIR = ROOT / "data_insights"
ART_DIR = ROOT / "artifacts"
DB_PATH = ART_DIR / "segment_lattice_v3.sqlite"
CARDS_PATH = ART_DIR / "choice_cards_v3.json"

print("=" * 60)
print("Building segment-lattice index (force=True) ...")
print(f"  DB_PATH   : {DB_PATH}")
print(f"  CARDS_PATH: {CARDS_PATH}")
print(f"  DATA_DIR  : {DATA_DIR}")
print("=" * 60)

t0 = time.time()
build_index(
    data_dir=DATA_DIR,
    insights_dir=INSIGHTS_DIR,
    db_path=DB_PATH,
    cards_path=CARDS_PATH,
    force=True,
)
elapsed = time.time() - t0

print("=" * 60)
print(f"Done in {elapsed:.1f}s")
print(f"DB size   : {DB_PATH.stat().st_size / 1_048_576:.1f} MB")
print(f"Cards size: {CARDS_PATH.stat().st_size / 1_048_576:.2f} MB")
print("Upload both files to /content/drive/MyDrive/swiss_law/artifacts/")
