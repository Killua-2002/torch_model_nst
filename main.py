import os
import sys
import time
import subprocess
from pathlib import Path

# =============================================================================
# CONFIG - edit this block first
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent

# Pipeline toggles
RUN_PREPARE_SINGLE_CHROMOSOMES = False
RUN_GENERATE_SYNTHETIC_IMAGES = False
RUN_PREPROCESS_TO_SIZE = False
RUN_SPLIT_DATA = False
RUN_TRAIN = True
RUN_EVALUATE = True

# Data generation config
NUM_SAMPLES = 5000
SEED = 42
GENERATION_WORKERS = max(2, min(8, (os.cpu_count() or 2)))

# Train/Eval config
DATASET_DIR = PROJECT_ROOT / "dataset"
RESULTS_DIR = PROJECT_ROOT / "results_all_in_one"
EPOCHS = 100
BATCH_SIZE = 16
LR = 1e-4

def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}

def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else int(raw)

def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else float(raw)

def env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else Path(raw)

# Apply environment overrides
RUN_PREPARE_SINGLE_CHROMOSOMES = env_bool("RUN_PREPARE_SINGLE_CHROMOSOMES", RUN_PREPARE_SINGLE_CHROMOSOMES)
RUN_GENERATE_SYNTHETIC_IMAGES = env_bool("RUN_GENERATE_SYNTHETIC_IMAGES", RUN_GENERATE_SYNTHETIC_IMAGES)
RUN_PREPROCESS_TO_SIZE = env_bool("RUN_PREPROCESS_TO_SIZE", RUN_PREPROCESS_TO_SIZE)
RUN_SPLIT_DATA = env_bool("RUN_SPLIT_DATA", RUN_SPLIT_DATA)
RUN_TRAIN = env_bool("RUN_TRAIN", RUN_TRAIN)
RUN_EVALUATE = env_bool("RUN_EVALUATE", RUN_EVALUATE)

NUM_SAMPLES = env_int("NUM_SAMPLES", NUM_SAMPLES)
SEED = env_int("SEED", SEED)
GENERATION_WORKERS = env_int("GENERATION_WORKERS", GENERATION_WORKERS)
EPOCHS = env_int("EPOCHS", EPOCHS)
BATCH_SIZE = env_int("BATCH_SIZE", BATCH_SIZE)
LR = env_float("LR", LR)

DATASET_DIR = env_path("DATASET_DIR", DATASET_DIR)
RESULTS_DIR = env_path("RESULTS_DIR", RESULTS_DIR)

def command_text(args: list[str | Path]) -> str:
    parts = []
    for arg in args:
        text = str(arg)
        parts.append(f'"{text}"' if " " in text else text)
    return " ".join(parts)

def run_cmd(args: list[str | Path], title: str, check: bool = True) -> int:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    print("$", command_text(args))
    sys.stdout.flush()
    t0 = time.time()
    proc = subprocess.Popen(
        [str(a) for a in args],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    while True:
        char = proc.stdout.read(1)
        if not char:
            break
        sys.stdout.write(char)
        sys.stdout.flush()
    code = proc.wait()
    print(f"\n[exit={code}] elapsed={time.time() - t0:.1f}s")
    if check and code != 0:
        raise RuntimeError(f"Command failed: {command_text(args)}")
    return code

def py_script(path: str) -> list[str | Path]:
    return [sys.executable, "-u", PROJECT_ROOT / path]

def main():
    print(f"Project root: {PROJECT_ROOT}")

    if RUN_PREPARE_SINGLE_CHROMOSOMES:
        run_cmd(py_script("preprocessing/2v1_prepare_single_chromosomes.py"), "Step 1: Prepare single chromosomes")

    if RUN_GENERATE_SYNTHETIC_IMAGES:
        run_cmd(py_script("preprocessing/3v1_generate_synthetic_masks.py") + [
            "--num-samples", str(NUM_SAMPLES),
            "--seed", str(SEED),
            "--workers", str(GENERATION_WORKERS)
        ], "Step 2: Generate synthetic images")

    if RUN_PREPROCESS_TO_SIZE:
        run_cmd(py_script("preprocessing/4v1_preprocess_to_256.py"), "Step 3: Preprocess to 256x256")

    if RUN_SPLIT_DATA:
        run_cmd(py_script("preprocessing/5v1_split_data.py"), "Step 4: Split train/val/test")

    if RUN_TRAIN:
        run_cmd(py_script("train.py") + [
            "--dataset-dir", str(DATASET_DIR),
            "--output-dir", str(RESULTS_DIR),
            "--epochs", str(EPOCHS),
            "--batch-size", str(BATCH_SIZE),
            "--lr", str(LR)
        ], "Step 5: Train all-in-one model")

    if RUN_EVALUATE:
        weights_path = RESULTS_DIR / "best_model.pth"
        if weights_path.exists():
            for split in ["test", "real_test"]:
                split_dir = DATASET_DIR / split
                if split_dir.exists():
                    run_cmd(py_script("evaluate.py") + [
                        "--dataset-dir", str(DATASET_DIR),
                        "--weights", str(weights_path),
                        "--output-dir", str(RESULTS_DIR),
                        "--split", split
                    ], f"Step 6: Evaluate all-in-one model on {split}")
        else:
            print(f"Warning: {weights_path} not found. Skipping evaluation.")

if __name__ == "__main__":
    main()
