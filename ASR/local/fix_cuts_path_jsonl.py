import argparse
import gzip
import json
import logging
from pathlib import Path

from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        help="Path to the input cutset",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Path to the output cutset",
    )

    parser.add_argument(
        "--storage_path_dir",
        type=Path,
        help=(
            "Old storage directory for features when replacing prefix. "
            "If --new-storage_path_dir is not provided, this is treated as target directory."
        ),
    )
    parser.add_argument(
        "--new-storage_path_dir",
        "--new_storage_path_dir",
        dest="new_storage_path_dir",
        type=Path,
        help="Target storage directory for features",
    )
    return parser.parse_args()


def open_text_auto(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode=mode, encoding="utf-8")
    return open(path, mode=mode, encoding="utf-8")


def fix_path(old_path: str, old_storage_dir: Path, new_storage_dir: Path):
    old_path_p = Path(old_path)

    if old_storage_dir is not None:
        try:
            rel = old_path_p.relative_to(old_storage_dir)
            return str(new_storage_dir / rel)
        except ValueError:
            pass

    return str(new_storage_dir / old_path_p.name)


def rewrite_record(record: dict, old_storage_dir: Path, new_storage_dir: Path):
    features = record.get("features")
    if not isinstance(features, dict):
        return False

    old_path = features.get("storage_path")
    if not isinstance(old_path, str) or not old_path:
        return False

    new_path = fix_path(old_path, old_storage_dir, new_storage_dir)
    if new_path == old_path:
        return False

    features["storage_path"] = new_path
    return True


def main():
    args = get_args()

    old_storage_dir = args.storage_path_dir
    new_storage_dir = args.new_storage_path_dir

    logging.info(f"Input: {args.input}")
    logging.info(f"Output: {args.output}")
    logging.info(f"Old storage path: {old_storage_dir}")
    logging.info(f"New storage path: {new_storage_dir}")

    fixed = 0
    skipped = 0
    processed = 0

    with open_text_auto(args.input, "rt") as f_in, open_text_auto(args.output, "wt") as f_out:
        for line in tqdm(f_in, desc="Rewriting cuts"):
            if not line.strip():
                continue

            processed += 1
            record = json.loads(line)
            changed = rewrite_record(record, old_storage_dir, new_storage_dir)

            if changed:
                fixed += 1
            else:
                skipped += 1

            f_out.write(json.dumps(record, ensure_ascii=False))
            f_out.write("\n")

    logging.info(
        "Processed %s lines. Updated %s storage paths. Skipped %s lines.",
        processed,
        fixed,
        skipped,
    )
    logging.info(f"Saved rewritten cuts to {args.output}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
