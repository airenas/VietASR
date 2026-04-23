import argparse
import logging
from pathlib import Path

from lhotse import load_manifest_lazy, CutSet
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        help="Path to the input cutset",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Path to the output cutset directory",
    )

    parser.add_argument(
        "--split-into",
        type=int,
        default=10,
        help="How many parts to make",
    )
    return parser.parse_args()


def main():
    args = get_args()
    logging.info(vars(args))
    cuts = load_manifest_lazy(args.input)
    pbar = tqdm(cuts, desc="Calculating cuts duration", unit="s")
    total = 0.0
    for cut in pbar:
        total += cut.duration
        pbar.update(cut.duration)

    wanted_secs = total / args.split_into
    logging.info(f"Splitting into {args.split_into} parts, by {wanted_secs:.2f} secs each, total {total:.2f} secs")

    cuts = load_manifest_lazy(args.input)
    pbar = tqdm(cuts, desc="Splitting", unit="s", total=total)
    selected = []
    total = 0.0
    num = 0

    def save_subset():
        nonlocal selected, total, num
        subset = CutSet.from_cuts(selected)
        file_name = args.output_dir / f"cuts_pretrain_train_{num:03d}.jsonl.gz"
        logging.info(f"Saving subset: {len(subset)} cuts, {total:.2f} secs to {file_name}")
        subset.to_file(file_name)        
        selected = []
        total = 0.0   
        num += 1

    for cut in pbar:
        if total >= wanted_secs:
            save_subset()
        selected.append(cut)
        total += cut.duration
        pbar.update(cut.duration)

    if total >= 0:
        save_subset()

    logging.info(f"Saved")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
