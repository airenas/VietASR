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
        "--output",
        type=Path,
        help="Path to the output cutset",
    )

    parser.add_argument(
        "--secs",
        type=int,
        default=100 * 3600,
        help="How many secs to take",
    )
    return parser.parse_args()


def main():
    args = get_args()
    logging.info(vars(args))
    cuts = load_manifest_lazy(args.input)
    logging.info(f"Shuffle")
    cuts = cuts.shuffle()
    logging.info(f"Taking subset")
    selected = []
    total = 0.0
    pbar = tqdm(cuts, desc="Selecting cuts", unit="s", total=args.secs)
    for cut in pbar:
        if total >= args.secs:
            break
        selected.append(cut)
        total += cut.duration
        pbar.update(cut.duration)
    subset = CutSet.from_cuts(selected)
    logging.info(f"Saving subset: {len(subset)} cuts, {total:.2f} secs")
    subset.to_file(args.output)
    logging.info(f"Saved {len(subset)} cuts to {args.output}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
