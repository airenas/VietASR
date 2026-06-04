import argparse
import logging
import os
import random
from pathlib import Path

from lhotse import CutSet
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        help="""Manifest file.
        """,
    )
    parser.add_argument(
        "--out-dev",
        type=Path,
        help="""Manifest file for the development set.
        """,
    )
    parser.add_argument(
        "--out-train",
        type=Path,
        help="""Manifest file for the training set.
        """,
    )
    parser.add_argument(
        "--secs-for-dev",
        type=int,
        default=10 * 3600,
        help="""Number of seconds for the development set.
        """,
    )
    args = parser.parse_args()

    logging.info(f"Input set: {args.input}")
    logging.info(f"Training set: {args.out_train}")
    logging.info(f"Development set: {args.out_dev}")
    logging.info(f"Dev secs: {args.secs_for_dev}")

    random.seed(42)
    input_path = Path(args.input)

    cuts = CutSet.from_file(input_path)
    lc = len(cuts)
    logging.info(f"Total cuts: {lc}")
    ca = []
    for cut in tqdm(cuts, desc="Iterating cuts", total=lc):
        ca.append((cut.id, cut.duration))
    random.shuffle(ca)
    dev_ids = set()
    dev_total = 0.0
    for cut_id, duration in tqdm(ca, desc="Selecting dev cuts"):
        if dev_total <= args.secs_for_dev:
            dev_ids.add(cut_id)
            dev_total += duration
        else:
            break
    logging.info(f"selected {len(dev_ids)}, duration = {dev_total / 3600:.2f} h")
    dev = []
    dev_total = 0.0
    train = []
    train_total = 0.0

    for cut in tqdm(cuts, desc="Selecting cuts", total=lc):
        if cut.id in dev_ids:
            dev.append(cut)
            dev_total += cut.duration
        else:
            train.append(cut)
            train_total += cut.duration
    subset = CutSet.from_cuts(dev)
    logging.info(f"Shuffling dev subset")
    subset = subset.shuffle()
    logging.info(f"Saving dev subset: {len(dev)} cuts, {dev_total / 3600:.2f} h")
    subset.to_file(args.out_dev)
    subset = CutSet.from_cuts(train)
    logging.info(f"Shuffling train subset")
    subset = subset.shuffle()
    logging.info(f"Saving train subset: {len(train)} cuts, {train_total / 3600:.2f} h")
    subset.to_file(args.out_train)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter,
                        level=getattr(logging, os.environ.get("LOGLEVEL", "INFO").upper(), logging.WARNING))

    logging.info(f"Starting")

    main()

    logging.info("Done")
