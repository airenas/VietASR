#!/usr/bin/env python3

import argparse
import logging
from multiprocessing import Lock
from pathlib import Path

import torch
from lhotse import CutSet, KaldifeatFbank, KaldifeatFbankConfig, FbankConfig, Fbank

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

device_lock = Lock()


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--input",
        type=Path,
    )
    parser.add_argument(
        "--batch-duration",
        type=float,
        default=600.0,
        help="The maximum number of audio seconds in a batch."
             "Determines batch size dynamically.",
    )

    parser.add_argument(
        "--output",
        type=Path,
    )

    return parser


def compute_fbank(input, output, args):
    num_digits = 8  # num_digits is fixed by lhotse split-lazy

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    num_mel_bins = 80
    extractor = Fbank(FbankConfig(num_mel_bins=num_mel_bins))
    # extractor = KaldifeatFbank(KaldifeatFbankConfig(device=device))
    logging.info(f"device: {device}")

    logging.info(f"Loading {input}")
    idx = input.stem.split("_")[-1]
    logging.info(f"Computing features for split {idx}")
    cut_set = CutSet.from_file(input)

    out_dir = output.parent
    feat_dir = out_dir / "feats"
    feat_dir.mkdir(parents=True, exist_ok=True)

    cut_set = cut_set.compute_and_store_features_batch(
        extractor=extractor,
        storage_path=f"{feat_dir}/pretrain_feats_{idx}",
        num_workers=4,
        batch_duration=args.batch_duration,
        overwrite=True,
    )

    logging.info("About to split cuts into smaller chunks.")
    cut_set = cut_set.trim_to_supervisions(
        keep_overlapping=False, min_duration=None
    )

    logging.info(f"Saving to {output}")
    cut_set.to_file(output)
    logging.info(f"Saved to {output}")


def main():
    parser = get_parser()
    args = parser.parse_args()

    logging.info(vars(args))
    compute_fbank(args.input, args.output, args)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    logging.info("Starting main process")
    main()
    logging.info("Done")
