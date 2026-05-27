import argparse
import logging
import os
from pathlib import Path

from lhotse import CutSet
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        help="""Input file.
        """,
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="""Output file.
            """,
    )
    
    return parser.parse_args()

def main():
    args = get_args()
    logging.info(f"Loading cuts from {args.input}")
    cuts = CutSet.from_jsonl_lazy(args.input)
    logging.info(f"Saving cuts to {args.output}")

    with CutSet.open_writer(args.output) as writer:
        for cut in tqdm(iter(cuts.resample(16000))):
            writer.write(cut)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter,
                        level=getattr(logging, os.environ.get("LOGLEVEL", "WARNING").upper(), logging.WARNING))

    logging.info(f"Starting")
    main()
    logging.info(f"Done")        