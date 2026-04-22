import argparse
import logging
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--template",
        type=str,
        required=True,
        help="Template to use",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Path to the output file",
    )

    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Tasks count",
    )
    return parser.parse_args()


def main():
    args = get_args()
    logging.info(vars(args))
    with open(args.output, "w") as f:
        for i in range(args.count):
            old = args.template.replace("{}", f"{i:03d}")
            new = args.template.replace("{}", f"l_{i:03d}")
            f.write(f"{old} {new}\n")

    logging.info(f"Saved {args.output}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
