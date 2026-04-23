import argparse
import logging
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--template-in",
        type=str,
        required=True,
        help="Template to use for input files",
    )

    parser.add_argument(
        "--template-out",
        type=str,
        required=True,
        help="Template to use for output files",
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
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Gpu count",
    )
    return parser.parse_args()


def main():
    args = get_args()
    logging.info(vars(args))
    res = []
    for i in range(args.count):
        old = args.template_in.replace("{}", f"{i:03d}")
        new = args.template_out.replace("{}", f"{i:03d}")
        res.append(f"{old} {new}")

    in_file = args.count / args.gpus

    start = 0
    end = in_file
    for i in range(args.gpus):
        if i == args.gpus - 1:
            end = len(res)
        else:
            end = int((i + 1) * in_file)
        logging.info(f"GPU {i}: {start} - {end}")
        with open(f"{args.output}{i}", "w") as f:
            for r in res[start: end]:
                f.write(f"{r}\n")
        start = end

    logging.info(f"Saved {args.output}.* files")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
