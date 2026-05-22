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
        "--storage_path_dir",
        type=Path,
        help="Path to the storage directory for features",
    )
    return parser.parse_args()


def fix_path(old_path, storage_path_dir):
    return str(Path(storage_path_dir) / Path(old_path).name)


def main():
    args = get_args()
    logging.info(vars(args))
    cuts = load_manifest_lazy(args.input)
    logging.info("Fixing paths")

    def map_cut(cut):
        if cut.has_features:
            new_path = fix_path(cut.features.storage_path, args.storage_path_dir)
            cut.features.storage_path = new_path
        else:
            logging.warning(f"Cut {cut.id} has no features")
        return cut

    # logging.info(f"Call mapping")
    # mapped = (map_cut(cut) for cut in tqdm(iter(cuts), desc="Rewriting cuts"))
    # logging.info(f"Call save")
    # CutSet.from_cuts(mapped).to_file(args.output)
    with CutSet.open_writer(args.output) as writer:
        for cut in tqdm(iter(cuts), desc="Rewriting cuts"):
            writer.write(map_cut(cut))
    logging.info(f"Saved {len(cuts)} cuts to {args.output}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
