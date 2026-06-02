# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import sys

import joblib
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("learn_kmeans")


def get_km_model(
        n_clusters,
        init,
        max_iter,
        batch_size,
        tol,
        max_no_improvement,
        n_init,
        reassignment_ratio,
):
    logger.info("Creating MiniBatchKMeans model")
    return MiniBatchKMeans(
        n_clusters=n_clusters,
        init=init,
        max_iter=max_iter,
        batch_size=batch_size,
        verbose=1,
        compute_labels=False,
        tol=tol,
        max_no_improvement=max_no_improvement,
        init_size=None,
        n_init=n_init,
        reassignment_ratio=reassignment_ratio,
    )


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--km-path", type=str)
    parser.add_argument("--n-clusters", type=int)
    parser.add_argument("--input", type=str, required=True, help="Path to the input features")
    parser.add_argument("--init", default="k-means++")
    parser.add_argument("--max-iter", default=100, type=int)
    parser.add_argument("--batch-size", default=10000, type=int)
    parser.add_argument("--tol", default=0.0, type=float)
    parser.add_argument("--max-no-improvement", default=100, type=int)
    parser.add_argument("--n-init", default=20, type=int)
    parser.add_argument("--reassignment-ratio", default=0.0, type=float)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--prune-range",
        type=int,
        default=5,
        help="The prune range for rnnt loss, it means how many symbols(context)"
             "we are using to compute the loss",
    )
    return parser


def learn_kmeans(
        input_file,
        km_path,
        n_clusters,
        seed,
        init,
        max_iter,
        batch_size,
        tol,
        n_init,
        reassignment_ratio,
        max_no_improvement,
):
    np.random.seed(seed)
    km_model = get_km_model(
            n_clusters,
            init,
            max_iter,
            batch_size,
            tol,
            max_no_improvement,
            n_init,
            reassignment_ratio,
        )

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    logging.info(f"Device: {device}")

    logging.info(f"Loading data from {input_file}")
    part_feats = np.load(input_file)

    logging.info(f"data size: {part_feats.shape}")
    logger.info(f"Loaded {part_feats.nbytes / 1e9:.2f} GB into memory")

    km_model.fit(part_feats)
    joblib.dump(km_model, km_path)
    inertia = -km_model.score(part_feats) / len(part_feats)
    logging.info(f"Total inertia: {inertia:.5f}")

    logger.info("finished successfully")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    args.feature_dim = 80
    logging.info(str(args))

    learn_kmeans(
        input_file=args.input,
        km_path=args.km_path,
        n_clusters=args.n_clusters,
        seed=args.seed,
        init=args.init,
        max_iter=args.max_iter,
        batch_size=args.batch_size,
        tol=args.tol,
        n_init=args.n_init,
        reassignment_ratio=args.reassignment_ratio,
        max_no_improvement=args.max_no_improvement,
    )
