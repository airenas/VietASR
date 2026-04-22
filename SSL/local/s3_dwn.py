import argparse
import io
import logging
import os
import shutil
import sys
import tarfile
import time
import traceback
from multiprocessing import JoinableQueue, Process, Queue
from pathlib import Path

import botocore
from tqdm import tqdm

class S3AudioLoader:
    def __init__(self, bucket):
        import boto3
        self.bucket = bucket
        self.s3_client = boto3.client(
            "s3",
            config=botocore.config.Config(
                retries={"max_attempts": 5, "mode": "standard"}
            ),
        )
        ## check access to bucket
        self.list_top_files(n=10)
        logging.info(f"Initialized S3AudioLoader with bucket {bucket}")

    def list_top_files(self, n=10):
        bucket, prefix = self.bucket.split("/", 1)
        resp = self.s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=n)
        files = []
        for obj in resp.get("Contents", []):
            files.append(obj["Key"])
        return files

    def load(self, name):
        start = time.perf_counter()
        bucket, key = self.bucket.split("/", 1)
        name = f"{key}/{name}"
        logging.info(f"Loading {name}")
        res = io.BytesIO()
        self.s3_client.download_fileobj(bucket, name, res)
        elapsed = time.perf_counter() - start
        logging.info(f"Loaded {name} from S3 in {elapsed:.2f} seconds")
        res.seek(0)
        return res


def main(argv):
    logging.info("Starting")
    parser = argparse.ArgumentParser(description="Copy audio to destination from segments")
    parser.add_argument("--s3-bucket", nargs='?', required=True, help="S3 bucket to read segments from")
    parser.add_argument("--s3-from", nargs='?', required=False, default=0, type=int, help="Index to start from")
    parser.add_argument("--s3-to", nargs='?', required=False, default=999999, type=int, help="Index to end at")
    parser.add_argument("--workers", nargs='?', required=False, default=1, type=int, help="Augment workers count")
    parser.add_argument("--dest-dir", nargs='?', required=False, help="Destination dir of downloaded files")

    args = parser.parse_args(args=argv)

    logging.info(f"S3          : {args.s3_bucket} from {args.s3_from} to {args.s3_to}")
    logging.info(f"Workers     : {args.workers}")
    logging.info(f"Destination : {args.dest_dir}")

    workers = []
    load_queue = JoinableQueue(maxsize=10)
    error_queue = Queue(maxsize=args.workers)

    for i in range(args.workers):
        p = Process(target=loader, args=(load_queue, error_queue, S3AudioLoader(args.s3_bucket), args.dest_dir))
        p.start()
        workers.append(p)

    for i in tqdm(range(args.s3_from, args.s3_to)):
        load_queue.put(i)

    logging.info("waiting for load to finish")
    for _ in workers:
        load_queue.put(None)
    for p in workers:
        p.join()

    is_err = False
    while not error_queue.empty():
        logging.error(f"Error in worker: {error_queue.get()}")
        is_err = True

    if is_err:
        raise RuntimeError("Errors occurred in workers, check logs for details")

    logging.info(f"Done")


def loader(load_queue, error_queue, file_loader: S3AudioLoader, dest_dir):
    logging.info("Worker started")

    extract_dir = f"{dest_dir}"
    os.makedirs(extract_dir, exist_ok=True)

    is_error = False
    while True:
        (_id) = load_queue.get()
        if _id is None:
            logging.info("Worker exiting")
            break
        tar_name = f"{_id:06d}.tar"
        try:
            if not error_queue.empty():
                is_error = True
            if is_error:
                logging.warning(f"Worker skipping due to previous error {tar_name}")
                continue
            logging.info(f"got task {tar_name}")

            tar_buffer = file_loader.load(tar_name)
            logging.debug(f"Loaded {tar_name} from S3, extracting")
            files = []
            with tarfile.open(fileobj=tar_buffer, mode="r:*") as tar:
                for member in tar.getmembers():
                    if member.isfile() and (member.name.endswith(".wav") or member.name.endswith(".flac")):
                        # Know the name
                        wav_name = member.name
                        logging.info(f"Processing: {wav_name}")
                        local_path = os.path.join(extract_dir, member.name)
                        os.makedirs(Path(local_path).parent, exist_ok=True)
                        with tar.extractfile(member) as src, open(local_path, "wb") as out:
                            shutil.copyfileobj(src, out)
                        files.append(local_path)
                        logging.debug(f"Extracted to: {local_path}")
            logging.debug(f"Saved {tar_name}")

        except Exception as e:
            tb = traceback.format_exc()
            error_queue.put(f"error in {tar_name}: {tb}")
            is_error = True

        finally:
            load_queue.task_done()


if __name__ in {"__main__", "__mp_main__"}:
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    main(sys.argv[1:])
