#!/usr/bin/env python3
import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Tuple

from lhotse import RecordingSet, SupervisionSet, CutSet, Recording, SupervisionSegment
from lhotse.utils import Pathlike
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus-dir",
        type=str,
        help="""Corpus dir.
        """,
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="""Output file.
            """,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="""Number of workers to use for processing""",
    )

    return parser.parse_args()


def _parse_utterance(
        part_path: Pathlike,
        lang: Optional[str] = None,
) -> Optional[Tuple[Recording, SupervisionSegment]]:
    recording_id = f"{os.path.basename(os.path.dirname(str(part_path)))}-{os.path.splitext(os.path.basename(str(part_path)))[0]}"
    audio_path = part_path
    if not audio_path.is_file():
        logging.warning(f"No such file: {audio_path}")
        return None

    recording = Recording.from_file(
        path=audio_path,
        recording_id=recording_id,
    )

    segment = SupervisionSegment(
        id=recording_id,
        recording_id=recording_id,
        start=0.0,
        duration=recording.duration,
        channel=0,
        language=lang,
        text="",
    )
    return recording, segment


def iter_audio(root: Path, follow_symlinks: bool = True):
    """
    Recursively yield .wav files.

    Prevents infinite recursion caused by symlink loops by tracking
    visited directories using (device, inode).
    """

    visited = set()

    for dirpath, dirnames, filenames in os.walk(
            root,
            followlinks=follow_symlinks,
    ):
        try:
            stat = os.stat(dirpath)
        except OSError:
            continue

        dir_id = (stat.st_dev, stat.st_ino)

        # Prevent revisiting the same directory via symlink loops
        if dir_id in visited:
            dirnames[:] = []
            continue

        visited.add(dir_id)

        for filename in filenames:
            if filename.lower().endswith(".wav") or filename.lower().endswith(".flac"):
                yield Path(dirpath) / filename


def main():
    args = get_args()

    logging.info(f"collecting files from : {args.corpus_dir}")
    logging.info(f"output                :  {args.output_file}")
    logging.info(f"workers               :  {args.workers}")

    corpus_dir = Path(args.corpus_dir)
    output_file = Path(args.output_file)

    # collect files in dir
    wav_paths = iter_audio(root=corpus_dir)

    process, skip = 0, 0
    with CutSet.open_writer(output_file) as writer:
        with ThreadPoolExecutor(args.workers) as ex:

            futures = []
            for wav_path in tqdm(wav_paths, desc="Distributing tasks"):
                futures.append(ex.submit(_parse_utterance, Path(wav_path), "lt"))

            for future in tqdm(futures, desc="Processing"):
                result = future.result()
                if result is None:
                    continue
                recording, segment = result
                if recording.duration <= 0.2:
                    logging.warning(f"Recording {recording.id} is too short ({recording.duration:.2f} secs) - skipping")
                    skip += 1
                    continue
                one_cutset = CutSet.from_manifests(
                    recordings=RecordingSet.from_recordings([recording]),
                    supervisions=SupervisionSet.from_segments([segment]),
                )
                one_cutset = one_cutset.resample(16000)
                for cut in one_cutset:
                    writer.write(cut)
                    process += 1

    logging.info(f"Collected {process} recordings, skipped too short{skip}")
    logging.info(f"Written cuts to {output_file}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter,
                        level=getattr(logging, os.environ.get("LOGLEVEL", "WARNING").upper(), logging.WARNING))

    logging.info(f"Starting")
    main()
    logging.info(f"Done")
