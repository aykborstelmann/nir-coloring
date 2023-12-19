#! /usr/bin/env python

from nircoloring.dataset.downloader import (
    serengeti_nir_incandescent_dataset_downloader, \
    serengeti_nir_incandescent_large_dataset_downloader, \
    serengeti_nir_incandescent_split_dataset_downloader, \
    gray_dataset_downloader, \
    nir_dataset_downloader, \
    nir_incandescent_dataset_downloader, \
    DatasetDownloader, \
    PARALLEL_DOWNLOAD_COUNT)

import argparse

DATASET_NAME_MAP: dict[str, DatasetDownloader] = {
    "serengeti-night": serengeti_nir_incandescent_dataset_downloader,
    "serengeti-night-large": serengeti_nir_incandescent_large_dataset_downloader,
    "serengeti-night-day-split": serengeti_nir_incandescent_split_dataset_downloader,
    "caltech": nir_dataset_downloader,
    "caltech-grayscale": gray_dataset_downloader,
    "caltech-night": nir_incandescent_dataset_downloader
}

parser = argparse.ArgumentParser(
    prog="tools/dataset_downloader.py",
    description="Download Pre-Created Datasets"
)

parser.add_argument("dataset_name", help="the name of the dataset that should be downloaded",
                    choices=DATASET_NAME_MAP.keys())
parser.add_argument("--parallel_downloads", type=int, help="maximum parallel downloads", default=PARALLEL_DOWNLOAD_COUNT)

if __name__ == '__main__':
    args = parser.parse_args()

    dataset_downloader = DATASET_NAME_MAP[args.dataset_name]
    print(f"Downloading {args.dataset_name}")
    dataset_downloader.download_dataset(parallel_download_count=args.parallel_downloads)