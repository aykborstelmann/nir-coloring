import os
from os.path import join, abspath
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = os.path.join(os.path.dirname(abspath(__file__)), os.pardir)

DATASET_TEMP = os.getenv("DATESET_TEMP") if os.getenv("DATESET_TEMP") is not None else join(ROOT_DIR, "tmp/dataset")
DATASET_TEMP_IMAGES = join(DATASET_TEMP, "images")

CYCLE_GAN_DIR = join(ROOT_DIR, "cycle-gan")
CYCLE_GAN_DIR_RESULTS = join(CYCLE_GAN_DIR, "results")
CYCLE_GAN_DIR_RESULTS_LARGE = join(CYCLE_GAN_DIR, "results-large")

DATA_DIRECTORY = join(ROOT_DIR, "data/dataset")
CALTECH_DATASET_METADATA_FILE = join(DATA_DIRECTORY, 'caltech_images.json')
SNAPSHOT_SERENGETI_DATASET_METADATA_FILES = [join(DATA_DIRECTORY, f'SnapshotSerengetiS0{i}.json') for i in
                                             range(1, 4)]

CALTECH_NIR_DATASET_OUT = os.getenv("DATASET_OUT") if os.getenv("DATASET_OUT") is not None else join(CYCLE_GAN_DIR,
                                                                                                     "datasets/caltech")
CALTECH_NIR_DATASET_SPECIFICATION = join(DATA_DIRECTORY, "nir_dataset_spec.json")

CALTECH_NIR_INCANDESCENT_DATASET_OUT = os.getenv("DATASET_OUT") if os.getenv("DATASET_OUT") is not None else join(
    CYCLE_GAN_DIR, "datasets/caltech-incandescent")
CALTECH_NIR_INCANDESCENT_DATASET_SPECIFICATION = join(DATA_DIRECTORY, "nir_incandescent_dataset_spec.json")

CALTECH_GRAY_DATASET_OUT = os.getenv("DATASET_OUT") if os.getenv("DATASET_OUT") is not None else join(CYCLE_GAN_DIR,
                                                                                                      "datasets/caltech-gray")
CALTECH_GRAY_DATASET_SPECIFICATION = join(DATA_DIRECTORY, "gray_dataset_spec.json")

SERENGETI_NIR_INCANDESCENT_DATASET_OUT = os.getenv("DATASET_OUT") if os.getenv("DATASET_OUT") is not None else join(
    CYCLE_GAN_DIR, "datasets/serengeti-incandescent")
SERENGETI_NIR_INCANDESCENT_DATASET_SPECIFICATION = join(DATA_DIRECTORY, "serengeti_incandescent_dataset_spec.json")

CALTECH_DOWNLOAD_IMAGE_URL_TEMPLATE = "https://lilablobssc.blob.core.windows.net/caltech-unzipped/cct_images/{filename}"
SNAPSHOT_SERENGETI_DOWNLOAD_IMAGE_URL_TEMPLATE = "https://lilablobssc.blob.core.windows.net/snapshotserengeti-unzipped/{filename}"

SUNRISE_API_URL = "https://api.sunrise-sunset.org/json?lat=34&lng=110&date=2011-03-22"
