import os
from os.path import join, abspath
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = os.path.join(os.path.dirname(abspath(__file__)), os.pardir)

DATASET_OUT = os.getenv("DATASET_OUT") if os.getenv("DATASET_OUT") is not None else join(ROOT_DIR, "out/dataset/")
DATASET_IMAGES = join(DATASET_OUT, "images")

DATASET_DIRECTORY = join(ROOT_DIR, "data/dataset/")
DATASET_SUBSET = join(DATASET_DIRECTORY, "dataset_subset.txt")
DATASET_METADATA_FILE = join(DATASET_DIRECTORY, 'caltech_images.json')

CALTECH_DOWNLOAD_IMAGE_URL_TEMPLATE = "https://lilablobssc.blob.core.windows.net/caltech-unzipped/cct_images/{filename}"


def get_dataset_image_file(filename):
    return abspath(join(DATASET_IMAGES, filename))
