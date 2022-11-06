import os
from os.path import join, abspath
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = os.path.join(os.path.dirname(abspath(__file__)), os.pardir)


DATASET_TEMP = os.getenv("DATESET_TEMP") if os.getenv("DATESET_TEMP") is not None else join(ROOT_DIR, "tmp/dataset")
DATASET_TEMP_IMAGES = join(DATASET_TEMP, "images")

DATASET_OUT = os.getenv("DATASET_OUT") if os.getenv("DATASET_OUT") is not None else join(ROOT_DIR, "../CycleGAN", "datasets/caltech")
DATASET_TRAIN_A = join(DATASET_OUT, "trainA")
DATASET_TRAIN_B = join(DATASET_OUT, "trainB")
DATASET_TEST_A = join(DATASET_OUT, "testA")
DATASET_TEST_B = join(DATASET_OUT, "testB")

DATA_DIRECTORY = join(ROOT_DIR, "data/dataset")
DATASET_SUBSET = join(DATA_DIRECTORY, "dataset_subset.txt")
DATASET_TRAIN_TEST_SPLIT = join(DATA_DIRECTORY, "dataset_train_test.json")
DATASET_METADATA_FILE = join(DATA_DIRECTORY, 'caltech_images.json')

CALTECH_DOWNLOAD_IMAGE_URL_TEMPLATE = "https://lilablobssc.blob.core.windows.net/caltech-unzipped/cct_images/{filename}"


def get_dataset_temp_image_file(filename):
    return abspath(join(DATASET_TEMP_IMAGES, filename))
