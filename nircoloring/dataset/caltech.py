import asyncio
import dataclasses
import imghdr
import io
import json
import os
import random
from dataclasses import dataclass
from typing import Dict

import aiofiles
import tqdm.asyncio
from PIL import Image, ImageChops
from azure.storage.blob.aio import BlobClient, StorageStreamDownloader

from nircoloring.config import *
from nircoloring.config import DATASET_TRAIN_TEST_SPLIT

IMAGE_DOWNLOAD_SIZE = 1024
TRAIN_DATASET_PROPORTION = 0.8


@dataclass
class DatasetSubset:
    trainA: list[str]
    trainB: list[str]
    testA: list[str]
    testB: list[str]


def load_metadata():
    with open(DATASET_METADATA_FILE, "r") as file:
        return json.load(file)


def create_random_database_subset(dataset, size=1000):
    images = dataset["images"]
    images = list(filter(lambda image: not (image["width"] == 800 and image["height"] == 584), images))
    return random.sample(images, size)


def divide_into_nir_and_rgb(filenames):
    nir_images = []
    rgb_images = []
    for filename in filenames:
        if is_nir_image(filename):
            nir_images.append(filename)
        else:
            rgb_images.append(filename)

    return nir_images, rgb_images


def create_random_image_filename_subset_and_save():
    metadata = load_metadata()
    subset = create_random_database_subset(metadata)
    filenames = map(lambda data: data["file_name"].strip(), subset)

    os.makedirs(DATASET_OUT, exist_ok=True)
    with open(DATASET_SUBSET, "w+") as file:
        for filename in filenames:
            file.write(filename + "\n")


async def fetch_or_move(sema, directory, filename):
    target_directory = join(DATASET_OUT, directory)
    filepath = join(target_directory, filename)
    if os.path.exists(filepath):
        return

    async with sema:
        tmp_filepath = join(DATASET_TEMP_IMAGES, filename)
        if os.path.exists(tmp_filepath):
            os.rename(tmp_filepath, filepath)
            return

        await fetch_file_from_blob(asyncio.BoundedSemaphore(1), filename, target_directory)


def create_train_and_test_devision():
    filenames = load_filenames()
    nir_images, rgb_images = divide_into_nir_and_rgb(filenames)
    dataset_subset = DatasetSubset(
        trainA=nir_images[:int(len(nir_images) * TRAIN_DATASET_PROPORTION)],
        trainB=rgb_images[:int(len(rgb_images) * TRAIN_DATASET_PROPORTION)],
        testA=nir_images[int(len(nir_images) * TRAIN_DATASET_PROPORTION):],
        testB=rgb_images[int(len(rgb_images) * TRAIN_DATASET_PROPORTION):]
    )
    os.makedirs(DATA_DIRECTORY, exist_ok=True)
    with open(DATASET_TRAIN_TEST_SPLIT, "w+") as file:
        json.dump(dataclasses.asdict(dataset_subset), file)


def load_filenames():
    with open(DATASET_SUBSET, "r") as file:
        images = file.readlines()
    return [image.strip() for image in images]


def load_dataset_subset():
    with open(DATASET_TRAIN_TEST_SPLIT, "r") as file:
        json_file_as_dict: Dict = json.load(file)
        return DatasetSubset(**json_file_as_dict)


async def fetch_file_from_blob(sema, filename, directory):
    filepath = os.path.join(directory, filename)
    if os.path.exists(filepath) and imghdr.what(filepath) == 'jpeg':
        return

    blob_url = CALTECH_DOWNLOAD_IMAGE_URL_TEMPLATE.format(filename=filename)
    async with sema:
        buffer = io.BytesIO()

        async with BlobClient.from_blob_url(blob_url) as client:
            downloader: StorageStreamDownloader = await client.download_blob()
            content = await downloader.readall()
            image = Image.open(io.BytesIO(content))
            image = crop_square_from_center(image)
            image.save(buffer, format="JPEG")
        async with aiofiles.open(filepath, "wb") as outfile:
            await outfile.write(buffer.getbuffer())

    await fetch_file_from_blob(sema, filename, directory)


def crop_square_from_center(image):
    left = int((image.width - IMAGE_DOWNLOAD_SIZE) / 2)
    top = int((image.height - IMAGE_DOWNLOAD_SIZE) / 2)
    right = int((image.width + IMAGE_DOWNLOAD_SIZE) / 2)
    bottom = int((image.height + IMAGE_DOWNLOAD_SIZE) / 2)
    return image.crop((left, top, right, bottom))


async def wrap_in_progress_bar(tasks):
    for task in tqdm.asyncio.tqdm.as_completed(tasks):
        await task


def download_files():
    os.makedirs(DATASET_TEMP_IMAGES, exist_ok=True)
    images = load_filenames()

    sema = asyncio.BoundedSemaphore(10)
    tasks = [fetch_file_from_blob(sema, filename, DATASET_TEMP_IMAGES) for filename in images]
    asyncio.run(wrap_in_progress_bar(tasks))


def create_dataset_subset_and_download():
    if not os.path.exists(DATASET_SUBSET):
        create_random_image_filename_subset_and_save()
    download_files()


def is_nir_image(filename):
    img = Image.open(get_dataset_image_file(filename))
    r, g, b = img.split()
    return ImageChops.difference(r, g).getbbox() is None and ImageChops.difference(g, b).getbbox() is None


def get_dimensions(filename):
    img = Image.open(get_dataset_image_file(filename))
    return f"{img.size[0]}x{img.size[1]}"


if __name__ == '__main__':
    if not os.path.exists(DATASET_SUBSET):
        create_dataset_subset_and_download()
        create_train_and_test_devision()

    os.makedirs(DATASET_TEST_A, exist_ok=True)
    os.makedirs(DATASET_TEST_B, exist_ok=True)
    os.makedirs(DATASET_TRAIN_A, exist_ok=True)
    os.makedirs(DATASET_TRAIN_B, exist_ok=True)

    dataset_subset = load_dataset_subset()
    sema = asyncio.BoundedSemaphore(10)
    tasks = [
        fetch_or_move(sema, directory, filename)
        for directory, filenames in dataclasses.asdict(dataset_subset).items()
        for filename in filenames
    ]
    asyncio.run(wrap_in_progress_bar(tasks))
