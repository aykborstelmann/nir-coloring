from os.path import join
import aiofiles
import asyncio
import imghdr
import io
import json
import os
import random
import tqdm.asyncio
import pandas as pd
from PIL import Image, ImageChops
from azure.storage.blob.aio import BlobClient, StorageStreamDownloader
from typing import List, Tuple, TypedDict

from nircoloring.config import DATASET_TRAIN_TEST_SPLIT, DATASET_METADATA_FILE, DATASET_SUBSET, DATA_DIRECTORY, \
    DATASET_TEMP_IMAGES, DATASET_OUT, DATASET_TEST_A, get_dataset_temp_image_file, DATASET_TEST_B, DATASET_TRAIN_A, \
    DATASET_TRAIN_B, CALTECH_DOWNLOAD_IMAGE_URL_TEMPLATE

EXCLUDE_CATEGORIES = {30, 33, 97}

DATASET_SIZE = 5000
PARALLEL_DOWNLOAD_COUNT = 30
IMAGE_DOWNLOAD_SIZE = 1024
TRAIN_DATASET_PROPORTION = 0.8


class DatasetEntry(TypedDict):
    filename: str
    crop_box: Tuple[int, int, int, int]


class DatasetSubset(TypedDict):
    trainA: List[DatasetEntry]
    trainB: List[DatasetEntry]
    testA: List[DatasetEntry]
    testB: List[DatasetEntry]


def create_random_crop_box(filename):
    with Image.open(get_dataset_temp_image_file(filename)) as img:
        crop_size = min(img.width, img.height)

        x_random = random.randrange(0, img.width - crop_size) if img.width - crop_size > 0 else 0
        y_random = random.randrange(0, img.height - crop_size) if img.height - crop_size > 0 else 0

        left = x_random
        top = y_random
        right = crop_size + x_random
        bottom = crop_size + y_random

        return left, top, right, bottom


def create_dataset_subset_with_random_crop_boxes(train_a: List[str],
                                                 train_b: List[str],
                                                 test_a: List[str],
                                                 test_b: List[str]) -> DatasetSubset:
    def to_dataset_entry_with_random_crop_box(filename) -> DatasetEntry:
        return {
            'filename': filename,
            'crop_box': create_random_crop_box(filename)
        }

    return {
        "testA": list(map(to_dataset_entry_with_random_crop_box, test_a)),
        "testB": list(map(to_dataset_entry_with_random_crop_box, test_b)),
        "trainA": list(map(to_dataset_entry_with_random_crop_box, train_a)),
        "trainB": list(map(to_dataset_entry_with_random_crop_box, train_b))
    }


def create_random_dataset_subset(dataset, size=DATASET_SIZE):
    images = pd.DataFrame(data=dataset["images"])

    annotations = pd.DataFrame(data=dataset["annotations"])
    annotations["has_animal"] = ~annotations["category_id"].isin(EXCLUDE_CATEGORIES)
    animal_occurrences = annotations.groupby("image_id")["has_animal"].any()

    images = images.merge(animal_occurrences, how="left", left_on="id", right_on="image_id")
    images = images[images["has_animal"]]

    images = images[~((images["width"] == 800) & (images["height"] == 584))]

    location_occurrences = images.groupby("location").size()
    weights = 1 / location_occurrences.rename("weight")
    images = images.merge(weights, how="left", on="location")

    return images.sample(size, weights="weight")

def is_nir_image(filename):
    with Image.open(get_dataset_temp_image_file(filename)) as img:
        r, g, b = img.split()
        return ImageChops.difference(r, g).getbbox() is None and ImageChops.difference(g, b).getbbox() is None


def get_dimensions(filename):
    with Image.open(get_dataset_temp_image_file(filename)) as img:
        return f"{img.size[0]}x{img.size[1]}"


def remove_header_and_footer(img):
    return img.crop((0, 30, img.width, img.height - 90))


def make_crop_and_scale_function(crop_box, resize_dimensions=(IMAGE_DOWNLOAD_SIZE, IMAGE_DOWNLOAD_SIZE)):
    return lambda img: img.crop(crop_box).resize(resize_dimensions, Image.Resampling.LANCZOS)


def divide_into_nir_and_rgb(filenames):
    nir_images = []
    rgb_images = []
    for filename in filenames:
        if is_nir_image(filename):
            nir_images.append(filename)
        else:
            rgb_images.append(filename)

    return nir_images, rgb_images


def load_filenames():
    with open(DATASET_SUBSET, "r") as file:
        images = file.readlines()
    return [image.strip() for image in images]


def load_metadata():
    with open(DATASET_METADATA_FILE, "r") as file:
        return json.load(file)


def create_random_image_filename_subset_and_save():
    metadata = load_metadata()
    subset = create_random_dataset_subset(metadata)
    filenames = subset["file_name"].str.strip()

    os.makedirs(DATASET_OUT, exist_ok=True)
    with open(DATASET_SUBSET, "w+") as file:
        for filename in filenames:
            file.write(filename + "\n")
    return list(filenames)


async def fetch_or_move(filename, directory, sema, in_place_transformation):
    target_directory = join(DATASET_OUT, directory)
    filepath = join(target_directory, filename)
    if os.path.exists(filepath):
        return

    tmp_filepath = join(DATASET_TEMP_IMAGES, filename)
    if os.path.exists(tmp_filepath):
        async with sema:
            with Image.open(tmp_filepath) as img:
                img = in_place_transformation(img)
                await save_image(img, filepath)
                return

    await fetch_file_from_blob(filename, target_directory, sema, in_place_transformation)


async def save_image(img, filepath):
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    async with aiofiles.open(filepath, "wb") as outfile:
        await outfile.write(buffer.getbuffer())


def create_train_and_test_division(filenames=None):
    if filenames is None:
        filenames = load_filenames()

    nir_images, rgb_images = divide_into_nir_and_rgb(filenames)
    dataset = create_dataset_subset_with_random_crop_boxes(
        train_a=nir_images[:int(len(nir_images) * TRAIN_DATASET_PROPORTION)],
        train_b=rgb_images[:int(len(rgb_images) * TRAIN_DATASET_PROPORTION)],
        test_a=nir_images[int(len(nir_images) * TRAIN_DATASET_PROPORTION):],
        test_b=rgb_images[int(len(rgb_images) * TRAIN_DATASET_PROPORTION):]
    )

    os.makedirs(DATA_DIRECTORY, exist_ok=True)
    with open(DATASET_TRAIN_TEST_SPLIT, "w+") as file:
        json.dump(dataset, file)

    return dataset


def load_dataset_subset() -> DatasetSubset:
    with open(DATASET_TRAIN_TEST_SPLIT, "r") as file:
        return json.load(file)


async def fetch_file_from_blob(filename,
                               directory,
                               sema=asyncio.BoundedSemaphore(1),
                               in_place_transformation=remove_header_and_footer):
    filepath = os.path.join(directory, filename)
    if os.path.exists(filepath) and imghdr.what(filepath) == 'jpeg':
        return

    blob_url = CALTECH_DOWNLOAD_IMAGE_URL_TEMPLATE.format(filename=filename)
    async with sema:
        async with BlobClient.from_blob_url(blob_url) as client:
            downloader: StorageStreamDownloader = await client.download_blob()
            content = await downloader.readall()

        with Image.open(io.BytesIO(content)) as image:
            image = remove_header_and_footer(image)
            if callable(in_place_transformation):
                image = in_place_transformation(image)

        await save_image(image, filepath)

    await fetch_file_from_blob(filename, directory, sema, in_place_transformation)


async def wrap_in_progress_bar(tasks):
    for task in tqdm.asyncio.tqdm.as_completed(tasks):
        await task


def download_files(filenames=None):
    if filenames is None:
        filenames = load_filenames()

    print("Downloading temporary files")

    os.makedirs(DATASET_TEMP_IMAGES, exist_ok=True)

    sema = asyncio.BoundedSemaphore(PARALLEL_DOWNLOAD_COUNT)
    tasks = [fetch_file_from_blob(filename, DATASET_TEMP_IMAGES, sema) for filename in filenames]
    asyncio.run(wrap_in_progress_bar(tasks))

    return filenames


def create_or_load_filenames() -> List[str]:
    if os.path.exists(DATASET_SUBSET):
        return load_filenames()

    return create_random_image_filename_subset_and_save()


def create_or_load_dataset_subset():
    if os.path.exists(DATASET_TRAIN_TEST_SPLIT):
        return load_dataset_subset()

    filenames = create_or_load_filenames()
    download_files(filenames)
    return create_train_and_test_division(filenames)


if __name__ == '__main__':
    dataset_subset = create_or_load_dataset_subset()
    print("Downloading or moving final files")

    os.makedirs(DATASET_TEST_A, exist_ok=True)
    os.makedirs(DATASET_TEST_B, exist_ok=True)
    os.makedirs(DATASET_TRAIN_A, exist_ok=True)
    os.makedirs(DATASET_TRAIN_B, exist_ok=True)

    sema = asyncio.BoundedSemaphore(PARALLEL_DOWNLOAD_COUNT)
    tasks = [
        fetch_or_move(
            filename=dataset_entry['filename'],
            directory=directory, sema=sema,
            in_place_transformation=make_crop_and_scale_function(dataset_entry['crop_box'])
        )
        for directory, dataset_entries in dataset_subset.items()
        for dataset_entry in dataset_entries
    ]
    asyncio.run(wrap_in_progress_bar(tasks))
