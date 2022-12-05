from os.path import join

import aiofiles
import asyncio
import imghdr
import io
import json
import os
import pandas as pd
import random
import tqdm.asyncio
from PIL import Image, ImageChops
from azure.storage.blob.aio import BlobClient, StorageStreamDownloader
from typing import List, Tuple, TypedDict

from nircoloring.config import CALTECH_NIR_DATASET_SPECIFICATION, DATASET_METADATA_FILE, DATASET_TEMP_IMAGES, \
    CALTECH_NIR_DATASET_OUT, CALTECH_GRAY_DATASET_OUT, CALTECH_GRAY_DATASET_SPECIFICATION, \
    CALTECH_DOWNLOAD_IMAGE_URL_TEMPLATE

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
    valA: List[DatasetEntry]
    valB: List[DatasetEntry]


def create_weighted_and_filtered_meta_dataset(dataset):
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

    return images


def convert_to_grayscale(img: Image.Image):
    return img.convert("L")


def is_nir_image(filepath):
    with Image.open(filepath) as img:
        r, g, b = img.split()
        return ImageChops.difference(r, g).getbbox() is None and ImageChops.difference(g, b).getbbox() is None


def remove_header_and_footer(img):
    return img.crop((0, 30, img.width, img.height - 90))


def make_crop_and_scale_function(crop_box, resize_dimensions=(IMAGE_DOWNLOAD_SIZE, IMAGE_DOWNLOAD_SIZE)):
    return lambda img: img.crop(crop_box).resize(resize_dimensions, Image.Resampling.LANCZOS)


def load_metadata():
    with open(DATASET_METADATA_FILE, "r") as file:
        return json.load(file)


async def save_image(img, filepath):
    with io.BytesIO() as buffer:
        img.save(buffer, format="JPEG")
        async with aiofiles.open(filepath, "wb") as outfile:
            await outfile.write(buffer.getbuffer())


def load_dataset_subset(dataset_subset_file=CALTECH_NIR_DATASET_SPECIFICATION) -> DatasetSubset:
    with open(dataset_subset_file, "r") as dataset_subset_file:
        return json.load(dataset_subset_file)


async def fetch_file_from_blob(filename,
                               target_file,
                               sema=None,
                               in_place_transformation=remove_header_and_footer):
    if not sema:
        sema = asyncio.BoundedSemaphore(1)

    if os.path.exists(target_file) and imghdr.what(target_file) == 'jpeg':
        return

    blob_url = CALTECH_DOWNLOAD_IMAGE_URL_TEMPLATE.format(filename=filename)
    async with sema:
        async with BlobClient.from_blob_url(blob_url) as client:
            downloader: StorageStreamDownloader = await client.download_blob()
            content = await downloader.readall()

        with io.BytesIO(content) as buffer, Image.open(buffer) as image:
            image = remove_header_and_footer(image)
            if callable(in_place_transformation):
                image = in_place_transformation(image)

        await save_image(image, target_file)


async def wrap_in_progress_bar(tasks, **tqdm_arguments):
    results = []
    for task in tqdm.asyncio.tqdm.as_completed(tasks, **tqdm_arguments):
        results.append(await task)

    return results


class CaltechDatasetGenerator:

    def __init__(self, seed, n, train_fraction, test_fraction, val_fraction, temp_dir) -> None:
        super().__init__()
        assert train_fraction + test_fraction + val_fraction == 1

        self.train_fraction = train_fraction
        self.test_fraction = test_fraction
        self.val_fraction = val_fraction

        self.seed = seed
        self.n = n

        self.temp_dir = temp_dir

        self.sema = asyncio.BoundedSemaphore(PARALLEL_DOWNLOAD_COUNT)

    def generate(self) -> DatasetSubset:
        pass

    def create_random_crop_box(self, filename):
        with Image.open(join(self.temp_dir, filename)) as img:
            crop_size = min(img.width, img.height)

            x_random = random.randrange(0, img.width - crop_size) if img.width - crop_size > 0 else 0
            y_random = random.randrange(0, img.height - crop_size) if img.height - crop_size > 0 else 0

            left = x_random
            top = y_random
            right = crop_size + x_random
            bottom = crop_size + y_random

            return left, top, right, bottom

    def create_dataset_subset_with_random_crop_boxes(self,
                                                     train_a: List[str],
                                                     train_b: List[str],
                                                     test_a: List[str],
                                                     test_b: List[str],
                                                     val_a: List[str],
                                                     val_b: List[str]) -> DatasetSubset:
        def to_dataset_entry_with_random_crop_box(filename) -> DatasetEntry:
            return {
                'filename': filename,
                'crop_box': self.create_random_crop_box(filename)
            }

        return {
            "trainA": list(map(to_dataset_entry_with_random_crop_box, train_a)),
            "trainB": list(map(to_dataset_entry_with_random_crop_box, train_b)),
            "testA": list(map(to_dataset_entry_with_random_crop_box, test_a)),
            "testB": list(map(to_dataset_entry_with_random_crop_box, test_b)),
            "valA": list(map(to_dataset_entry_with_random_crop_box, val_a)),
            "valB": list(map(to_dataset_entry_with_random_crop_box, val_b))
        }

    async def download_sample(self, sample):
        await fetch_file_from_blob(sample, join(self.temp_dir, sample))

    def sampler(self, metadata):
        while len(metadata) > 0:
            sample = metadata.sample(weights="weight", random_state=self.seed)
            metadata.drop(index=sample.index, inplace=True)
            filename, = sample["file_name"]
            yield filename

    def is_nir_image(self, sample):
        with Image.open(join(self.temp_dir, sample)) as img:
            r, g, b = img.split()
        return ImageChops.difference(r, g).getbbox() is None and ImageChops.difference(g, b).getbbox() is None


class CaltechUnalignedNirRgbDatasetGenerator(CaltechDatasetGenerator):

    def generate(self):
        nir_images, rgb_images = self.find_nir_and_rgb_images()

        train_end = int(self.n * 0.5 * self.train_fraction)
        test_end = int(self.n * 0.5 * (self.train_fraction + self.test_fraction))

        return self.create_dataset_subset_with_random_crop_boxes(
            train_a=nir_images[:train_end],
            train_b=rgb_images[:train_end],
            test_a=nir_images[train_end:test_end],
            test_b=rgb_images[train_end:test_end],
            val_a=nir_images[test_end:],
            val_b=rgb_images[test_end:]
        )

    def find_nir_and_rgb_images(self):
        rgb_images = []
        nir_images = []

        metadata = load_metadata()
        metadata = create_weighted_and_filtered_meta_dataset(metadata)

        sampler = self.sampler(metadata)

        tasks = [self.sample_nir_or_rgb_image(sampler, nir_images, rgb_images) for _ in range(self.n)]

        asyncio.run(wrap_in_progress_bar(tasks, desc="Sampling & analzing dataset"))

        return nir_images, rgb_images

    async def sample_nir_or_rgb_image(self, sampler, nir_images, rgb_images):
        async with self.sema:
            sample = next(sampler)

            await self.download_sample(sample)

            is_nir_image = self.is_nir_image(sample)
            if is_nir_image and len(nir_images) < self.n * 0.5:
                nir_images.append(sample)
                return
            elif not is_nir_image and len(rgb_images) < self.n * 0.5:
                rgb_images.append(sample)
                return

            await self.sample_nir_or_rgb_image(sampler, nir_images, rgb_images)


class CaltechUnalignedGrayRgbDatasetGenerator(CaltechDatasetGenerator):

    def generate(self) -> DatasetSubset:
        rgb_images = self.find_rgb_images()

        train_a_end = int(len(rgb_images) * self.train_fraction * 0.5)
        train_b_end = int(len(rgb_images) * self.train_fraction)
        test_a_end = int(len(rgb_images) * (self.train_fraction + self.test_fraction * 0.5))
        test_b_end = int(len(rgb_images) * (self.train_fraction + self.test_fraction))
        val_a_end = int(len(rgb_images) * (self.train_fraction + self.test_fraction + self.val_fraction * 0.5))

        return self.create_dataset_subset_with_random_crop_boxes(
            train_a=rgb_images[:train_a_end],
            train_b=rgb_images[train_a_end:train_b_end],
            test_a=rgb_images[train_b_end:test_a_end],
            test_b=rgb_images[test_a_end:test_b_end],
            val_a=rgb_images[test_b_end:val_a_end],
            val_b=rgb_images[val_a_end:]
        )

    def find_rgb_images(self) -> list[str]:
        metadata = load_metadata()
        metadata = create_weighted_and_filtered_meta_dataset(metadata)

        sampler = self.sampler(metadata)

        tasks = [self.sample_rgb_image(sampler) for _ in range(self.n)]
        return asyncio.run(wrap_in_progress_bar(tasks, desc="Sampling & analzing dataset"))

    async def sample_rgb_image(self, sampler) -> str:
        async with self.sema:
            sample = next(sampler)
            await self.download_sample(sample)

        if not self.is_nir_image(sample):
            return sample

        return await self.sample_rgb_image(sampler)


class CaltechDatasetDownloader:

    def __init__(self, temp_directory, target_directory, dataset_information_file,
                 dataset_generator: CaltechDatasetGenerator = None) -> None:
        super().__init__()
        self.dataset_subset = None
        self.sema = asyncio.BoundedSemaphore(PARALLEL_DOWNLOAD_COUNT)
        self.temp_directory = temp_directory
        self.target_directory = target_directory
        self.train_test_split_file = dataset_information_file
        self.dataset_generator = dataset_generator

    def create_or_load_dataset_subset(self):
        if os.path.exists(self.train_test_split_file):
            self.dataset_subset = load_dataset_subset(self.train_test_split_file)
            return

        self.dataset_subset = self.dataset_generator.generate()

        os.makedirs(os.path.dirname(self.train_test_split_file), exist_ok=True)
        with open(self.train_test_split_file, "w+") as file:
            json.dump(self.dataset_subset, file)

    async def download_file_from_blob(self, filename, filepath, in_place_transformation):
        await fetch_file_from_blob(filename, filepath, self.sema, in_place_transformation)

    async def move_from_temp_or_download(self, subdirectory, dataset_entry):
        filename = dataset_entry['filename']
        in_place_transformation = self.create_transformation_function(subdirectory, dataset_entry)

        filepath = join(self.target_directory, subdirectory, filename)
        if os.path.exists(filepath):
            return

        tmp_filepath = join(self.temp_directory, filename)

        if os.path.exists(tmp_filepath):
            async with self.sema:
                with Image.open(tmp_filepath) as img:
                    img = in_place_transformation(img)
                    await save_image(img, filepath)
            return

        await self.download_file_from_blob(filename, filepath, in_place_transformation)

    def create_transformation_function(self, subdirectory, dataset_entry):
        return make_crop_and_scale_function(dataset_entry['crop_box'])

    def download_dataset(self):
        self.create_or_load_dataset_subset()

        for directory in [join(self.target_directory, subdirectory) for subdirectory in self.dataset_subset.keys()]:
            os.makedirs(directory, exist_ok=True)

        tasks = [
            self.move_from_temp_or_download(directory, dataset_entry)
            for directory, dataset_entries in self.dataset_subset.items()
            for dataset_entry in dataset_entries
        ]
        asyncio.run(wrap_in_progress_bar(tasks, desc="Download or move final files"))


class CaltechGrayRgbDatasetDownloader(CaltechDatasetDownloader):
    def create_transformation_function(self, subdirectory, dataset_entry):
        crop_and_scale_function = make_crop_and_scale_function(dataset_entry['crop_box'])
        if subdirectory not in ["trainA", "testA", "valA"]:
            return crop_and_scale_function

        return lambda img: convert_to_grayscale(crop_and_scale_function(img))


if __name__ == '__main__':
    nir_dataset_generator = CaltechUnalignedNirRgbDatasetGenerator(10, 5000, 0.8, 0.1, 0.1, DATASET_TEMP_IMAGES)
    nir_dataset_downloader = CaltechDatasetDownloader(DATASET_TEMP_IMAGES, CALTECH_NIR_DATASET_OUT,
                                                      CALTECH_NIR_DATASET_SPECIFICATION, nir_dataset_generator)

    gray_dataset_generator = CaltechUnalignedGrayRgbDatasetGenerator(10, 5000, 0.8, 0.1, 0.1, DATASET_TEMP_IMAGES)
    gray_dataset_downloader = CaltechGrayRgbDatasetDownloader(DATASET_TEMP_IMAGES, CALTECH_GRAY_DATASET_OUT,
                                                              CALTECH_GRAY_DATASET_SPECIFICATION,
                                                              gray_dataset_generator)

    nir_dataset_downloader.download_dataset()
    gray_dataset_downloader.download_dataset()
