import abc
import asyncio
import datetime
import imghdr
import io
import itertools
import json
import random
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, TypedDict

import aiofiles
import cv2
import pandas as pd
import tqdm.asyncio
from PIL import Image, ImageChops
from aiohttp import ServerTimeoutError
from astral import LocationInfo
from astral.sun import sun
from azure.core.exceptions import ResourceNotFoundError, ServiceRequestError
from azure.storage.blob import BlobClient, StorageStreamDownloader

from nircoloring.config import *

CALTECH_EXCLUDE_CATEGORIES = {30, 33, 97}
SERENGETI_EXCLUDE_CATEGORIES = {0, 1}

DATASET_SIZE = 5000
PARALLEL_DOWNLOAD_COUNT = 40
IMAGE_DOWNLOAD_SIZE = 1024
TRAIN_DATASET_PROPORTION = 0.8

SOUTH_WEST_US_SUNSET_LATEST_LNG = -117
SOUTH_WEST_US_SUNSET_LATEST_LAT = 39
SOUTH_WEST_US_SUNRISE_EARLIEST_LNG = -104
SOUTH_WEST_US_SUNRISE_EARLIEST_LAT = 29

SERENGETI_NATIONAL_PARK_LAT = "2° 19' 59.9988'' S"
SERENGETI_NATIONAL_PARK_LNG = "34° 49' 59.9952'' E"


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


def convert_to_grayscale(img: Image.Image):
    return img.convert("L")


def is_nir_image(filepath):
    with Image.open(filepath) as img:
        r, g, b = img.split()
        return ImageChops.difference(r, g).getbbox() is None and ImageChops.difference(g, b).getbbox() is None


def remove_header_and_footer(img: Image.Image):
    return img.crop((0, 40, img.width, img.height - 100))


def make_crop_and_scale_function(crop_box, resize_dimensions=(IMAGE_DOWNLOAD_SIZE, IMAGE_DOWNLOAD_SIZE)):
    return lambda img: img.crop(crop_box).resize(resize_dimensions, Image.Resampling.LANCZOS)


async def save_image(img: Image.Image, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    bytes_io = io.BytesIO()

    save_image_to_buffer(bytes_io, img)

    async with aiofiles.open(filepath, "wb") as outfile:
        await outfile.write(bytes_io.getbuffer())


def save_image_to_buffer(buffer, img):
    if "exif" in img.info.keys():
        img.save(buffer, format="jpeg", quality=95, exif=img.info.get("exif"))
    else:
        img.save(buffer, format="jpeg", quality=95)


def load_dataset_subset(dataset_subset_file) -> DatasetSubset:
    with open(dataset_subset_file, "r") as dataset_subset_file:
        return json.load(dataset_subset_file)


def download_bytes_from_blob_blocking(url):
    with BlobClient.from_blob_url(url) as client:
        blob: StorageStreamDownloader = client.download_blob()
        return blob.readall()


async def download_bytes_from_blob(url, pool):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(pool, download_bytes_from_blob_blocking, url)


async def fetch_file_from_blob(filename, target_file, url_template, pool, sema=None,
                               in_place_transformation=remove_header_and_footer):
    if not sema:
        sema = asyncio.BoundedSemaphore(1)

    if os.path.exists(target_file) and imghdr.what(target_file) == 'jpeg' and not is_corrupted(target_file):
        return

    blob_url = url_template.format(filename=filename)

    content = await download_bytes_from_blob(blob_url, pool)

    async with sema:
        with io.BytesIO(content) as buffer, Image.open(buffer) as image:
            image = remove_header_and_footer(image)
            if callable(in_place_transformation):
                image = in_place_transformation(image)

        await save_image(image, target_file)


def is_corrupted(target_file):
    try:
        Image.open(target_file).verify()
        cv2.imread(target_file).shape
        return False
    except:
        return True


async def wrap_in_progress_bar(tasks, **tqdm_arguments):
    results = []
    for task in tqdm.asyncio.tqdm.as_completed(tasks, **tqdm_arguments):
        results.append(await task)

    return results


class AbstractMetaDataSource(abc.ABC):

    @abstractmethod
    async def download_file_from_blob(self, filename, filepath, pool, sema=None, in_place_transformation=None):
        pass

    @abstractmethod
    def load_images(self) -> pd.DataFrame:
        pass

    def filter_not_animal_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    @abstractmethod
    def load_annotations(self) -> pd.DataFrame:
        pass


class SerengetiMetaDataSource(AbstractMetaDataSource):

    def __init__(self) -> None:
        self.file_download_template = SNAPSHOT_SERENGETI_DOWNLOAD_IMAGE_URL_TEMPLATE

    async def download_file_from_blob(self, filename, filepath, pool, sema=None, in_place_transformation=None):
        await fetch_file_from_blob(filename, filepath, self.file_download_template, pool, sema=sema,
                                   in_place_transformation=in_place_transformation)

    def load_images(self) -> pd.DataFrame:
        df = pd.DataFrame()
        for meta_data_file in SNAPSHOT_SERENGETI_DATASET_METADATA_FILES:
            with open(meta_data_file, "r") as file:
                metadata = json.load(file)
                metadata = pd.DataFrame(data=metadata["images"])
                df = pd.concat([df, metadata])

        return df

    def load_annotations(self) -> pd.DataFrame:
        df = pd.DataFrame()
        for meta_data_file in SNAPSHOT_SERENGETI_DATASET_METADATA_FILES:
            with open(meta_data_file, "r") as file:
                metadata = json.load(file)
                metadata = pd.DataFrame(data=metadata["annotations"])
                df = pd.concat([df, metadata])

        return df

    def load_categories(self) -> pd.DataFrame:
        df = pd.DataFrame()
        for meta_data_file in SNAPSHOT_SERENGETI_DATASET_METADATA_FILES:
            with open(meta_data_file, "r") as file:
                metadata = json.load(file)
                metadata = pd.DataFrame(data=metadata["categories"])
                df = pd.concat([df, metadata])
        return df.drop_duplicates()

    def filter_not_animal_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        annotations = self.load_annotations()
        annotations["has_animal"] = ~annotations["category_id"].isin(SERENGETI_EXCLUDE_CATEGORIES)
        animal_occurrences = annotations.groupby("image_id")["has_animal"].any()
        df = df.merge(animal_occurrences, how="left", left_on="id", right_on="image_id")
        df = df[df["has_animal"]]
        return df


class CaltechMetaDataSource(AbstractMetaDataSource):

    def __init__(self) -> None:
        self.file_download_template = CALTECH_DOWNLOAD_IMAGE_URL_TEMPLATE

    async def download_file_from_blob(self, filename, filepath, pool,  sema=None, in_place_transformation=None):
        await fetch_file_from_blob(filename, filepath, self.file_download_template, pool, sema=sema,
                                   in_place_transformation=in_place_transformation)

    def load_images(self) -> pd.DataFrame:
        with open(CALTECH_DATASET_METADATA_FILE, "r") as file:
            metadata = json.load(file)

        metadata = pd.DataFrame(data=metadata["images"])
        metadata = metadata.rename(columns={"date_captured": "datetime"})

        return metadata

    def load_annotations(self) -> pd.DataFrame:
        with open(CALTECH_DATASET_METADATA_FILE, "r") as file:
            metadata = json.load(file)

        return pd.DataFrame(data=metadata["annotations"])


class DatasetGenerator(abc.ABC):

    def __init__(self, seed, n, train_fraction, test_fraction, val_fraction, temp_dir,
                 meta_data_source: AbstractMetaDataSource) -> None:
        super().__init__()
        assert train_fraction + test_fraction + val_fraction == 1

        self.train_fraction = train_fraction
        self.test_fraction = test_fraction
        self.val_fraction = val_fraction

        self.seed = seed
        self.n = n

        self.temp_dir = temp_dir

        self.sema = asyncio.BoundedSemaphore(PARALLEL_DOWNLOAD_COUNT)
        self.meta_data_source = meta_data_source
        self.metadata = None

    @abstractmethod
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
        def to_dataset_entry_with_random_crop_box(filename: str) -> DatasetEntry:
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

    async def download_sample(self, sample: str):
        with ThreadPoolExecutor() as pool:
            await self.meta_data_source.download_file_from_blob(sample, join(self.temp_dir, sample), pool)

    def sampler(self):
        while len(self.metadata) > 0:
            sample = self.metadata.sample(weights="weight", random_state=self.seed)
            self.metadata.drop(index=sample.index, inplace=True)
            filename, = sample["file_name"]
            yield filename

    async def is_nir_image(self, sample):
        def is_nir_image_blocking():
            with Image.open(join(self.temp_dir, sample)) as img:
                r, g, b = img.split()
            return ImageChops.difference(r, g).getbbox() is None and ImageChops.difference(g, b).getbbox() is None

        result, = await asyncio.gather(asyncio.to_thread(is_nir_image_blocking))
        return result

    def create_weighted_and_filtered_meta_dataset(self, dataset: pd.DataFrame):
        images = self.filter_dataset(dataset)
        images = self.add_weights(images)
        return images

    def filter_dataset(self, images: pd.DataFrame) -> pd.DataFrame:
        return self.meta_data_source.filter_not_animal_categories(images)

    def add_weights(self, images):
        location_occurrences = images.groupby("location").size()
        weights = 1 / location_occurrences.rename("weight")
        images = images.merge(weights, how="left", on="location")
        return images


class UnalignedNirRgbDatasetGenerator(DatasetGenerator):

    def __init__(self, seed, n, train_fraction, test_fraction, val_fraction, temp_dir,
                 meta_data_source: AbstractMetaDataSource):
        super().__init__(seed, n, train_fraction, test_fraction, val_fraction, temp_dir, meta_data_source)
        self.metadata = None

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

        self.metadata = self.meta_data_source.load_images()
        self.metadata = self.create_weighted_and_filtered_meta_dataset(self.metadata)

        sampler = self.sampler()

        tasks = [self.sample_nir_or_rgb_image(sampler, nir_images, rgb_images) for _ in range(self.n)]

        asyncio.run(wrap_in_progress_bar(tasks, desc="Sampling & analzing dataset"))

        return nir_images, rgb_images

    async def sample_nir_or_rgb_image(self, sampler, nir_images, rgb_images):
        while True:
            async with self.sema:
                sample = await self.sample_and_download(sampler)

                is_nir_image = await self.is_nir_image(sample)
                if is_nir_image and len(nir_images) < self.n * 0.5:
                    nir_images.append(sample)
                    return
                elif not is_nir_image and len(rgb_images) < self.n * 0.5:
                    rgb_images.append(sample)
                    return

    async def sample_and_download(self, sampler):
        while True:
            sample = next(sampler)
            try:
                await self.download_sample(sample)
            except (ResourceNotFoundError, BufferError) as e:
                print(f"Could not download {sample}, got {e}")
                continue

            return sample


def is_in_night(dt: datetime.datetime, sunrise_location: LocationInfo, sunset_location: LocationInfo):
    dawn = sun(sunrise_location.observer, date=dt.date(), tzinfo=sunrise_location.tzinfo)["dawn"]

    dusk = sun(sunset_location.observer, date=dt.date(), tzinfo=sunset_location.tzinfo)["dusk"]

    return dt < dawn or dt > dusk


class UnalignedNirWeightedIncandescentRgbDatasetGenerator(UnalignedNirRgbDatasetGenerator):

    def __init__(self, seed, n, train_fraction, test_fraction, val_fraction, temp_dir,
                 meta_data_source: AbstractMetaDataSource,
                 sunrise_location: LocationInfo, sunset_location: LocationInfo) -> None:
        super().__init__(seed, n, train_fraction, test_fraction, val_fraction, temp_dir, meta_data_source)

        self.sunriseLocation = sunrise_location
        self.sunsetLocation = sunset_location

    def filter_dataset(self, images):
        images = super().filter_dataset(images)

        images["datetime"] = pd.to_datetime(images["datetime"], errors="coerce").dt.tz_localize(
            self.sunriseLocation.timezone,
            ambiguous="NaT",
            nonexistent="NaT")
        images.dropna(subset=["datetime"], inplace=True)

        return images

    def add_weights(self, images):
        images = super().add_weights(images)

        is_night = images.datetime.apply(is_in_night, sunrise_location=self.sunriseLocation,
                                         sunset_location=self.sunsetLocation)
        images["weight"] += images["weight"] * 10000 * is_night

        return images


class UnalignedNirSplitIncandescentRgbDatasetGenerator(UnalignedNirRgbDatasetGenerator):

    def __init__(self, seed, n, train_fraction, test_fraction, val_fraction, temp_dir,
                 meta_data_source: AbstractMetaDataSource,
                 sunrise_location: LocationInfo, sunset_location: LocationInfo) -> None:
        super().__init__(seed, n, train_fraction, test_fraction, val_fraction, temp_dir, meta_data_source)

        self.preferred_nir_images = None
        self.removed_day_images = False
        self.removed_night_images = False
        self.sunriseLocation = sunrise_location
        self.sunsetLocation = sunset_location

    def filter_dataset(self, images):
        images = super().filter_dataset(images)

        images["datetime"] = pd.to_datetime(images["datetime"], errors="coerce").dt.tz_localize(
            self.sunriseLocation.timezone,
            ambiguous="NaT",
            nonexistent="NaT")
        images.dropna(subset=["datetime"], inplace=True)

        images["is_night"] = images.datetime.apply(is_in_night, sunrise_location=self.sunriseLocation,
                                                   sunset_location=self.sunsetLocation)
        return images

    def find_nir_and_rgb_images(self):
        rgb_day_images = []
        rgb_night_images = []
        nir_day_images = []
        nir_night_images = []

        self.metadata = self.meta_data_source.load_images()
        self.metadata = self.create_weighted_and_filtered_meta_dataset(self.metadata)

        sampler = self.sampler()

        tasks = [self.sample_nir_or_rgb_image_day_and_night(sampler, nir_day_images, nir_night_images, rgb_day_images,
                                                            rgb_night_images) for _ in range(self.n)]

        asyncio.run(wrap_in_progress_bar(tasks, desc="Sampling & analyzing dataset"))
        nir_images = list(itertools.chain.from_iterable(zip(nir_day_images, nir_night_images)))
        rgb_images = list(itertools.chain.from_iterable(zip(rgb_day_images, rgb_night_images)))
        return nir_images, rgb_images

    def sampler(self):
        while len(self.metadata) > 0:
            sample = self.metadata.sample(weights="weight", random_state=self.seed)
            self.metadata.drop(index=sample.index, inplace=True)
            filename, = sample["file_name"]
            is_night, = sample["is_night"]
            yield filename, is_night

    async def sample_nir_or_rgb_image_day_and_night(self, sampler, nir_day_images, nir_night_images, rgb_day_images,
                                                    rgb_night_images):
        while True:
            async with self.sema:
                if len(nir_night_images) == int(self.n * 0.25) and len(rgb_night_images) == int(
                        self.n * 0.25) and not self.removed_night_images:
                    self.removed_night_images = True
                    print("Remove night images")
                    self.metadata = self.metadata[~self.metadata["is_night"].astype(bool)]

                if len(nir_day_images) == int(self.n * 0.25) and len(rgb_day_images) == int(
                        self.n * 0.25) and not self.removed_day_images:
                    self.removed_day_images = True
                    print("Remove day images")
                    self.metadata = self.metadata[self.metadata["is_night"]]

                sample, is_night_image = await self.sample_and_download(sampler)

                is_nir_image = await self.is_nir_image(sample)
                if is_nir_image and not is_night_image and len(nir_day_images) < self.n * 0.25:
                    nir_day_images.append(sample)
                    return
                elif is_nir_image and is_night_image and len(nir_night_images) < self.n * 0.25:
                    nir_night_images.append(sample)
                    return
                elif not is_nir_image and not is_night_image and len(rgb_day_images) < self.n * 0.25:
                    rgb_day_images.append(sample)
                    return
                elif not is_nir_image and is_night_image and len(rgb_night_images) < self.n * 0.25:
                    rgb_night_images.append(sample)
                    return

    async def sample_and_download(self, sampler):
        while True:
            file_name, is_night = next(sampler)
            try:
                await self.download_sample(file_name)
            except (ResourceNotFoundError, BufferError, ServiceRequestError, ServerTimeoutError) as e:
                print(f"Could not download {file_name}, got {e}")
                continue

            return file_name, is_night


class UnalignedFilteredNirIncandescentRgbDatasetGenerator(UnalignedNirRgbDatasetGenerator):

    def __init__(self, seed, n, train_fraction, test_fraction, val_fraction, temp_dir,
                 meta_data_source: AbstractMetaDataSource,
                 sunrise_location: LocationInfo, sunset_location: LocationInfo) -> None:
        super().__init__(seed, n, train_fraction, test_fraction, val_fraction, temp_dir, meta_data_source)

        self.sunriseLocation = sunrise_location
        self.sunsetLocation = sunset_location

    def filter_dataset(self, images):
        images = super().filter_dataset(images)

        images["datetime"] = pd.to_datetime(images["datetime"], errors="coerce").dt.tz_localize(
            self.sunriseLocation.timezone,
            ambiguous="NaT",
            nonexistent="NaT")
        images.dropna(subset=["datetime"], inplace=True)

        images["is_night"] = images.datetime.apply(is_in_night, sunrise_location=self.sunriseLocation,
                                                   sunset_location=self.sunsetLocation)

        images = images[images["is_night"]]

        return images


class UnalignedGrayRgbDatasetGenerator(DatasetGenerator):

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
        self.metadata = self.meta_data_source.load_images()
        self.metadata = self.create_weighted_and_filtered_meta_dataset(self.metadata)

        sampler = self.sampler()

        tasks = [self.sample_rgb_image(sampler) for _ in range(self.n)]
        return asyncio.run(wrap_in_progress_bar(tasks, desc="Sampling & analzing dataset"))

    async def sample_rgb_image(self, sampler) -> str:
        async with self.sema:
            sample = next(sampler)
            await self.download_sample(sample)

        if not await self.is_nir_image(sample):
            return sample

        return await self.sample_rgb_image(sampler)


class DatasetDownloader:

    def __init__(self, temp_directory, target_directory, dataset_information_file,
                 meta_data_source: AbstractMetaDataSource,
                 dataset_generator: DatasetGenerator = None) -> None:
        super().__init__()
        self.sema = None
        self.pool = None
        self.meta_data_source = meta_data_source
        self.dataset_subset = None
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

    async def move_from_temp_or_download(self, subdirectory, dataset_entry):
        filename = dataset_entry['filename']
        filename_without_ext, ext = os.path.splitext(filename)
        filename_with_lowercase_ext = filename_without_ext + ext.lower()

        in_place_transformation = self.create_transformation_function(subdirectory, dataset_entry)

        filepath = join(self.target_directory, subdirectory, os.path.basename(filename_with_lowercase_ext))

        if os.path.exists(filepath):
            if not is_corrupted(filepath):
                return
            print(f"Image {filepath} is corrupt")

        tmp_filepath = join(self.temp_directory, filename_with_lowercase_ext)

        if os.path.exists(tmp_filepath):
            async with self.sema:
                with Image.open(tmp_filepath) as img:
                    img = in_place_transformation(img)
                    await save_image(img, filepath)
            return

        await self.meta_data_source.download_file_from_blob(filename, filepath, self.pool, self.sema, in_place_transformation)

    def create_transformation_function(self, subdirectory, dataset_entry):
        return make_crop_and_scale_function(dataset_entry['crop_box'])

    def download_dataset(self, parallel_download_count=PARALLEL_DOWNLOAD_COUNT):
        self.sema = asyncio.BoundedSemaphore(parallel_download_count)
        self.pool = ThreadPoolExecutor(parallel_download_count)

        self.create_or_load_dataset_subset()

        for directory in [join(self.target_directory, subdirectory) for subdirectory in self.dataset_subset.keys()]:
            os.makedirs(directory, exist_ok=True)

        tasks = [
            self.move_from_temp_or_download(directory, dataset_entry)
            for directory, dataset_entries in self.dataset_subset.items()
            for dataset_entry in dataset_entries
        ]
        asyncio.run(wrap_in_progress_bar(tasks, desc="Download or move final files"))


class GrayRgbDatasetDownloader(DatasetDownloader):
    def create_transformation_function(self, subdirectory, dataset_entry):
        crop_and_scale_function = make_crop_and_scale_function(dataset_entry['crop_box'])
        if subdirectory not in ["trainA", "testA", "valA"]:
            return crop_and_scale_function

        return lambda img: convert_to_grayscale(crop_and_scale_function(img))


caltech_downloader = CaltechMetaDataSource()

nir_dataset_generator = UnalignedNirRgbDatasetGenerator(10, 5000, 0.8, 0.1, 0.1, DATASET_TEMP_IMAGES,
                                                        caltech_downloader)
nir_dataset_downloader = DatasetDownloader(DATASET_TEMP_IMAGES, CALTECH_NIR_DATASET_OUT,
                                           CALTECH_NIR_DATASET_SPECIFICATION, caltech_downloader,
                                           nir_dataset_generator)

gray_dataset_generator = UnalignedGrayRgbDatasetGenerator(10, 5000, 0.8, 0.1, 0.1, DATASET_TEMP_IMAGES,
                                                          caltech_downloader)
gray_dataset_downloader = GrayRgbDatasetDownloader(DATASET_TEMP_IMAGES, CALTECH_GRAY_DATASET_OUT,
                                                   CALTECH_GRAY_DATASET_SPECIFICATION, caltech_downloader,
                                                   gray_dataset_generator)

southWestUsSunset = LocationInfo("South West US", "US", "US/Pacific", SOUTH_WEST_US_SUNSET_LATEST_LAT,
                                 SOUTH_WEST_US_SUNSET_LATEST_LNG)
southWestUsSunrise = LocationInfo("South West US", "US", "US/Pacific", SOUTH_WEST_US_SUNRISE_EARLIEST_LAT,
                                  SOUTH_WEST_US_SUNRISE_EARLIEST_LNG)
nir_incandescent_dataset_generator = UnalignedNirWeightedIncandescentRgbDatasetGenerator(10, 5000, 0.8, 0.1, 0.1,
                                                                                         DATASET_TEMP_IMAGES,
                                                                                         caltech_downloader,
                                                                                         southWestUsSunrise,
                                                                                         southWestUsSunset)
nir_incandescent_dataset_downloader = DatasetDownloader(DATASET_TEMP_IMAGES,
                                                        CALTECH_NIR_INCANDESCENT_DATASET_OUT,
                                                        CALTECH_NIR_INCANDESCENT_DATASET_SPECIFICATION,
                                                        caltech_downloader,
                                                        nir_incandescent_dataset_generator)

serengeti_downloader = SerengetiMetaDataSource()

serengeti_location = LocationInfo("serengeti national park", "Tansania", "Africa/Dar_es_Salaam",
                                  SERENGETI_NATIONAL_PARK_LAT,
                                  SERENGETI_NATIONAL_PARK_LNG)
serengeti_nir_incandescent_dataset_generator = UnalignedFilteredNirIncandescentRgbDatasetGenerator(10, 5000, 0.8,
                                                                                                   0.1, 0.1,
                                                                                                   DATASET_TEMP_IMAGES,
                                                                                                   serengeti_downloader,
                                                                                                   serengeti_location,
                                                                                                   serengeti_location)

serengeti_nir_incandescent_dataset_downloader = DatasetDownloader(DATASET_TEMP_IMAGES,
                                                                  SERENGETI_NIR_INCANDESCENT_DATASET_OUT,
                                                                  SERENGETI_NIR_INCANDESCENT_DATASET_SPECIFICATION,
                                                                  serengeti_downloader,
                                                                  serengeti_nir_incandescent_dataset_generator)

serengeti_nir_incandescent_large_dataset_generator = UnalignedFilteredNirIncandescentRgbDatasetGenerator(10, 10000, 0.8,
                                                                                                         0.1, 0.1,
                                                                                                         DATASET_TEMP_IMAGES,
                                                                                                         serengeti_downloader,
                                                                                                         serengeti_location,
                                                                                                         serengeti_location)

serengeti_nir_incandescent_large_dataset_downloader = DatasetDownloader(DATASET_TEMP_IMAGES,
                                                                        SERENGETI_NIR_INCANDESCENT_LARGE_DATASET_OUT,
                                                                        SERENGETI_NIR_INCANDESCENT_LARGE_DATASET_SPECIFICATION,
                                                                        serengeti_downloader,
                                                                        serengeti_nir_incandescent_large_dataset_generator)

serengeti_nir_incandescent_split_dataset_generator = UnalignedNirSplitIncandescentRgbDatasetGenerator(10, 5000, 0.8,
                                                                                                      0.1, 0.1,
                                                                                                      DATASET_TEMP_IMAGES,
                                                                                                      serengeti_downloader,
                                                                                                      serengeti_location,
                                                                                                      serengeti_location)
serengeti_nir_incandescent_split_dataset_downloader = DatasetDownloader(DATASET_TEMP_IMAGES,
                                                                        SERENGETI_NIR_INCANDESCENT_SPLIT_DATASET_OUT,
                                                                        SERENGETI_NIR_INCANDESCENT_SPLIT_DATASET_SPECIFICATION,
                                                                        serengeti_downloader,
                                                                        serengeti_nir_incandescent_split_dataset_generator)

if __name__ == '__main__':
    nir_dataset_downloader.download_dataset()
    gray_dataset_downloader.download_dataset()
    serengeti_nir_incandescent_dataset_downloader.download_dataset()
    serengeti_nir_incandescent_large_dataset_downloader.download_dataset()
    serengeti_nir_incandescent_split_dataset_downloader.download_dataset()
