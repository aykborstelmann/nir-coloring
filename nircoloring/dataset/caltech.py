import asyncio
import json
import random

import aiofiles
import tqdm.asyncio
from azure.storage.blob.aio import BlobClient, StorageStreamDownloader

from nircoloring.config import *


def load_metadata():
    with open(DATASET_METADATA_FILE, "r") as file:
        return json.load(file)


def create_random_database_subset(dataset, size=10000):
    images = dataset["images"]
    return random.sample(images, size)


def create_random_image_filename_subset_and_save():
    metadata = load_metadata()
    subset = create_random_database_subset(metadata)
    filenames = list(map(lambda data: data["file_name"], subset))

    os.makedirs(DATASET_OUT, exist_ok=True)
    with open(DATASET_SUBSET, "w+") as file:
        for filename in filenames:
            file.write(filename + "\n")


def load_filenames():
    with open(DATASET_SUBSET, "r") as file:
        images = file.readlines()
    return images


async def fetch_file(sema, filename):
    filepath = os.path.join(DATASET_IMAGES, filename)
    if os.path.exists(filepath):
        return

    blob_url = CALTECH_DOWNLOAD_IMAGE_URL_TEMPLATE.format(filename=filename)
    async with sema:
        async with BlobClient.from_blob_url(blob_url) as client:
            downloader: StorageStreamDownloader = await client.download_blob()
            content = await downloader.readall()
        async with aiofiles.open(filepath, "wb") as outfile:
            await outfile.write(content)


async def wrap_in_progress_bar(tasks):
    for task in tqdm.asyncio.tqdm.as_completed(tasks):
        await task


def download_files():
    os.makedirs(DATASET_IMAGES, exist_ok=True)
    images = load_filenames()

    sema = asyncio.BoundedSemaphore(5)
    tasks = [fetch_file(sema, url) for url in images]
    asyncio.run(wrap_in_progress_bar(tasks))


def create_dataset_subset_and_download():
    if not os.path.exists(DATASET_SUBSET):
        create_random_image_filename_subset_and_save()
    download_files()


if __name__ == '__main__':
    create_dataset_subset_and_download()
