from os.path import join
from unittest import TestCase, mock

import json
import os
import shutil
import tempfile
from PIL import Image
from astral import LocationInfo

from dataset.caltech import CaltechDatasetDownloader, DatasetSubset, DatasetEntry, IMAGE_DOWNLOAD_SIZE, \
    CaltechUnalignedNirRgbDatasetGenerator, CaltechUnalignedGrayRgbDatasetGenerator, \
    is_nir_image, CaltechUnalignedNirIncandescentRgbDatasetGenerator, \
    SOUTH_WEST_US_SUNSET_LATEST_LAT, SOUTH_WEST_US_SUNSET_LATEST_LNG, SOUTH_WEST_US_SUNRISE_EARLIEST_LNG, \
    SOUTH_WEST_US_SUNRISE_EARLIEST_LAT


class CaltechDatasetDownloaderTest(TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.test_dir = tempfile.mkdtemp()
        self.train_split_file = join(self.test_dir, "split.json")
        self.target_directory = join(self.test_dir, "target/")
        self.temp_directory = join(self.test_dir, "tmp/")
        self.dataset_generator = mock.Mock()
        self.downloader = CaltechDatasetDownloader(self.temp_directory, self.target_directory, self.train_split_file,
                                                   self.dataset_generator)

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir)

    def test_create_subset(self):
        dataset_subset = DatasetSubset(
            trainA=[DatasetEntry(filename="5968c0f9-23d2-11e8-a6a3-ec086b02610b.jpg", crop_box=(0, 0, 100, 100))]
        )

        self.dataset_generator.generate.return_value = dataset_subset

        self.downloader.create_or_load_dataset_subset()

        self.assertTrue(self.dataset_generator.generate.called)
        self.assertTrue(os.path.exists(self.train_split_file))
        self.assertEquals(json.dumps(self.downloader.dataset_subset), json.dumps(dataset_subset))

    def test_load_subset(self):
        os.makedirs(os.path.dirname(self.train_split_file), exist_ok=True)

        dataset_subset = DatasetSubset(
            trainA=[DatasetEntry(filename="5968c0f9-23d2-11e8-a6a3-ec086b02610b.jpg", crop_box=(0, 0, 100, 100))]
        )

        with open(self.train_split_file, "w+") as file:
            file.write(json.dumps(dataset_subset))

        self.downloader.create_or_load_dataset_subset()
        self.assertEquals(json.dumps(self.downloader.dataset_subset), json.dumps(dataset_subset))

    def test_download_dataset(self):
        os.makedirs(os.path.dirname(self.train_split_file), exist_ok=True)

        filename = "5968c0f9-23d2-11e8-a6a3-ec086b02610b.jpg"
        dataset_subset = DatasetSubset(
            trainA=[DatasetEntry(filename=filename, crop_box=(0, 0, 100, 100))]
        )

        with open(self.train_split_file, "w+") as file:
            file.write(json.dumps(dataset_subset))

        self.downloader.download_dataset()

        filepath = join(self.target_directory, "trainA", filename)
        self.assertTrue(os.path.exists(filepath))
        with Image.open(filepath) as img:
            self.assertEquals(img.size, (IMAGE_DOWNLOAD_SIZE, IMAGE_DOWNLOAD_SIZE))


class CaltechUnalignedNirRgbDatasetGeneratorTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.test_dir = tempfile.mkdtemp()
        self.n = 10
        self.generator = CaltechUnalignedNirRgbDatasetGenerator(1, self.n, 0.6, 0.2, 0.2, self.test_dir)

    def test_find_rgb_and_nir_images(self):
        nir_images, rgb_images = self.generator.find_nir_and_rgb_images()

        assert len(nir_images) == self.n * 0.5

        for file in nir_images:
            filepath = join(self.test_dir, file)
            assert os.path.exists(filepath)
            assert is_nir_image(filepath)

        assert len(rgb_images) == self.n * 0.5

        for file in rgb_images:
            filepath = join(self.test_dir, file)
            assert os.path.exists(filepath)
            assert not is_nir_image(filepath)

    def test_generate(self):
        dataset = self.generator.generate()

        assert len(dataset['trainA']) == 3
        assert len(dataset['trainB']) == 3
        assert len(dataset['testA']) == 1
        assert len(dataset['testB']) == 1
        assert len(dataset['valA']) == 1
        assert len(dataset['valB']) == 1

        for entry in dataset['trainA'] + dataset['testA'] + dataset['valA']:
            filepath = join(self.test_dir, entry['filename'])
            assert os.path.exists(filepath)
            assert is_nir_image(filepath)

        for entry in dataset['trainB'] + dataset['testB'] + dataset['valB']:
            filepath = join(self.test_dir, entry['filename'])
            assert os.path.exists(filepath)
            assert not is_nir_image(filepath)


class CaltechUnalignedNirIncandescentRgbDatasetGeneratorTest(TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.test_dir = tempfile.mkdtemp()
        self.n = 10

        self.southWestUsSunset = LocationInfo("South West US", "US", "US/Pacific", SOUTH_WEST_US_SUNSET_LATEST_LAT,
                                              SOUTH_WEST_US_SUNSET_LATEST_LNG)
        self.southWestUsSunrise = LocationInfo("South West US", "US", "US/Pacific", SOUTH_WEST_US_SUNRISE_EARLIEST_LAT,
                                               SOUTH_WEST_US_SUNRISE_EARLIEST_LNG)
        self.generator = CaltechUnalignedNirIncandescentRgbDatasetGenerator(1, self.n, 0.6, 0.2, 0.2, self.test_dir,
                                                                            self.southWestUsSunrise, self.southWestUsSunset)

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir)

    def test_find_rgb_and_nir_images(self):
        nir_images, rgb_images = self.generator.find_nir_and_rgb_images()

        assert len(nir_images) == self.n * 0.5

        for file in nir_images:
            filepath = join(self.test_dir, file)
            assert os.path.exists(filepath)
            assert is_nir_image(filepath)

        assert len(rgb_images) == self.n * 0.5

        for file in rgb_images:
            filepath = join(self.test_dir, file)
            assert os.path.exists(filepath)
            assert not is_nir_image(filepath)

    def test_generate(self):
        dataset = self.generator.generate()

        assert len(dataset['trainA']) == 3
        assert len(dataset['trainB']) == 3
        assert len(dataset['testA']) == 1
        assert len(dataset['testB']) == 1
        assert len(dataset['valA']) == 1
        assert len(dataset['valB']) == 1

        for entry in dataset['trainA'] + dataset['testA'] + dataset['valA']:
            filepath = join(self.test_dir, entry['filename'])
            assert os.path.exists(filepath)
            assert is_nir_image(filepath)

        for entry in dataset['trainB'] + dataset['testB'] + dataset['valB']:
            filepath = join(self.test_dir, entry['filename'])
            assert os.path.exists(filepath)
            assert not is_nir_image(filepath)


class CaltechUnalignedGrayRgbDatasetGeneratorTest(TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.test_dir = tempfile.mkdtemp()
        self.n = 10
        self.generator = CaltechUnalignedGrayRgbDatasetGenerator(1, self.n, 0.6, 0.2, 0.2, self.test_dir)

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir)

    def test_find_rgb_files(self):
        rgb_files = self.generator.find_rgb_images()

        assert len(rgb_files) == self.n
        for file in rgb_files:
            filepath = join(self.test_dir, file)
            assert os.path.exists(filepath)
            assert not is_nir_image(filepath)

    def test_generate(self):
        dataset = self.generator.generate()

        assert len(dataset['trainA']) == 3
        assert len(dataset['trainB']) == 3
        assert len(dataset['testA']) == 1
        assert len(dataset['testB']) == 1
        assert len(dataset['valA']) == 1
        assert len(dataset['valB']) == 1
