from os.path import join
import os

import numpy as np
import pathlib

from PIL import Image
from pytorch_fid.fid_score import calculate_frechet_distance, \
    calculate_activation_statistics, ImagePathDataset
from pytorch_fid.inception import InceptionV3
import torch
from threadpoolctl import ThreadpoolController
from torch.nn.functional import adaptive_avg_pool2d
from torch.utils.data import TensorDataset, Dataset
from torch.utils.data.dataset import T_co
from tqdm import tqdm
import torchvision.transforms as TF
from hashlib import sha1

controller = ThreadpoolController()

AVAILABLE_CPUS = len(os.sched_getaffinity(0))
WORKERS = min(AVAILABLE_CPUS, 8)
BATCH_SIZE = 50

DEVICE = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

DIMS = 2048
BLOCK_INDEX = InceptionV3.BLOCK_INDEX_BY_DIM[DIMS]
MODEL = InceptionV3([BLOCK_INDEX]).to(DEVICE)

DEFAULT_MATCHER = "*.png"


class NumpyDataset(Dataset):

    def __init__(self, images, transforms=TF.ToTensor()) -> None:
        super().__init__()
        self.images = images
        self.transforms = transforms

    def __getitem__(self, index) -> T_co:
        image = self.images[index]

        image = Image.fromarray(image).convert("RGB")

        image = self.transforms(image)

        return image

    def __len__(self):
        return len(self.images)


def cache_activation_statistics(inception_values_files, mu, sigma):
    np.savez(inception_values_files, mu=mu, sigma=sigma)


def load_activation_statistics_from_cache(inception_values_files):
    with np.load(inception_values_files) as f:
        return f['mu'][:], f['sigma'][:]


def get_activations(dataset, model, batch_size=50, dims=2048, device='cpu', num_workers=1):
    model.eval()

    if batch_size > len(dataset):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(dataset)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)

    pred_arr = np.empty((len(dataset), dims))

    start_idx = 0

    for batch in tqdm(dataloader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_activation_statistics(dataset, model, batch_size=50, dims=2048, device='cpu', num_workers=1):
    act = get_activations(dataset, model, batch_size, dims, device, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def get_inception_values_file(directory, dataset):
    if isinstance(dataset, ImagePathDataset):
        return join(directory, "inception-values.npz")

    hash = sha1(np.array([image.numpy() for image in dataset]))
    return join(directory, f"inception-values-{hash.hexdigest()}.npz")


def calculate_or_read_activation_statistics(directory, dataset):
    inception_values_files = get_inception_values_file(directory, dataset)
    if os.path.exists(inception_values_files):
        return load_activation_statistics_from_cache(inception_values_files)

    mu, sigma = calculate_activation_statistics(dataset, MODEL, BATCH_SIZE, DIMS, DEVICE, WORKERS)
    cache_activation_statistics(inception_values_files, mu, sigma)

    return mu, sigma


def calculate_or_read_activation_statistics_from_directory_images(directory, matcher=DEFAULT_MATCHER):
    path = pathlib.Path(directory)
    files = sorted(list(path.glob(matcher)))
    dataset = ImagePathDataset(files, transforms=TF.ToTensor())

    return calculate_or_read_activation_statistics(directory, dataset)


def calculate_or_read_activation_statistics_from_image_array(directory, images):
    dataset = NumpyDataset(images)
    return calculate_or_read_activation_statistics(directory, dataset)


def calculate_fid_from_dir(
        first_directory,
        second_directory,
        first_matcher=DEFAULT_MATCHER,
        second_matcher=DEFAULT_MATCHER
):
    mu1, sigma1 = calculate_or_read_activation_statistics_from_directory_images(first_directory, first_matcher)
    mu2, sigma2, = calculate_or_read_activation_statistics_from_directory_images(second_directory, second_matcher)
    with controller.limit(limits=1, user_api="blas"):
        return calculate_frechet_distance(mu1, sigma1, mu2, sigma2)


def calculate_fid_from_images(first_directory, first_images, second_directory, second_images):
    mu1, sigma1 = calculate_or_read_activation_statistics_from_image_array(first_directory, first_images)
    mu2, sigma2 = calculate_or_read_activation_statistics_from_image_array(second_directory, second_images)
    with controller.limit(limits=1, user_api="blas"):
        return calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
