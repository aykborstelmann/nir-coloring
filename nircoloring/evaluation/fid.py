import hashlib
import os
from os.path import join

import numpy as np
import torch
import torchvision.transforms as TF
from PIL import Image
from cleanfid.features import build_feature_extractor
from cleanfid.fid import frechet_distance, get_batch_features
from cleanfid.resize import build_resizer
from threadpoolctl import ThreadpoolController
from torch.utils.data import TensorDataset, Dataset
from torch.utils.data.dataset import T_co
from tqdm import tqdm

controller = ThreadpoolController()

AVAILABLE_CPUS = len(os.sched_getaffinity(0))
WORKERS = min(AVAILABLE_CPUS, 8)
BATCH_SIZE = 50

DEVICE = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

DEFAULT_MATCHER = "*.png"


class NumpyDataset(Dataset):

    def __init__(self, images, transforms=TF.ToTensor(), mode="clean") -> None:
        super().__init__()
        self.images = images
        self.transforms = transforms
        self.resize = build_resizer(mode)

    def __getitem__(self, index) -> T_co:
        image = self.images[index]

        image = Image.fromarray(image).convert("RGB")
        image = np.array(image)

        image = self.resize(image)

        if image.dtype == "uint8":
            image = self.transforms(np.array(image)) * 255
        elif image.dtype == "float32":
            image = self.transforms(image)

        return image

    def __len__(self):
        return len(self.images)


def cache_activation_statistics(inception_values_files, mu, sigma):
    np.savez(inception_values_files, mu=mu, sigma=sigma)


def load_activation_statistics_from_cache(inception_values_files):
    with np.load(inception_values_files) as f:
        return f['mu'][:], f['sigma'][:]


def get_activations(dataset, mode, clip, batch_size=50, device=DEVICE, num_workers=1, verbose=True, description=""):
    if clip:
        from cleanfid.clip_features import CLIP_fx, img_preprocess_clip
        model = CLIP_fx("ViT-B/32", device)
        dataset.resize = img_preprocess_clip
    else:
        model = build_feature_extractor(mode, device)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size, shuffle=False,
                                             drop_last=False, num_workers=num_workers)

    # collect all inception features
    l_feats = []
    if verbose:
        pbar = tqdm(dataloader, desc=description)
    else:
        pbar = dataloader

    for batch in pbar:
        l_feats.append(get_batch_features(batch, model, device))
    np_feats = np.concatenate(l_feats)
    return np_feats


def calculate_activation_statistics(dataset, mode, clip, batch_size=50, device=DEVICE, num_workers=1):
    act = get_activations(dataset, mode, clip, batch_size, device, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def get_inception_values_file(directory, dataset, clip):
    data_array = np.array([image.numpy() for image in dataset])
    hashed_dataset_and_arguments = hashlib.md5(data_array.tobytes() + bytes(clip))
    return join(directory, f"inception-values-{hashed_dataset_and_arguments.hexdigest()}.npz")


def calculate_or_read_activation_statistics(directory, dataset, mode, clip):
    inception_values_files = get_inception_values_file(directory, dataset, clip)
    if os.path.exists(inception_values_files):
        return load_activation_statistics_from_cache(inception_values_files)

    mu, sigma = calculate_activation_statistics(dataset, mode, clip, BATCH_SIZE, DEVICE, WORKERS)
    cache_activation_statistics(inception_values_files, mu, sigma)

    return mu, sigma


def calculate_or_read_activation_statistics_from_image_array(directory, images, mode="clean", clip=False):
    dataset = NumpyDataset(images, mode=mode)
    return calculate_or_read_activation_statistics(directory, dataset, mode, clip)


def calculate_fid_from_images(first_directory, first_images, second_directory, second_images, mode="clean", clip=False):
    mu1, sigma1 = calculate_or_read_activation_statistics_from_image_array(first_directory, first_images, mode, clip)
    mu2, sigma2 = calculate_or_read_activation_statistics_from_image_array(second_directory, second_images, mode, clip)
    return frechet_distance(mu1, sigma1, mu2, sigma2)
