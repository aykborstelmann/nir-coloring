from os.path import join
import os


import numpy as np
import pathlib
from pytorch_fid.fid_score import calculate_frechet_distance, \
    calculate_activation_statistics
from pytorch_fid.inception import InceptionV3
import torch
from threadpoolctl import ThreadpoolController

controller = ThreadpoolController()

DEFAULT_ACTIVATION_STATISTICS_FILE = "inception-values.npz"

AVAILABLE_CPUS = len(os.sched_getaffinity(0))
WORKERS = min(AVAILABLE_CPUS, 8)
BATCH_SIZE = 50

DEVICE = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

DIMS = 2048
BLOCK_INDEX = InceptionV3.BLOCK_INDEX_BY_DIM[DIMS]
MODEL = InceptionV3([BLOCK_INDEX]).to(DEVICE)

DEFAULT_MATCHER = "*.png"


def cache_activation_statistics(inception_values_files, mu, sigma):
    np.savez(inception_values_files, mu=mu, sigma=sigma)


def load_activation_statistics_from_cache(inception_values_files):
    with np.load(inception_values_files) as f:
        return f['mu'][:], f['sigma'][:]

def calculate_or_read_activation_statistics(directory, matcher=DEFAULT_MATCHER):
    inception_values_files = join(directory, DEFAULT_ACTIVATION_STATISTICS_FILE)
    if os.path.exists(inception_values_files):
        return load_activation_statistics_from_cache(inception_values_files)

    path = pathlib.Path(directory)
    files = sorted(list(path.glob(matcher)))

    mu, sigma = calculate_activation_statistics(files, MODEL, BATCH_SIZE, DIMS, DEVICE, WORKERS)
    cache_activation_statistics(inception_values_files, mu, sigma)

    return mu, sigma


def calculate_fid(first_directory, second_directory, first_matcher=DEFAULT_MATCHER,
                  second_matcher=DEFAULT_MATCHER):
    mu1, sigma1 = calculate_or_read_activation_statistics(first_directory, first_matcher)
    mu2, sigma2, = calculate_or_read_activation_statistics(second_directory, second_matcher)
    with controller.limit(limits=1, user_api="blas"):
        return calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
