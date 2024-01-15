from pathlib import PosixPath

from nircoloring.config import ROOT_DIR, SERENGETI_NIR_INCANDESCENT_DATASET_OUT, \
    SERENGETI_NIR_INCANDESCENT_LARGE_DATASET_OUT
from nircoloring.evaluation.utils import EvaluationResult, Result, NpzEvaluationResult
from os.path import join

RESULT_DIR = join(ROOT_DIR, "results")

CYCLE_GAN_RESULT_DIR = join(RESULT_DIR, "cycle-gan")
IHFS_RESULT_DIR = join(RESULT_DIR, "ihfs")
DEOLDIFY_RESULT_DIR = join(RESULT_DIR, "deoldify")

### serengeti-night dataset
serengeti_night_nir_train = Result(
    join(SERENGETI_NIR_INCANDESCENT_DATASET_OUT, "trainA"),
    "NIR (Train)",
    load_size=(128, 128)
)
serengeti_night_nir_test = Result(
    join(SERENGETI_NIR_INCANDESCENT_DATASET_OUT, "testA"),
    "NIR (Test)",
    load_size=(128, 128)
)
serengeti_night_nir_val = Result(
    join(SERENGETI_NIR_INCANDESCENT_DATASET_OUT, "valA"),
    "NIR (Val)",
    load_size=(128, 128)
)
serengeti_night_rgb_train = Result(
    join(SERENGETI_NIR_INCANDESCENT_DATASET_OUT, "trainB"),
    "RGB (Train)",
    load_size=(128, 128)
)
serengeti_night_rgb_test = Result(
    join(SERENGETI_NIR_INCANDESCENT_DATASET_OUT, "testB"),
    "RGB (Test)",
    load_size=(128, 128)
)
serengeti_night_rgb_val = Result(
    join(SERENGETI_NIR_INCANDESCENT_DATASET_OUT, "valB"),
    "RGB (Val)",
    load_size=(128, 128)
)

### serengeti-night-large dataset
serengeti_night_large_nir_train = Result(
    join(SERENGETI_NIR_INCANDESCENT_LARGE_DATASET_OUT, "trainA"),
    "NIR (Train)",
    load_size=(128, 128)
)
serengeti_night_large_nir_test = Result(
    join(SERENGETI_NIR_INCANDESCENT_LARGE_DATASET_OUT, "testA"),
    "NIR (Test)",
    load_size=(128, 128)
)
serengeti_night_large_nir_val = Result(
    join(SERENGETI_NIR_INCANDESCENT_LARGE_DATASET_OUT, "valA"),
    "NIR (Val)",
    load_size=(128, 128)
)
serengeti_night_large_rgb_train = Result(
    join(SERENGETI_NIR_INCANDESCENT_LARGE_DATASET_OUT, "trainB"),
    "RGB (Train)",
    load_size=(128, 128)
)
serengeti_night_large_rgb_test = Result(
    join(SERENGETI_NIR_INCANDESCENT_LARGE_DATASET_OUT, "testB"),
    "RGB (Test)",
    load_size=(128, 128)
)
serengeti_night_large_rgb_val = Result(
    join(SERENGETI_NIR_INCANDESCENT_LARGE_DATASET_OUT, "valB"),
    "RGB (Val)",
    load_size=(128, 128)
)

### IHFS
ihfs_serengeti_night_old = EvaluationResult(
    join(IHFS_RESULT_DIR, "serengeti-night-old"),
    "IHFS SN (Old)",
    fid_reference=serengeti_night_rgb_test,
    load_size=(128, 128),
)

ihfs_serengeti_night = EvaluationResult(
    join(IHFS_RESULT_DIR, "serengeti-night"),
    "IHFS SN",
    fid_reference=serengeti_night_rgb_test,
    load_size=(128, 128),
)
ihfs_serengeti_night_large_val = EvaluationResult(
    join(IHFS_RESULT_DIR, "serengeti-night-large", "sigma_2.4"),
    "IHFS SNL",
    fid_reference=serengeti_night_large_rgb_val,
    load_size=(128, 128),
)
ihfs_serengeti_night_large_test = EvaluationResult(
    join(IHFS_RESULT_DIR, "serengeti-night-large-test"),
    "IHFS SNL",
    fid_reference=serengeti_night_large_rgb_test,
    load_size=(128, 128),
)
ihfs_serengeti_night_large_train = EvaluationResult(
    join(IHFS_RESULT_DIR, "serengeti-night-large-train"),
    "IHFS SNL",
    fid_reference=serengeti_night_large_rgb_train,
    load_size=(128, 128),
)

ihfs_serengeti_night_large_test_sigma_8 = EvaluationResult(
    join(IHFS_RESULT_DIR, "serengeti-night-large-test", "sigma_8.0"),
    "IHFS SNL $\sigma=8$",
    fid_reference=serengeti_night_large_rgb_test,
    load_size=(128, 128),
)


def parse_sigma(folder: PosixPath):
    sigma = folder.name.split("_")[1]
    return float(sigma)


def create_ihfs_sigma_serengeti_night_large_map():
    parent_folder = join(IHFS_RESULT_DIR, "serengeti-night-large")
    available_folders = PosixPath(parent_folder).glob("sigma_*")
    sigma_folder_map = {
        parse_sigma(folder): folder
        for folder in available_folders
    }
    sigma_folder_map = dict(sorted(sigma_folder_map.items(), key=lambda pair: pair[0]))
    sigma_evaluation_map = {
        sigma: EvaluationResult(
            folder,
            f"IHFS SNL {sigma}",
            fid_reference=serengeti_night_large_rgb_val,
            load_size=(128, 128)
        )
        for sigma, folder in sigma_folder_map.items()
    }
    return sigma_evaluation_map


ihfs_sigma_serengeti_night_large_map = create_ihfs_sigma_serengeti_night_large_map()

### IIS
IIS_RESULT_DIR = join(RESULT_DIR, "iis")

iis_serengeti_night_large_test = EvaluationResult(
    join(IIS_RESULT_DIR, "serengeti-night-large-test"),
    "IIS SNL",
    fid_reference=serengeti_night_large_rgb_test,
    load_size=(128, 128),
)

iis_serengeti_night_large_val = EvaluationResult(
    join(IIS_RESULT_DIR, "serengeti-night-large-val"),
    "IIS SNL",
    fid_reference=serengeti_night_large_rgb_val,
    load_size=(128, 128),
)
iis_serengeti_night_large_train = EvaluationResult(
    join(IIS_RESULT_DIR, "serengeti-night-large-train"),
    "IIS SNL",
    fid_reference=serengeti_night_large_rgb_train,
    load_size=(128, 128),
)


iis_cycle_gan_serengeti_night_large_test = EvaluationResult(
    join(IIS_RESULT_DIR, "cycle-gan-serengeti-night-large-test"),
    "IIS CycleGAN",
    fid_reference=serengeti_night_large_rgb_test,
    load_size=(128, 128)
)
iis_cycle_gan_serengeti_night_large_train = EvaluationResult(
    join(IIS_RESULT_DIR, "cycle-gan-serengeti-night-large-train"),
    "IIS CycleGAN",
    fid_reference=serengeti_night_large_rgb_train,
    load_size=(128, 128)
)
iis_cycle_gan_serengeti_night_large_val = EvaluationResult(
    join(IIS_RESULT_DIR, "cycle-gan-serengeti-night-large-val"),
    "IIS CycleGAN",
    fid_reference=serengeti_night_large_rgb_val,
    load_size=(128, 128)
)


### Unconditional
unconditional_serengeti_night_large = NpzEvaluationResult(
    join(RESULT_DIR, "unconditional-serengeti-night-large.npz"),
    title="Unconditional SNL",
    fid_reference=serengeti_night_large_rgb_test,
    load_size=(128, 128),
)

### CycleGAN
cycle_gan_serengeti_night = EvaluationResult(
    join(CYCLE_GAN_RESULT_DIR, "serengeti-night", "test_200", "images"),
    "CycleGAN SN",
    fid_reference=serengeti_night_rgb_test,
    filename_matcher="*_fake.png",
    load_size=(128, 128),
)

cycle_gan_serengeti_night_trained_on_serengeti_night_large = EvaluationResult(
    join(CYCLE_GAN_RESULT_DIR, "serengeti-night-trained-on-serengeti-night-large", "test_200", "images"),
    "CycleGAN SN (trained on SNL)",
    fid_reference=serengeti_night_rgb_test,
    filename_matcher="*_fake.png",
    load_size=(128, 128),
)

cycle_gan_serengeti_night_large_train = EvaluationResult(
    join(CYCLE_GAN_RESULT_DIR, "serengeti-night-large", "train_200", "images"),
    "CycleGAN SNL (Train)",
    fid_reference=serengeti_night_large_rgb_train,
    filename_matcher="*_fake.png",
    load_size=(128, 128),
)

cycle_gan_serengeti_night_large_test = EvaluationResult(
    join(CYCLE_GAN_RESULT_DIR, "serengeti-night-large", "test_200", "images"),
    "CycleGAN SNL (Test)",
    fid_reference=serengeti_night_large_rgb_test,
    filename_matcher="*_fake.png",
    load_size=(128, 128),
)
cycle_gan_serengeti_night_large_val = EvaluationResult(
    join(CYCLE_GAN_RESULT_DIR, "serengeti-night-large", "val_200", "images"),
    "CycleGAN SNL (Val)",
    fid_reference=serengeti_night_large_rgb_val,
    filename_matcher="*_fake.png",
    load_size=(128, 128),
)

### DeOldify
deoldlify_correct_size_serengeti_night_large_test = EvaluationResult(
    join(DEOLDIFY_RESULT_DIR, "serengeti-night-large-test", "correct-size"),
    "DeOldify Correct Size",
    fid_reference=serengeti_night_large_rgb_test,
    load_size=(128, 128),
)
deoldlify_original_size_serengeti_night_large_test = EvaluationResult(
    join(DEOLDIFY_RESULT_DIR, "serengeti-night-large-test", "original-size"),
    "DeOldify Original Size",
    fid_reference=serengeti_night_large_rgb_test,
    load_size=(128, 128),
)
