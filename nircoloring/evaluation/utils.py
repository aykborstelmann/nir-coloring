import os
import pathlib
from os.path import join
from pathlib import PosixPath, Path
from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyiqa
import seaborn as sns
import tqdm
from PIL import Image
from matplotlib import figure, axes
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

from nircoloring.config import ROOT_DIR
from nircoloring.evaluation.fid import calculate_fid_from_images

GRAPHICS_DIR = join(ROOT_DIR, "doc/paper/gfx")

sns.set_theme()


def is_image_file(file: Path):
    if not file.is_file():
        return False

    lowercase_filename = file.name.lower()
    return lowercase_filename.endswith(".jpg") or lowercase_filename.endswith(".png")


def list_image_files_from_directory(directory, matcher="*") -> list[Path]:
    path = pathlib.Path(directory)
    all_filenames = path.glob(matcher)
    all_image_filenames = filter(is_image_file, all_filenames)
    return sorted(list(all_image_filenames))


def load_images(all_image_filenames, load_size) -> list[np.array]:
    def read_image(filename):
        image = Image.open(filename)
        image.thumbnail(load_size, Image.Resampling.LANCZOS)
        image = np.asarray(image)
        return image

    all_images = map(read_image, all_image_filenames)
    return list(all_images)


def load_npz_images(file, load_size) -> list[np.array]:
    images = []
    for image in np.load(file)["arr_0"]:
        pil_image = Image.fromarray(image)
        pil_image.thumbnail(load_size, Image.Resampling.LANCZOS)
        resized_image = np.array(pil_image)

        images.append(resized_image)

    return images


class Result:
    def __init__(self, directory, title, load_size=(64, 64), filename_matcher="*"):
        self.title = title
        self.directory = directory
        self.image_filenames = list_image_files_from_directory(directory, filename_matcher)
        self.images = None
        self.fid = None
        self.load_size = load_size

    def load_images(self) -> List[np.array]:
        if self.images is not None:
            return self.images

        self.images = load_images(self.image_filenames, self.load_size)

        return self.images


class EvaluationResult(Result):
    def __init__(self, directory, title, fid_reference: Result, **args):
        super().__init__(directory, title, **args)
        self.fid_reference = fid_reference
        self.fid = {}

    def load_fid(self, clip=False):
        if clip in self.fid:
            return self.fid[clip]

        self.fid[clip] = calculate_fid_from_images(
            self.directory, self.load_images(),
            self.fid_reference.directory, self.fid_reference.load_images(),
            clip=clip
        )

        return self.fid[clip]


class NpzEvaluationResult(EvaluationResult):
    def __init__(self, file, **args):
        parent_dir = Path(file).parent.absolute()
        self.file = file
        super().__init__(parent_dir, **args)

    def load_images(self) -> List[np.array]:
        if self.images is not None:
            return self.images

        self.images = load_npz_images(self.file, self.load_size)

        return self.images


def plot_grid(results_to_plot: list[Result], columns, rows=4, column_titles=None):
    image_count_to_plot = min(columns * rows, len(results_to_plot[0].load_images()))

    column_size = len(results_to_plot)

    image_columns = column_size * columns

    fig, axes = plt.subplots(nrows=rows, ncols=image_columns, figsize=(image_columns * 2, rows * 2))

    axes_matrix = axes.reshape((rows * columns, column_size)).T

    for result, column_axis in zip(results_to_plot, axes_matrix):
        for image, ax in zip(result.load_images()[:image_count_to_plot], column_axis):
            if len(image.shape) == 2:
                ax.imshow(image, cmap='gray')
            else:
                ax.imshow(image)

            ax.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])

    if column_titles is None:
        column_titles = [result.title for result in results_to_plot]

    assert len(column_titles) == column_size

    for title, ax in zip(column_titles * columns, axes[0]):
        ax.set_title(title)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def plot_diff_heatmap(first: Result, second: Result, columns, rows=4, column_titles=None, cbar_with=0.08,
                      intensity=False, title=None):
    image_count_to_plot = min(columns * rows, len(first.load_images()))
    column_size = 3

    image_columns = column_size * columns

    fig, axes = plt.subplots(nrows=rows, ncols=image_columns + 1,
                             figsize=(image_columns * 2 + cbar_with, rows * 2 + 0.5),
                             gridspec_kw={'width_ratios': [1.] * image_columns + [cbar_with]})

    fig.suptitle(title)
    cbar_axes = axes[:, -1]

    axes_matrix = axes[:, :image_columns].reshape((rows * columns, column_size)).T

    for image, ax in zip(first.load_images()[:image_count_to_plot], axes_matrix[0]):
        if len(image.shape) == 2:
            ax.imshow(image, cmap='gray')
        else:
            if intensity:
                ax.imshow(np.average(image, axis=2), cmap='gray')
            else:
                ax.imshow(image)

        ax.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    for image, ax in zip(second.load_images()[:image_count_to_plot], axes_matrix[1]):
        if len(image.shape) == 2:
            ax.imshow(image, cmap='gray')
        else:
            if intensity:
                ax.imshow(np.average(image, axis=2), cmap='gray')
            else:
                ax.imshow(image)

        ax.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    first_images = np.array(first.load_images()[:image_count_to_plot])
    second_images = np.array(second.load_images()[:image_count_to_plot])

    if intensity:
        first_images = np.average(first_images, axis=3)
        second_images = np.average(second_images, axis=3)

    images_diff = first_images - second_images
    images_diff = np.abs(images_diff)

    if not intensity:
        images_diff = np.sum(images_diff, axis=3)

    vmin = np.min(images_diff)
    vmax = np.percentile(images_diff, 99.9)  # np.max(images_diff)

    for i, (diff, ax) in enumerate(zip(images_diff, axes_matrix[2])):
        is_last_column = (i % columns == columns - 1)
        current_cbar = cbar_axes[int(i / columns)]

        sns.heatmap(diff, ax=ax, vmin=vmin, vmax=vmax, cbar=is_last_column, cbar_ax=current_cbar, square=True)
        ax.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    if column_titles is None:
        column_titles = [first.title, second.title, "Diff"]

    assert len(column_titles) == column_size

    for title, ax in zip(column_titles * columns, axes[0]):
        ax.set_title(title)

    fig.tight_layout()


def plot_grid_latex(results_to_plot: list[Result], columns, fig_title, rows=4, column_titles=None, gfx_folder="gfx"):
    column_size = len(results_to_plot)
    image_columns = column_size * columns

    if column_titles is None:
        column_titles = [result.title for result in results_to_plot]

    file_image_title_pair_list = (
        zip(get_or_create_image_filenames(result), result.load_images(), [column_title] * len(result.load_images()))
        for result, column_title in zip(results_to_plot, column_titles)
    )
    generator = zip(*file_image_title_pair_list)

    assert len(column_titles) == column_size

    row_definitions = []
    commands: List[Tuple[PosixPath, str]] = []

    for row in range(rows):
        row_file_image_title_pair_list = []
        for column in range(columns):
            row_file_image_title_pair_list += list(next(generator))

        corrected_filenames: List[str] = []
        for filename, image, column_title in row_file_image_title_pair_list:
            corrected_filename = f"{gfx_folder}/{fig_title.lower()}/{column_title.lower()}_{filename.name}"

            corrected_filenames.append(corrected_filename)
            commands.append((corrected_filename, image))

        include_graphics_statements = map(lambda filename: f"\includegraphics{{{filename}}}", corrected_filenames)
        row_definition = "    " + " & ".join(include_graphics_statements)
        row_definitions.append(row_definition)

    grid_content = "\\\\\n".join(row_definitions)

    one_column_definition = "Y"
    structure_definition = " ".join([one_column_definition] * image_columns)
    title_definition = "    " + " & ".join(column_titles * columns)

    tabularx_definition = f'''
\\begin{{tabularx}}{{\\textwidth}}{{{structure_definition}}}
{title_definition} \\\\
{grid_content}
\end{{tabularx}}
    '''

    def copy_function(latex_root_path=GRAPHICS_DIR):
        for rel_target_file, image in commands:
            target_file = PosixPath(latex_root_path).joinpath(rel_target_file)
            os.makedirs(target_file.parent.resolve(), exist_ok=True)
            print(f"Creating {target_file.parent.resolve()}")

            target_file = target_file.resolve()
            plt.imsave(target_file, image)
            plt.close()
            print(f"Saving {target_file}")

    return tabularx_definition, copy_function


def plot_diff_heatmap_for_latex(first: Result, second: Result, columns, save_dir, rows=4, intensity=False):
    os.makedirs(save_dir, exist_ok=True)
    image_count_to_plot = min(columns * rows, len(first.load_images()))

    for i, image in enumerate(first.load_images()[:image_count_to_plot]):
        fig, ax = plt.subplots(figsize=(2, 2))  # type: figure.Figure, axes.Axes

        if len(image.shape) == 2:
            ax.imshow(image, cmap='gray')
        else:
            ax.imshow(np.average(image, axis=2), cmap='gray')

        ax.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        fig.tight_layout()
        fig.savefig(f"{save_dir}/first_{i:03}.png", bbox_inches='tight', pad_inches=0)

    for i, image in enumerate(second.load_images()[:image_count_to_plot]):
        fig, ax = plt.subplots(figsize=(2, 2))  # type: figure.Figure, axes.Axes

        if len(image.shape) == 2:
            ax.imshow(image, cmap='gray')
        else:
            ax.imshow(np.average(image, axis=2), cmap='gray')

        ax.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        fig.tight_layout()
        fig.savefig(f"{save_dir}/second_{i:03}.png", bbox_inches='tight', pad_inches=0)

    first_images = np.array(first.load_images()[:image_count_to_plot])
    second_images = np.array(second.load_images()[:image_count_to_plot])

    if intensity:
        first_images = np.average(first_images, axis=3)
        second_images = np.average(second_images, axis=3)

    images_diff = first_images - second_images
    images_diff = np.abs(images_diff)

    if not intensity:
        images_diff = np.sum(images_diff, axis=3)

    vmin = np.min(images_diff)
    vmax = np.percentile(images_diff, 99.9)  # np.max(images_diff)

    for i, diff in enumerate(images_diff):
        is_last_column = (i % columns == columns - 1)

        if is_last_column:
            cbar_fig, cbar_ax = plt.subplots(figsize=((0.075, 1.5)))  # type:figure.Figure, axes.Axes
        else:
            cbar_fig, cbar_ax = None, None

        heatmap_fig, heatmap_ax = plt.subplots(figsize=(2, 2))  # type:figure.Figure, axes.Axes
        sns.heatmap(diff, ax=heatmap_ax, vmin=vmin, vmax=vmax, cbar=is_last_column, cbar_ax=cbar_ax, square=True)

        heatmap_ax.axis('off')
        heatmap_ax.set_xticklabels([])
        heatmap_ax.set_yticklabels([])

        if is_last_column:
            cbar_fig.savefig(f"{save_dir}/cbar_{i:03}.pdf", bbox_inches='tight', pad_inches=0, backend="pgf")

        heatmap_fig.savefig(f"{save_dir}/heatmap_{i:03}.png", bbox_inches='tight', pad_inches=0)


def get_or_create_image_filenames(result: Result):
    if isinstance(result, NpzEvaluationResult):
        return [Path(f"{i:05d}.png") for i in range(len(result.load_images()))]
    return result.image_filenames


def result_fid_to_df(results_to_plot: list[EvaluationResult]):
    data = {
        "FID": [result.load_fid() for result in results_to_plot],
    }
    return pd.DataFrame(data, index=[result.title for result in results_to_plot])


class NumpyDataset(Dataset):
    tensors = None

    def __init__(self, images: List[np.array]):
        to_tensor = ToTensor()
        self.tensors = [to_tensor(image) for image in images]

    def __getitem__(self, index):
        return self.tensors[index]

    def __len__(self):
        return len(self.tensors)


def result_to_quan_df(results_to_plot, metrics=None):
    if metrics is None:
        metrics = ["niqe", "brisque"]

    datasets = (NumpyDataset(result.load_images()) for result in results_to_plot)
    dataloaders = [DataLoader(dataset, batch_size=50) for dataset in datasets]

    metric_results = {
        metric.upper(): apply_metric(dataloaders, metric) for metric in metrics
    }

    data = {
        **metric_results,
        "FID": [result.load_fid() for result in results_to_plot],
    }
    return pd.DataFrame(data, index=[result.title for result in results_to_plot])


def apply_metric(dataloaders, metric_name):
    metric = pyiqa.create_metric(metric_name)

    dataloaders = tqdm.tqdm(dataloaders, f"Evaluating {metric_name}")
    return [apply_metric_for_one_dataloader(dataloader, metric) for dataloader in dataloaders]


def apply_metric_for_one_dataloader(dataloader, metric):
    results = []

    for data in tqdm.tqdm(dataloader):
        results.append(metric(data))

    result = np.array(results).reshape(len(results) * results[0].shape[0])

    return np.average(result)


def plot_fid_bars(df):
    df.plot.bar()
    plt.xticks(rotation=30, ha='right')


def set_size(width_pt, fraction=1, ratio=(5 ** .5 - 1) / 2, subplots=(1, 1)):
    """Set figure dimensions to sit nicely in our document.

    Parameters
    ----------
    width_pt: float
            Document width in points
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * ratio * (subplots[0] / subplots[1])

    plt.figure(figsize=(fig_width_in, fig_height_in))


def set_paper_style():
    plt.rcParams.update(matplotlib.rcParamsDefault)

    sns.set_context("paper")
    sns.set_style("whitegrid")
    sns.axes_style()
    sns.color_palette("hls", 8)

    tex_fonts = {
        # Use LaTeX to write all text
        "pgf.texsystem": "pdflatex",
        "font.family": "Serif",
        "text.usetex": True,

        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 11,
        "font.size": 11,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,

    }

    plt.rcParams.update(tex_fonts)
