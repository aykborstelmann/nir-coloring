# NIR-Coloring

## Getting Started

### TLDR
Create your Anaconda environment

```bash
conda env create -n nir-coloring --file environment.yml
conda activate nir-coloring
```

## Downloading Datasets
```bash
./tools/download_dataset.py <dataset_name>
```

For all available datasets options see
```bash
./tools/download_dataset.py --help
```

To increase the parallel downloads use the `--parallel_downloads` option.

## Downloading Results
Find all results at [here](https://drive.google.com/drive/folders/1vcjp1zyfOcF7_7Kqr9A0n37cijLMu4q6?usp=drive_link).

Download relevant experiments to `results` in the same folder structure.

## Links

### Dataset
We use [Caltech Camera Traps](https://lila.science/datasets/caltech-camera-traps) and [Snapshot Serengeti](https://lila.science/datasets/snapshot-serengeti) as data sources. 

To _create_ datasets from these two subsets, download their metadata files into `data/dataset/`. 
To only download a pre-created dataset, this is not necessary. 

Specifically this means: 
* [CCT Metadata](https://lilablobssc.blob.core.windows.net/caltechcameratraps/labels/caltech_camera_traps.json.zip) -> `data/dataset/caltech_images.json`
* [Snapshot Serengeti Season 1 Metadata](https://lilablobssc.blob.core.windows.net/snapshotserengeti-v-2-0/SnapshotSerengetiS01.json.zip) -> `data/dataset/SnapshotSerengetiS01.json`
* [Snapshot Serengeti Season 2 Metadata](https://lilablobssc.blob.core.windows.net/snapshotserengeti-v-2-0/SnapshotSerengetiS02.json.zip) -> `data/dataset/SnapshotSerengetiS02.json`
* [Snapshot Serengeti Season 3 Metadata](https://lilablobssc.blob.core.windows.net/snapshotserengeti-v-2-0/SnapshotSerengetiS03.json.zip) -> `data/dataset/SnapshotSerengetiS03.json`