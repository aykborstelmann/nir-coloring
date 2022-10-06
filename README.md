# NIR-Coloring

## Getting Started

### TLDR
Create your Anaconda environment

```bash
conda env create --file environment.yml
conda develop -n nir-coloring .
conda activate nir-coloring
```

## Downloading Datasets
### Caltech Camera Traps
```bash
python nircoloring/dataset/caltech.py
```

## Links

### Dataset
[Source](https://lila.science/datasets/caltech-camera-traps)


Extract [Image Level Annotations](https://lilablobssc.blob.core.windows.net/caltechcameratraps/labels/caltech_camera_traps.json.zip) to `data/dataset/caltech_images.json`