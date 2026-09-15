# Image Processing Python Tutorials

These runnable Python scripts are cleaned and corrected versions of the book's
examples. They preserve the original topics while fixing formatting,
channel-order, numeric-overflow, indentation, and image-processing issues.

## Setup

Python 3.9 or newer is recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Tutorials

Each script accepts an image path. The source document's example filenames are
shown below, but any compatible image can be used.

```powershell
python python-code/tutorial_01_getting_started.py tree.jpeg
python python-code/tutorial_02_image_basics.py Rose.jpg
python python-code/tutorial_03_intensity_histograms.py flower.jpg
python python-code/tutorial_04_frequency_filtering.py Snow.JPG
python python-code/tutorial_05_noise_models.py Rose.jpg
python python-code/tutorial_06_color_segmentation.py Rose.jpeg
python python-code/tutorial_07_edges_thresholding.py Balloon.jpg
python python-code/tutorial_08_morphology.py broken.tif
```

Run a script with `--help` to see its adjustable parameters. For example:

```powershell
python python-code/tutorial_06_color_segmentation.py Rose.jpeg --lower 0,0,200 --upper 145,90,255
```

The images referenced by the source document are not embedded in the Word or PDF
files, so they must be placed locally or supplied by full path.
