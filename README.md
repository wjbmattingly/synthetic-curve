# Synthetic Curve Text Generator

This project generates synthetic images of text with curved baselines, along with corresponding ALTO XML files containing baseline information.

## Features

- Generates random text with varying lengths
- Creates images with curved baselines (both subtle and extreme curves)
- Produces corresponding ALTO XML files with baseline coordinates
- Supports left-to-right text orientation
- Includes baseline visualization in the generated images

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the generator script:
```bash
python synthetic_curve_generator.py
```

This will:
1. Create an `output` directory
2. Generate 10 sample images with curved text
3. Create corresponding ALTO XML files for each image

The generated files will be named `sample_XXX.jpg` and `sample_XXX.xml`, where XXX is a three-digit number.

## Output Format

- Images are saved as JPG files
- ALTO XML files contain:
  - Image dimensions
  - Text content
  - Baseline coordinates
  - Text block and line information

## Customization

You can modify the following parameters in the code:
- Image size
- Font size and type
- Curve amplitude and frequency
- Number of samples to generate
- Text length range