# BEW-YOLOv8: Multi-scene and Multi-scale Flood Depth Estimation

[![Paper](https://img.shields.io/badge/Paper-Journal%20of%20Hydrology-1f6feb)](https://doi.org/10.1016/j.jhydrol.2024.132139)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.jhydrol.2024.132139-blue)](https://doi.org/10.1016/j.jhydrol.2024.132139)

Official code repository for the BEW-YOLOv8 study:

> Liu, B., Li, Y., Feng, X., & Lian, P. (2024). BEW-YOLOv8: A deep learning model for multi-scene and multi-scale flood depth estimation. *Journal of Hydrology*, 645. https://doi.org/10.1016/j.jhydrol.2024.132139

---

## Overview

BEW-YOLOv8 is a flood-depth estimation framework built on YOLOv8 for **multi-scene** and **multi-scale** conditions.  
The project focuses on estimating flood depth levels from visual observations (e.g., vehicle/person-related scene images) and is relevant to:

- emergency flood assessment,
- disaster monitoring,
- risk analysis, and
- urban resilience studies.

This repository is designed to support academic reproducibility and further development based on the published method.

## Paper Information

**Full citation**

Liu, B., Li, Y., Feng, X., & Lian, P. (2024). BEW-YOLOv8: A deep learning model for multi-scene and multi-scale flood depth estimation. *Journal of Hydrology*, 645. https://doi.org/10.1016/j.jhydrol.2024.132139

**DOI link**  
https://doi.org/10.1016/j.jhydrol.2024.132139

## Method

This project is implemented on top of the Ultralytics YOLOv8 codebase and includes BEW-related model variants under the `BEW/` directory.

According to the paper, BEW-YOLOv8 introduces three main improvements over the baseline YOLOv8 design. In this repository:

- backbone/neck/head-related experiments are organized via YAML model definitions in different subfolders (e.g., BiFPN, AKConv, SE, GAM, etc.);
- IoU-loss variation experiments are handled by alternative loss implementation files in `ultralytics/utils/`.

For technical details, ablation conclusions, and final architecture choices, please refer to the published paper.

## Switching Loss Functions

Loss switching in this repository is file-based.

`ultralytics/utils/` contains:

- `loss.py`
- `loss-Ciou.py`
- `loss-Wiou.py`

To switch the loss function:

1. **Backup** the current `ultralytics/utils/loss.py`.
2. Open the target loss file (e.g., `loss-Ciou.py` or `loss-Wiou.py`).
3. Copy the full content of that file.
4. Replace the full content of `ultralytics/utils/loss.py` with the copied content.
5. Re-run training.

> Important: Keep only one active implementation in `loss.py` during each experiment to avoid ambiguity.

Other architectural replacements can be done in the standard YOLOv8 way by selecting/modifying model YAML files.

## Repository Structure

```text
BEW-YOLOv8/
├── README.md
├── BEW/
│   ├── data/
│   │   ├── flood_data_car1.yaml
│   │   └── flood_data_person.yaml
│   ├── YOLO_AKConv/
│   │   ├── run_akconv.py
│   │   └── yolov8_akconv*.yaml
│   ├── YOLO_BiFPN/
│   │   ├── run_bifpn.py
│   │   └── yolo_bifpn.yaml
│   ├── YOLO_BiFPN_EffectiveSE/
│   ├── YOLO_BoTNet/
│   ├── YOLO_EffectiveSE/
│   ├── YOLO_GAM/
│   ├── YOLO_GCT/
│   ├── YOLO_SA/
│   ├── YOLO_SE/
│   └── test_code/
└── ultralytics/
    ├── cfg/
    ├── models/
    └── utils/
        ├── loss.py
        ├── loss-Ciou.py
        └── loss-Wiou.py
```

## Installation

Because environment lock files are not provided in this repository, set up a YOLOv8-compatible Python environment first, then install Ultralytics dependencies according to your local setup.

A typical setup is:

```bash
conda create -n bew_yolov8 python=3.8 -y
conda activate bew_yolov8
pip install -U pip
pip install ultralytics torch torchvision
```

If you run this repository in editable/local-source mode, adjust package installation according to your local environment and CUDA version.

## Dataset

The dataset used in this study can be downloaded from Baidu Netdisk:

https://pan.baidu.com/s/1xwqGxqM2CydsHTrf78NXlg?pwd=n9g3

Extraction code: `n9g3`

If you cannot access Baidu Netdisk, please contact the first author:

Bo Liu  
Email: trifurs@whu.edu.cn

### Dataset organization notes

- Please reorganize downloaded data into YOLO-format folders (`images/` and corresponding labels) for train/val/test.
- Example dataset YAML files in this repository:
  - `BEW/data/flood_data_car1.yaml`
  - `BEW/data/flood_data_person.yaml`
- The paths inside these YAMLs are local absolute paths from the original development environment; you should modify them to your own machine paths.

## Training

This repository includes multiple training entry scripts for different variants, for example:

```bash
# Example: BiFPN variant
python BEW/YOLO_BiFPN/run_bifpn.py

# Example: AKConv variant
python BEW/YOLO_AKConv/run_akconv.py
```

Before training, please confirm:

- dataset YAML paths are updated to your local paths,
- selected model YAML file matches your target experiment,
- active loss implementation in `ultralytics/utils/loss.py` is the one you intend to use.

## Validation / Evaluation

This repository does not provide a single unified custom validation wrapper at root level.  
You can use standard Ultralytics YOLOv8 validation workflow after training, e.g.:

```bash
yolo detect val model=path/to/best.pt data=path/to/your_data.yaml
```

Please adapt paths and task settings to your experiment.

## Inference / Prediction

Use standard YOLOv8 prediction commands with trained weights, e.g.:

```bash
yolo detect predict model=path/to/best.pt source=path/to/images
```

Or use Python APIs in the same style as the `run_*.py` scripts.

## Model Weights

Pretrained weights specific to BEW-YOLOv8 are not explicitly bundled in this repository as a dedicated release package.  
Users can train models using the provided scripts/configurations and the published dataset link.

## Results

For detailed experimental results, ablation studies, and comparisons with other methods, please refer to the original paper.

## Citation

If you use this repository in your research, please cite:

```bibtex
@article{liu2024bewyolov8,
  title={BEW-YOLOv8: A deep learning model for multi-scene and multi-scale flood depth estimation},
  author={Liu, Bo and Li, Y. and Feng, X. and Lian, P.},
  journal={Journal of Hydrology},
  volume={645},
  pages={132139},
  year={2024},
  doi={10.1016/j.jhydrol.2024.132139}
}
```

## Contact

For questions about the dataset or code, please contact:

Bo Liu  
Email: trifurs@whu.edu.cn

## Acknowledgements

This project is built upon the YOLOv8 framework and the Ultralytics open-source ecosystem. We thank the maintainers and contributors of these foundational projects.

## License

A license file is not currently provided in this repository.

The license will be updated later. Please contact the authors for permission before commercial use or redistribution.
