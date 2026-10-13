# DEF3C: Robust Multi-Source Conditioning for Diffusion-Based Energy Forecasting

## Installation

If Python 3.12 is not installed, install it along with the virtual environment package:

```bash
sudo apt install python3.12
sudo apt install python3.12-venv
```

Create and activate a virtual environment:

```bash
python3.12 -m venv ./venv
source ./venv/bin/activate
```

Install the required packages:

```bash
pip install -r ./requirements.txt
```

### Dataset preparation

Download the **GEFCom2014-Wind** and **GEFCom2014-Solar** datasets from their project sources.

Place the datasets in the following directory structure:

```text
data/
├── GEFCOM2014_WIND/
│   └── raw/
│       ├── Task 1/
│       ├── Task 2/
│       ├── .../
│       ├── Task 15/
│       └── Solution to Task 15/
└── GEFCOM2014_SOLAR/
    └── raw/
        ├── Task 1/
        ├── Task 2/
        ├── .../
        ├── Task 15/
        └── Solution to Task 15/
```

Process the datasets:

```bash
python3.12 ./data/gefcom2014_wind_setup.py
python3.12 ./data/gefcom2014_solar_setup.py
```

Download the **pre-trained model checkpoints** from [Google Drive](https://drive.google.com/drive/folders/1q7c9HQrmy1IeCv1nnWs-LZDO34hgZUz7?usp=sharing) to reproduce the reported results on the GEFCom2014-Wind and GEFCom2014-Solar tasks.

> **Hardware note:** By default, the method runs on a GPU. Ensure that your GPU is compatible with your PyTorch installation, and update the package versions if necessary.

## Evaluate the trained models

### GEFCom2014-Wind

```bash
python3.12 ./replicate_results.py --json ./json/GEFCOM_W/exp_wind_1.json --best_epochs 19 17 12 16 --num_samples 100 --batch_size 8 --sample_chunk 20
```

**Table 1.** Evaluation results for GEFCom2014-Wind: Mean Absolute Error (MAE), Root Mean Square Error (RMSE), coefficient of determination (R²), and Continuous Ranked Probability Score (CRPS).

| Metric | I | II | III | IV | V | VI | VII | VIII | IX | X | All |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| MAE | 0.075 | 0.097 | 0.111 | 0.121 | 0.126 | 0.125 | 0.069 | 0.090 | 0.087 | 0.143 | 0.104 |
| RMSE | 0.110 | 0.144 | 0.151 | 0.184 | 0.182 | 0.184 | 0.098 | 0.148 | 0.127 | 0.215 | 0.154 |
| R² | 0.825 | 0.591 | 0.674 | 0.657 | 0.690 | 0.688 | 0.842 | 0.660 | 0.723 | 0.533 | 0.688 |
| CRPS | 0.213 | 0.244 | 0.220 | 0.283 | 0.230 | 0.221 | 0.192 | 0.269 | 0.267 | 0.259 | 0.240 |

### GEFCom2014-Solar

```bash
python3.12 ./replicate_results.py --json ./json/GEFCOM_S/exp_solar_1.json --best_epochs 12 14 18 13 --num_samples 100 --batch_size 8 --sample_chunk 20
```

**Table 2.** Evaluation results for GEFCom2014-Solar: MAE, RMSE, R², and CRPS.

| Metric | I | II | III | All |
|:--|--:|--:|--:|--:|
| MAE | 0.034 | 0.036 | 0.034 | 0.035 |
| RMSE | 0.079 | 0.080 | 0.075 | 0.078 |
| R² | 0.796 | 0.797 | 0.836 | 0.810 |
| CRPS | 0.267 | 0.274 | 0.224 | 0.255 |

## Execution demos

### Training on GEFCom2014-Wind

```bash
python3.12 ./demo_fit.py --json ./json/GEFCOM_W/exp_wind_5.json
```

### Evaluating a trained GEFCom2014-Wind model

```bash
python3.12 ./demo_eval.py --json ./json/GEFCOM_W/exp_wind_5.json
```

## Citation

If you use any part of this implementation in your research, please cite the following paper:

> C. Symeonidis and N. Nikolaidis, “DEF3C: Robust Multi-Source Conditioning for Diffusion-Based Energy Forecasting,” *Neural Computing and Applications*, accepted for publication, 2026.
