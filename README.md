# Capita Selecta Group Assignment 3
TODO FILL IN README.MD
## Installation

To set up the environment for this project, you need to install the following dependencies:

### CUDA Toolkit

To install the CUDA toolkit follow the instructions on the [NVIDIA CUDA Toolkit website](https://developer.nvidia.com/cuda-toolkit) to install the appropriate version for your system.
### cuDNN

To install cuDNN download the cuDNN library follow the instructions on the [NVIDIA cuDNN website](https://developer.nvidia.com/cudnn-downloads)to install the appropriate version for your system.

### PyTorch

To install PyTorch for your cuda version, follow the instructions on the [PyTorch website](https://pytorch.org/get-started/locally/).

### Virtual Environment

To create a virtual environment, run the following command in your terminal:

```bash
python -m venv .venv
```

Activate the virtual environment:

- On Windows:
    ```bash
    .venv\Scripts\activate
    ```
- On macOS and Linux:
    ```bash
    source .venv/bin/activate
    ```

### Environment Variables

Copy the `.env_example` file and rename it to `.env`:

```bash
cp .env_example .env
```

### Install Requirements

With the virtual environment activated, install the required dependencies:

```bash
pip install -r requirements.txt
```