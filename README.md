# Epistem Land Use Land Cover Classification Algorithm

This repository contains the core backend algorithms and modules for the Epistem land use land cover mapping platform.

## File Structure

- **`src/epistemx/`**: The core Python package for this project. It contains all the backend logic, helper functions, and modules for interacting with Google Earth Engine.
- **`notebooks/`**: Jupyter notebooks used for development, experimentation, and demonstrating the functionality of the core modules.
- **`home.py` & `pages/`**: A minimal Streamlit application for testing and demonstrating the backend algorithms.
- **`environment.yml`**: The environment file for creating a reproducible environment. It lists all necessary Python packages and dependencies.
- **`pyproject.toml`**: The standard Python project configuration file. It defines project metadata and core dependencies for `pip`.

## Getting Started

Follow these instructions to set up and run the project on your local machine.

### 1. Prerequisites

Before you begin, ensure you have the following installed on your system:

- **Git**: A version control system for cloning the repository. [Installation Guide](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).
- **Python environment manager**: If you do not yet have one installed, we recommend [Miniforge](https://github.com/conda-forge/miniforge); it is lightweight, no-frills compared to Anaconda, and works well for this project. If you already have another Conda-compatible manager, you can continue using it.

To confirm these tools are available in your shell, run:
```bash
git --version
conda --version
```

### 2. Set Up the Python Environment

**Recommended (prepackaged Conda-Pack environment)**  

_Prepared on Windows 11 x64; follow this path on Windows 11 x64 machines for the smoothest setup._

1. Download the prepackaged `epistemx` conda-pack archive from [SharePoint](https://icrafcifor.sharepoint.com/:u:/r/sites/EPISTEM/Shared%20Documents/EPISTEM%20Consortium/1%20Monitoring%20Technology/Prototyping/python_environment/epistemx.tar.gz?csf=1&web=1&e=eGbscP). You will need access to the EPISTEM SharePoint workspace.

2. Unpack the archive and make it usable on your machine by following [these instructions](https://gist.github.com/pmbaumgartner/2626ce24adb7f4030c0075d2b35dda32) for restoring a conda-pack environment. In short, place the archive in the directory where you keep your Conda environments, extract it, and run `conda-unpack` inside the environment. Example commands (adapt paths to your platform):

   ```bash
   mkdir -p ~/miniconda3/envs/epistemx
   tar -xzf epistemx.tar.gz -C ~/miniconda3/envs/epistemx
   conda activate ~/miniconda3/envs/epistemx
   conda-unpack
   ```

3. After running `conda-unpack`, reactivate the environment. The `epistemx` environment now includes all Earth Engine dependencies JupyterLab, and Streamlit.

**Alternative (build from `environment.yml`)**  

_Recommended for macOS and Linux systems._

If you prefer to build the environment locally, use the provided `environment.yml` with [Miniforge](https://github.com/conda-forge/miniforge):

```bash
conda create -f environment.yml -n epistemx
conda activate epistemx
```

### 3. Clone the Repository

With the environment ready, clone the project and move into the repository:

```bash
git clone https://github.com/epistem-io/EpistemXBackend.git
cd EpistemXBackend
```

### 4. Usage

Activate the `epistemx` environment (prepackaged or locally built) and use the tooling included in it.

**Running the Jupyter Notebooks**

Before launching, install the `epistemx` package into the active environment so notebooks can import the source modules:

```bash
python -m pip install -e .
```

Launch Jupyter Lab from the project root to explore the project's modules and workflows:

```bash
jupyter lab
```

**Running the Streamlit Application**

The included Streamlit app is a minimal implementation for testing the backend. To launch it, run the following command from the project's root directory:

```bash
streamlit run home.py
```

This will open the application in your default web browser.
