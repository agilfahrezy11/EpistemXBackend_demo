# Epistem Land Use Land Cover Backend Algorithm

This repository contains the core backend algorithms and modules for the Epistem land use land cover mapping platform.

## File Structure Explanation

The repository is organized as follows:

- **`home.py` & `pages/`**: A minimal Streamlit application for testing and demonstrating the backend algorithms.
- **`src/epistemx/`**: The core Python package for this project. It contains all the backend logic, helper functions, and modules for interacting with Google Earth Engine.
- **`notebooks/`**: Jupyter notebooks used for development, experimentation, and demonstrating the functionality of the core modules.
- **`environment.yml`**: The environment file for creating a reproducible environment. It lists all necessary Python packages and dependencies.
- **`pyproject.toml`**: The standard Python project configuration file. It defines project metadata and core dependencies for `pip`.
- **`README.md`**: This file, providing an overview and instructions for the project.

## Getting Started

Follow these instructions to set up and run the project on your local machine.

### 1. Prerequisites

Before you begin, ensure you have the following installed on your system:

- **Git**: A version control system for cloning the repository. [Installation Guide](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).
- **Micromamba**: A lightweight, fast, cross-platform package manager. [Micromamba Installation Guide](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html).

### 2. Installation

**Step 1: Clone the repository**

Open your terminal, navigate to the directory where you want to store the project, and run the following command:

```bash
git clone https://github.com/epistem-io/EpistemXBackend.git
cd EpistemXBackend
```

**Step 2: Create and activate the Micromamba environment**

Use the `environment.yml` file to create a new environment named `epistemx` that contains all the necessary dependencies.

```bash
micromamba create -f environment.yml -n epistemx
micromamba activate epistemx
```

### 3. Usage

Once the environment is activated, you can run the notebooks or the Streamlit application.

**Running the Jupyter Notebooks**

To explore the project's modules and workflows, you can run the Jupyter notebooks:

```bash
# Launch Jupyter Lab (or jupyter notebook) from the project root
jupyter lab
```

**Running the Streamlit Application**

The included Streamlit app is a minimal implementation for testing the backend. To launch it, run the following command from the project's root directory:

```bash
streamlit run home.py
```

This will open the application in your default web browser.
