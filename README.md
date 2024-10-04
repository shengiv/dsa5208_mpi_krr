# DSA5208 Project 1: MPI Kernel Ridge Regression

---

## Team Members

| Name                 | Email Addess       | Student ID |
|:--------------------:|:------------------:|:----------:|
|Wong Zi Xin, Avellin  |e0580073@u.nus.edu  | A0225646B  |
|Name            | Email | ID  |

---

# Deliverables 

- In this submission folder, we have included a few files with remarks as follows:
    - `distcomp.yaml`: Contains the consolidated package names and versions to install within an enclosed conda environment.
    - `housing.tsv`: The provided dataset of California housing prices
    - `README.md`: This current document providing instructions on how to run the MPI pipeline.
    - `krr_mpi_script.py`: The MPI pipeline with Python binding, with which Kernel Ridge Regression will be implemented in distributed manner within MPI environment
    - `summary.pdf`: Pdf file that documents the MPI implementation method, HyperParameter tuning details, and best RMSE result.

---

# Instructions to Run the MPI Pipeline

## Step 1: Set up Conda Environment

- Create a new conda environment called `distcomp` based on packages that are dumped into YAML file below.
- Run below command in the terminal.

```bash
conda env create --file distcomp.yaml
```

## Step 2: Activate `distcomp` Conda Environment

- Run below command in the terminal.

```bash
conda activate distcomp
```

## Step 3: Run MPI Pipeline with Python Binding

- The number of participating processors can be adjusted accordingly by replacing the argument `8` below.
- Please ensure to run below command at same directory level as `housing.tsv` file.
- Run below command in the terminal. 

```bash
mpiexec -n 8 python krr_mpi_script.py
```