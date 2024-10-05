# DSA5208 Project 1: MPI Kernel Ridge Regression

---

## Team Members

| Name                 | Email Addess       | Student ID |
|:--------------------:|:------------------:|:----------:|
|Wong Zi Xin, Avellin  |e0580073@u.nus.edu  | A0225646B  |
|Karthikeyan Vigneshram            | e0689697@u.nus.edu | A0230109W  |

---

# Deliverables 

- In this submission folder, we have included a few files with remarks as follows:
    - `housing.tsv`: The provided dataset of California housing prices
    - `README.md`: This current document providing instructions on how to run the MPI pipeline.
    - `krr_mpi_script.py`: The MPI pipeline with Python binding, with which Kernel Ridge Regression will be implemented in distributed manner within MPI environment
    - `summary.pdf`: Pdf file that documents the MPI implementation method, HyperParameter tuning details, and best RMSE result.

---

# Instructions to Run the MPI Pipeline

## Step 1: Install the relevant Python dependencies using pip install



```bash
python -m pip install SomePackage
#Replace SomePackage with missing package
```

## Step 2: Install mpi4py

- Run below command in the terminal to install MPI for python if not already done so

```bash
pip install mpi4py
pip install impi-rt
```

## Step 3: Run MPI Pipeline 

- Run the below command in the terminal.
- The number of processes can be changed by replacing the argument `8` below.

```bash
mpiexec -n 8 python krr_mpi_script.py
```
