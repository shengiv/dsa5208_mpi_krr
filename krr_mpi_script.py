from mpi4py import MPI
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

s = float(sys.argv[1]) # Smoothness parameter of Gaussian kernel provided by user
lmbda = float(sys.argv[2]) #Regularization parameter of Gaussian kernel provided by user

# Loading and standardizing data
if rank == 0:
    scaler = StandardScaler()
    data = np.genfromtxt('housing.tsv', delimiter='\t', skip_header=0)
    X = data[:, :9]
    y = data[:, 9]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
else:
    X_train = None
    X_test = None
    y_train = None
    y_test = None

#All processes have all data points
comm.Bcast([X_test , MPI.DOUBLE], root=0)
comm.Bcast([y_test , MPI.DOUBLE], root=0)
comm.Bcast([y_train , MPI.DOUBLE], root=0)
comm.Bcast([X_train , MPI.DOUBLE], root=0)

#Distributed computation of matrix K

def gaussian_kernel(x, xi, s):
    return np.exp((-np.linalg.norm(x - xi) ** 2)/(2 * s ** 2))

num_total_samples = len(X_train)
row_offset_per_process = num_total_samples//size
num_of_rows_in_last_process = row_offset_per_process + (num_total_samples%size)

local_A = None
if rank == size - 1:
    local_A = np.zeros((num_of_rows_in_last_process, num_total_samples))
else:
    local_A = np.zeros((row_offset_per_process, num_total_samples))

for i in range(len(local_K)):
    row_num = rank*row_offset_per_process + i
    for j in range(num_total_samples):
        if row_num == j:
            local_A[i, j] = gaussian_kernel(X_train[row_num], X_train[j], s) + lmbda
        else:
            local_A[i, j] = gaussian_kernel(X_train[row_num], X_train[j], s)

#Conjugate gradient method to calculate alpha 

        



    