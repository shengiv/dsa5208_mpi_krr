from mpi4py import MPI
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
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

for i in range(len(local_A)):
    row_num = rank*row_offset_per_process + i
    for j in range(num_total_samples):
        if row_num == j:
            local_A[i, j] = gaussian_kernel(X_train[row_num], X_train[j], s) + lmbda
        else:
            local_A[i, j] = gaussian_kernel(X_train[row_num], X_train[j], s)

# Gathering all local matrices into the root process
if rank == 0:
    A = np.zeros((num_total_samples, num_total_samples))
else:
    A = None

# Gather local matrices
comm.Gather(local_A, A, root=0)

# The root process now has the complete matrix A = K + lambda*I

# #Conjugate gradient method to calculate alpha 
# if rank == 0:
#     # K += lmbda * np.eye(num_total_samples)

#     # Distributed Conjugate Gradient method
#     def distributed_conjugate_gradient(A, b, x0=None, tol=1e-10, max_iter=1000):
#         if x0 is None: # initial guess of the solution
#             x0 = np.zeros(b.shape)
#         r = b - (A @ x0)
#         p = r.copy()
#         SEold = np.dot(r, r)

#         for _ in range(max_iter):
#             # Local computation of w
#             local_w = np.zeros(len(r))
#             for i in range(len(local_A)):
#                 for j in range(len(p)):
#                     local_w[i] += local_A[i, j] * p[j]
            
#             # Reduce to get global w
#             w = np.zeros_like(r)
#             comm.Allreduce(local_w, w, op=MPI.SUM)

#             # Dot products
#             local_dot_pw = np.dot(p, w)
#             global_dot_pw = comm.allreduce(local_dot_pw, op=MPI.SUM)

#             s = SEold / global_dot_pw
#             x0 += s * p
#             r -= s * w
#             SEnew = np.dot(r, r)
#             SEnew_global = comm.allreduce(SEnew, op=MPI.SUM)

#             if np.sqrt(SEnew_global) < tol:
#                 break

#             beta = SEnew_global / SEold
#             p = r + beta * p
#             SEold = SEnew_global

#         return x0

#     # Solve for alpha using the distributed Conjugate Gradient method
#     alpha = distributed_conjugate_gradient(A, y_train) # Need to tune the max_iter hyperparameter 

# else:
#     alpha = None

# # Broadcast the coefficients alpha to all processes
# alpha = comm.bcast(alpha, root=0)

# # Use alpha for predictions (THIS IS DIRECTLY COPIED FROM CHATGPT W/ NO MODIFICATIONS SO ITS PROBABLY WRONG)
# def predict(X_train, X_new, alpha, s):
#     # Calculate the kernel between training and new data
#     num_train_samples = X_train.shape[0]
#     num_new_samples = X_new.shape[0]
#     K_new = np.zeros((num_new_samples, num_train_samples))

#     for i in range(num_new_samples):
#         for j in range(num_train_samples):
#             K_new[i, j] = gaussian_kernel(X_new[i], X_train[j], s)

#     # Calculate predictions
#     return K_new @ alpha
        



    