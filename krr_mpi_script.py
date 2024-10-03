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

if rank == 0:
    X_train_shape = X_train.shape
    X_test_shape = X_test.shape
    y_train_shape = y_train.shape
    y_test_shape = y_test.shape
else:
    X_train_shape = None
    X_test_shape = None
    y_train_shape = None
    y_test_shape = None
print("starting shape broadcast")
# Broadcast the shapes
X_train_shape = comm.bcast(X_train_shape, root=0)
X_test_shape = comm.bcast(X_test_shape, root=0)
y_train_shape = comm.bcast(y_train_shape, root=0)
y_test_shape = comm.bcast(y_test_shape, root=0)
print("end of shape broadcast")

# Initialize arrays on all processes
if rank != 0:
    X_train = np.empty(X_train_shape, dtype=np.float64)
    X_test = np.empty(X_test_shape, dtype=np.float64)
    y_train = np.empty(y_train_shape, dtype=np.float64)
    y_test = np.empty(y_test_shape, dtype=np.float64)

print("Starting broadcast")
#All processes have all data points, might need to initialize array shape in all processes before broadcasting
comm.Bcast([X_test , MPI.DOUBLE], root=0)
comm.Bcast([y_test , MPI.DOUBLE], root=0)
comm.Bcast([y_train , MPI.DOUBLE], root=0)
comm.Bcast([X_train , MPI.DOUBLE], root=0)
print("End of broadcast")

#Distributed computation of matrix K
def local_conjugate_gradient(local_A, local_y, local_alpha, comm, threshold):
    local_r = local_y - (local_A @ local_alpha)
    p = np.copy(local_r)
    squared_error = comm.allreduce(np.dot(local_r, local_r), op=MPI.SUM)
    max_iter = 1000
    curr_iter = 0
    while squared_error > threshold:
        w = local_A @ p
        s = squared_error/(np.dot(p,w))
        local_alpha += s*p
        local_r -= s*w
        new_squared_error = comm.allreduce(np.dot(local_r, local_r), op=MPI.SUM)
        Beta = new_squared_error/squared_error
        p = local_r + Beta*p
        squared_error = new_squared_error
        curr_iter += 1
        if (curr_iter > max_iter):
            break
    return local_alpha


def gaussian_kernel(x, xi, s):
    return np.exp((-np.linalg.norm(x - xi) ** 2)/(2 * s ** 2))

num_total_samples = len(X_train)
row_offset_per_process = num_total_samples//size
num_of_rows_in_last_process = row_offset_per_process + (num_total_samples%size)
number_of_rows_in_process = num_of_rows_in_last_process if rank == size - 1 else row_offset_per_process


local_A = None
local_y = None
local_alpha = None

local_A = np.zeros((number_of_rows_in_process, num_total_samples))
local_y = np.zeros(number_of_rows_in_process)
local_alpha = np.zeros(number_of_rows_in_process)

for i in range(len(local_A)):
    row_num = rank*row_offset_per_process + i
    local_y[i] = y_train[row_num]
    for j in range(num_total_samples):
        if row_num == j:
            local_A[i, j] = gaussian_kernel(X_train[row_num], X_train[j], s) + lmbda
        else:
            local_A[i, j] = gaussian_kernel(X_train[row_num], X_train[j], s)

local_alpha = local_conjugate_gradient(local_A, local_y, local_alpha, comm, 1e-6)

local_train_se = 0
local_test_se = 0

local_y_pred = np.zeros(len(y_test))
if rank == 0:
    global_y_pred = np.zeros(len(y_test))
else:
    global_y_pred = None

for i in range(len(y_test)):
    for j in range(number_of_rows_in_process):
        row_num = rank*row_offset_per_process + j
        local_y_pred[i] += local_alpha[j]*gaussian_kernel(X_test[i], X_train[row_num])

comm.Reduce([local_y_pred, MPI.DOUBLE], [global_y_pred, MPI.DOUBLE], op=MPI.SUM, root=0)

if rank == 0:
    mse = mean_squared_error(y_test, global_y_pred)
    print("Mean squared error of test set = " + str(mse))





#Might need barriers before allreduce?


# Gathering all local matrices into the root process
# if rank == 0:
#     A = np.zeros((num_total_samples, num_total_samples))
# else:
#     A = None

# # Gather local matrices
# comm.Gather(local_A, A, root=0)

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
        



    