from mpi4py import MPI
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from tqdm import tqdm

#Distributed computation of matrix A
def local_conjugate_gradient(local_A, local_y, local_alpha, local_row_offset, global_p, comm, threshold):
    #Initialize local_alpha with 0 vector so initial local_r = local_y
    local_r = np.copy(local_y)
    local_p = np.copy(local_r)
    squared_error = comm.allreduce(np.dot(local_r, local_r), op=MPI.SUM)
    max_iter = 1000
    curr_iter = 0
    local_sizes = np.array(comm.allgather(len(local_p)))
    offsets = np.array(comm.allgather(local_row_offset))
    while squared_error > threshold:
        #Matrix multiplication
        comm.Allgatherv(local_p, [global_p, local_sizes, offsets, MPI.DOUBLE])
        w = local_A @ global_p
        # s = squared_error/(np.dot(local_p,w))
        s = squared_error/comm.allreduce(np.dot(local_p, w), op=MPI.SUM)
        local_alpha += s*local_p
        local_r -= s*w
        new_squared_error = comm.allreduce(np.dot(local_r, local_r), op=MPI.SUM)
        Beta = new_squared_error/squared_error
        local_p = local_r + Beta*local_p
        squared_error = new_squared_error
        curr_iter += 1
        if (curr_iter > max_iter):
            break
        #if (curr_iter%50 == 0 and rank == 0):
        # print("iter: " + str(curr_iter) + "squared error: " + str(squared_error))
    return local_alpha

def gaussian_kernel_vectorised(x, xi, s):
    diff = x - xi
    squared_norm = np.sum(diff ** 2, axis=1)
    return np.exp(-squared_norm / (2 * s ** 2))

# def gaussian_kernel(x, xi, s):
#     return np.exp((-np.linalg.norm(x - xi) ** 2)/(2 * s ** 2))

def main(): 
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Loading and standardizing data
    if rank == 0:
        scaler = MinMaxScaler()
        print("loading housing data")
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

    #All processes have all data points, might need to initialize array shape in all processes before broadcasting
    X_train = comm.bcast(X_train, root=0)
    y_train = comm.bcast(y_train, root=0)
    X_test  = comm.bcast(X_test,  root=0)
    y_test  = comm.bcast(y_test,  root=0)

    
    # hyper_param_s = 1.5 # Smoothness parameter of Gaussian kernel provided by user
    # hyper_param_lmbda = 0.8 #Regularization parameter of Gaussian kernel provided by user
    # hyper_param_lmbda = 1.40e-2
    # hyper_param_s = 2.5

    ## Coarse-Grained HyperParameter Tuning Range
    ## Best s 1.00e-01, Best lambda 1.00e-01, Best RMSE_test 59770.2649...
    # s_list     = [1e-7*(10**i) for i in range(1,  7)]  # s is a smoothness parameter for Gaussian Kernel. 
    # lmbda_list     = [1e-6*(10**i) for i in range(1,  9)]  # Lmbda is a Regularization parameter for Gaussian Kernel

    ## Fine-Grained HyperParameter Tuning Range
    ## Best s 1.20e-01, Best lambda 1.10e-01, Best RMSE_test 56805.2882...
    s_list     = [3e-2   + 1e-2*i for i in range(10)]  # s is a smoothness parameter for Gaussian Kernel. 
    lmbda_list = [1.1e-1 + 1e-2*i for i in range(10)]  # Lmbda is a Regularization parameter for Gaussian Kernel

    best_s  = 1.0
    best_lmbda  = 1.0
    best_rmse_test   = np.inf
    combinations= [(s, lmbda) for s in s_list for lmbda in lmbda_list]

    for s, lmbda in tqdm(combinations, desc="Gaussian Kernel Tuning...", ascii=False, ncols=70):
        num_total_samples = len(X_train)
        row_offset_per_process = num_total_samples//size
        num_of_rows_in_last_process = row_offset_per_process + (num_total_samples%size)
        number_of_rows_in_process = num_of_rows_in_last_process if rank == size - 1 else row_offset_per_process

        local_A = np.zeros((number_of_rows_in_process, num_total_samples))
        local_y = np.zeros(number_of_rows_in_process)
        local_alpha = np.zeros(number_of_rows_in_process)
        local_row_offset = rank*row_offset_per_process
        global_p = np.zeros(num_total_samples)


        #Local kernel computation
        for i in range(len(local_A)):
            row_num = rank*row_offset_per_process + i
            local_y[i] = y_train[row_num]

            # print("row num", X_train[row_num].shape)
            # print(X_train.shape)
            local_A[i] = (gaussian_kernel_vectorised(X_train[row_num], X_train, s))
            local_A[i, row_num] += lmbda

        local_alpha = local_conjugate_gradient(local_A, local_y, local_alpha, local_row_offset, global_p, comm, 1e-6)

        # Prediction and Test
        local_y_pred = np.zeros(len(y_test))
        if rank == 0:
            global_y_pred = np.zeros(len(y_test))
        else:
            global_y_pred = None

        for i in range(len(y_test)):
            # for j in range(number_of_rows_in_process):
                # row_num = rank*row_offset_per_process + j
                start = rank*row_offset_per_process
                end = start + number_of_rows_in_process
                # print(X_test[i].shape)
                # print(X_train[row_num].shape)
                # local_y_pred[i] += local_alpha[j]*gaussian_kernel(X_test[i], X_train[row_num], hyper_param_s)
                local_y_pred[i] = np.dot(local_alpha, gaussian_kernel_vectorised(X_test[i], X_train[start:end], s))
                # print(local_y_pred[i])
                # print("shape", local_y_pred.shape)
                # print((local_alpha[j]*gaussian_kernel(X_test[i], X_train[row_num], hyper_param_s)).shape)
                # print(np.sum(local_alpha[j]*gaussian_kernel_vectorised(X_test[i], X_train, hyper_param_s)))


        comm.Reduce([local_y_pred, MPI.DOUBLE], [global_y_pred, MPI.DOUBLE], op=MPI.SUM, root=0)

        if rank == 0:
            rmse = root_mean_squared_error(y_test, global_y_pred)
            print("Mean squared error of test set = " + str(rmse))
            if rmse < best_rmse_test:
                best_s = s
                best_lmbda = lmbda
                best_rmse_test  = rmse
    
    if rank == 0: print(f"\nGaussian Kernel ==> Best s {best_s:.2e}, Best lambda {best_lmbda:.2e}, Best RMSE_test {best_rmse_test:.4f}...\n")

        

main()






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
        



    