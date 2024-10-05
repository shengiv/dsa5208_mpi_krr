from mpi4py import MPI
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from tqdm import tqdm
import sys

# Distributed computation of conjugate gradient
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
        # Matrix-vector multiplication
        comm.Allgatherv(local_p, [global_p, local_sizes, offsets, MPI.DOUBLE])
        w = local_A @ global_p
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
    return local_alpha

def gaussian_kernel_vectorised(x, xi, s):
    diff = x - xi
    squared_norm = np.sum(diff ** 2, axis=1)
    return np.exp(-squared_norm / (2 * s ** 2))


def main(): 
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    hyperparamTuning = int(sys.argv[1]) # 0 to run with the best hyperparameters found previously, 1 to tune 
    if hyperparamTuning != 0 and hyperparamTuning != 1: 
        if rank == 0: 
            print("Error -- last argument can only be 0 or 1")
        return

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

    # All processes have all data points
    X_train = comm.bcast(X_train, root=0)
    y_train = comm.bcast(y_train, root=0)
    X_test  = comm.bcast(X_test,  root=0)
    y_test  = comm.bcast(y_test,  root=0)

    ## Coarse-Grained HyperParameter Tuning Range
    ## Best s 1.00e-01, Best lambda 1.00e-01, Best RMSE_train 39133.2197, Best RMSE_test 59770.2649...
    # s_list     = [1e-7*(10**i) for i in range(1,  7)]  # s is a smoothness parameter for Gaussian Kernel. 
    # lmbda_list     = [1e-6*(10**i) for i in range(1,  9)]  # Lmbda is a Regularization parameter for Gaussian Kernel

    ## Fine-Grained HyperParameter Tuning Range
    ## Best s 1.20e-01, Best lambda 1.10e-01, Best RMSE_train 42310.3940, Best RMSE_test 56805.2882...
    s_list     = [3e-2   + 1e-2*i for i in range(10)]  # s is a smoothness parameter for Gaussian Kernel. 
    lmbda_list = [1.1e-1 + 1e-2*i for i in range(10)]  # Lmbda is a Regularization parameter for Gaussian Kernel

    best_s  = 1.0
    best_lmbda  = 1.0
    best_rmse_train = np.inf
    best_rmse_test   = np.inf
    if hyperparamTuning: combinations= [(s, lmbda) for s in s_list for lmbda in lmbda_list]
    else: combinations = [(1.20e-01, 1.10e-01)] # Best hyperparameters that we have tuned previously

    for s, lmbda in tqdm(combinations, desc="Gaussian Kernel Tuning..." if hyperparamTuning else \
                         "Running Gaussian KRR...", ascii=False, ncols=70):
        num_total_samples = len(X_train)
        row_offset_per_process = num_total_samples//size
        num_of_rows_in_last_process = row_offset_per_process + (num_total_samples%size)
        number_of_rows_in_process = num_of_rows_in_last_process if rank == size - 1 else row_offset_per_process

        local_A = np.zeros((number_of_rows_in_process, num_total_samples))
        local_y = np.zeros(number_of_rows_in_process)
        local_alpha = np.zeros(number_of_rows_in_process)
        local_row_offset = rank*row_offset_per_process
        global_p = np.zeros(num_total_samples)


        # Local kernel computation
        for i in range(len(local_A)):
            row_num = rank*row_offset_per_process + i
            local_y[i] = y_train[row_num]
            local_A[i] = (gaussian_kernel_vectorised(X_train[row_num], X_train, s))
            local_A[i, row_num] += lmbda

        local_alpha = local_conjugate_gradient(local_A, local_y, local_alpha, local_row_offset, global_p, comm, 1e-6)

        # Prediction and Test
        local_y_train_pred = np.zeros(len(y_train))
        if rank == 0:
            global_y_train_pred = np.zeros(len(y_train))
        else:
            global_y_train_pred = None
        
        for i in range(len(y_train)):
            start = rank*row_offset_per_process
            end = start + number_of_rows_in_process
            local_y_train_pred[i] = np.dot(local_alpha, gaussian_kernel_vectorised(X_train[i], X_train[start:end], s))

        comm.Reduce([local_y_train_pred, MPI.DOUBLE], [global_y_train_pred, MPI.DOUBLE], op=MPI.SUM, root=0)


        local_y_test_pred = np.zeros(len(y_test))
        if rank == 0:
            global_y_test_pred = np.zeros(len(y_test))
        else:
            global_y_test_pred = None

        for i in range(len(y_test)):
            start = rank*row_offset_per_process
            end = start + number_of_rows_in_process
            local_y_test_pred[i] = np.dot(local_alpha, gaussian_kernel_vectorised(X_test[i], X_train[start:end], s))

        comm.Reduce([local_y_test_pred, MPI.DOUBLE], [global_y_test_pred, MPI.DOUBLE], op=MPI.SUM, root=0)

        if rank == 0:
            rmse_train = root_mean_squared_error(y_train, global_y_train_pred)
            rmse_test = root_mean_squared_error(y_test, global_y_test_pred)
            if rmse_test < best_rmse_test:
                best_s = s
                best_lmbda = lmbda
                best_rmse_train = rmse_train
                best_rmse_test  = rmse_test
    
    if hyperparamTuning and rank == 0: print(f"\nGaussian Kernel ==> Best s {best_s:.2e}, Best lambda {best_lmbda:.2e}, Best RMSE_train {best_rmse_train:.4f}, Best RMSE_test {best_rmse_test:.4f}\n")
    elif rank == 0: print(f"\nGaussian Kernel ran with s = {best_s:.2e} and lambda = {best_lmbda:.2e}. RMSE_train = {best_rmse_train:.4f} and RMSE_test = {best_rmse_test:.4f}\n")
        
main()
 