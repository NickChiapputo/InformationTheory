from __future__ import print_function
import time
import sys
sys.path.append('./src/')
from naive import *
from coded import *
from replication import *
from avoidstragg import *
from partial_replication import *
from partial_coded import *
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if len(sys.argv) != 13:
    print("Usage: python main.py n_procs n_rows n_cols input_dir is_real dataset is_coded n_stragglers partial_straggler_partitions coded_ver num_itrs")
    sys.exit(0)

# Input variable descriptions:
# n_procs           Number of processes, workers + master.
# n_rows            Number of rows in the dataset.
# n_cols            Number of columns in the dataset.
# input_dir         Directory where the data is stored.
# is_real           Flag denoting whether the data is real or not.
# dataset           Name of the dataset (only used if the dataset is real).
# is_coded          Set if the scheme is coded or not. If false, only using naive scheme.
# n_stragglers      Number of stragglers in the simulation.
# straggler_delay   Delay for the stragglers during the simulation.
# n_iterations      Number of times the simulation will be run.
# partitions        Number of partitions. Only used in partial schemes. Will be 0 for naive, cyclic/fractional repetition, and ignore/avoid stragglers.
# coded_ver         Value used to choose specific schemes in the logic statements in combination with 'is_coded' and 'partitions'.

n_procs, n_rows, n_cols, input_dir, is_real, dataset, is_coded, n_stragglers, straggler_delay, n_iterations, partitions, coded_ver  = [x for x in sys.argv[1:]]
n_procs, n_rows, n_cols, is_real, is_coded, n_stragglers , partitions, coded_ver = int(n_procs), int(n_rows), int(n_cols), int(is_real), int(is_coded), int(n_stragglers), int(partitions), int(coded_ver)
input_dir = input_dir+"/" if not input_dir[-1] == "/" else input_dir


# Convert parameters to numbers where necessary
n_iterations = int( n_iterations )
straggler_delay = float( straggler_delay )

# ---- Modifiable parameters
alpha = 1.0/n_rows #sometimes we used 0.0001 # --- coefficient of l2 regularization

learning_rate_schedule = 10.0*np.ones(n_iterations)
# eta0=10.0
# t0 = 90.0
# learning_rate_schedule = [eta0*t0/(i + t0) for i in range(1,n_iterations+1)]

# -------------------------------------------------------------------------------------------------------------------------------

params = []
params.append(n_iterations)
params.append(alpha)
params.append(learning_rate_schedule)

if not size == n_procs:
    print("Number of processers doesn't match!")
    sys.exit(0)

if not is_real:
    dataset = "artificial-data/" + str(n_rows) + "x" + str(n_cols)

if is_coded:

    if partitions:
        if(coded_ver == 1):
            partial_replication_logistic_regression(n_procs, n_rows, n_cols, input_dir + dataset +"/partial/" + str((partitions-n_stragglers)*(n_procs-1)) + "/", n_stragglers, partitions, is_real, params)
        elif(coded_ver == 0):
            partial_coded_logistic_regression(n_procs, n_rows, n_cols, input_dir + dataset +"/partial/" + str((partitions-n_stragglers)*(n_procs-1)) + "/", n_stragglers, partitions, is_real, params)
            
    else:
        if(coded_ver == 0):
            coded_logistic_regression(n_procs, n_rows, n_cols, input_dir + dataset +"/" + str(n_procs-1) + "/", n_stragglers, straggler_delay, is_real, params)
            
        elif(coded_ver == 1):
            replication_logistic_regression(n_procs, n_rows, n_cols, input_dir + dataset +"/" + str(n_procs-1) + "/", n_stragglers, straggler_delay, is_real, params)

        elif(coded_ver ==2):
            avoidstragg_logistic_regression(n_procs, n_rows, n_cols, input_dir + dataset +"/" + str(n_procs-1) + "/", n_stragglers, straggler_delay, is_real, params)
else:
    naive_logistic_regression(n_procs, n_rows, n_cols, input_dir + dataset +"/" + str(n_procs-1) + "/", n_stragglers, straggler_delay, is_real, params)

comm.Barrier()
MPI.Finalize()
