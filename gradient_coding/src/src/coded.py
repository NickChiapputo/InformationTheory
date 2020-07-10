from __future__ import print_function
import sys
import random
from util import *
import os
import numpy as np
import time
from mpi4py import MPI
import scipy.special as sp
import scipy.sparse as sps

#							 Processes Rows       Columns     Directory  Stragglers    Delay          Real Data     Parametesr
def coded_logistic_regression(n_procs, n_samples, n_features, input_dir, n_stragglers, straggle_time, is_real_data, params):
	# Setup MPI data
	comm = MPI.COMM_WORLD							# Create communicator to send/receive data through MPI.
	rank = comm.Get_rank()							# Worker identifying number. Rank = 0 represents the master.
	size = comm.Get_size()							# Get total number of processes (equal to n_procs).
	
	# Set the number of iterations determined by the simulation parameters.
	rounds = params[0]

	# Set up worker and row counts.
	n_workers = n_procs - 1							# Number of workers is one less than number of processes.
	rows_per_worker = n_samples / ( n_procs - 1 )	# Evenly divide up rows for each worker.

	# Loading the data for the workers. Only do this for workers (not for master whose rank = 0)
	if( rank ):
		# Check if using a real dataset or using artificial dataset.
		if not is_real_data:
			# Using artifical dataset.

			# Load weights (y-values; either -1 or 1).
			y = load_data(input_dir+"label.dat")


			# Created matrices with zeros.
			# Rows (X and Y) = (s + 1) * rows_per_worker
			# Cols (Only X)  = dimension of data (n_cols from input, also given as 'p' in paper)

			# Assigned partitions from B_cyc
			X_current = np.zeros( [ ( 1 + n_stragglers ) * rows_per_worker, n_features ] )
			
			# 
			y_current = np.zeros( ( 1 + n_stragglers ) * rows_per_worker )

			for i in range( 1 + n_stragglers ):
				try:

					# print( "Rank {}: Load partition {}".format( rank, input_dir + str( ( rank - 1 + i ) % n_workers + 1 ) ) )

					# Load data set in a cyclic manner
					X_current[ i * rows_per_worker : ( i + 1 ) * rows_per_worker, : ] = load_data( input_dir + str( ( rank - 1 + i ) % n_workers + 1 ) + ".dat" )

					# Load labels with respect to data loaded previously
					y_current[ i * rows_per_worker : ( i + 1 ) * rows_per_worker ] = y[ ( ( rank - 1 + i ) % n_workers ) * rows_per_worker : ( ( rank - 1 + i ) % n_workers + 1 ) * rows_per_worker ]
				except ValueError:
					print( 	"Rank {0}, i = {1}:\n"
							"    Value Error: Data Shape: {2}; Destination Shape: {3}\n"
							"    Data: {4}\n".format( rank, i, x.shape, X_current[ i * rows_per_worker : ( i + 1 ) * rows_per_worker, : ].shape, input_dir + str( ( rank - 1 + i ) % n_workers + 1 ) + ".dat" ) )
					exit( 0 )

			# print( "Rank {}: Done loading data.".format( rank ) )
		else:
			# Using real dataset.

			y_current=np.zeros((1+n_stragglers)*rows_per_worker)
			y = load_data(input_dir+"label.dat")
			for i in range(1+n_stragglers):

				if i==0:
					X_current=load_sparse_csr(input_dir+str((rank-1+i)%n_workers+1))
				else:
					X_temp = load_sparse_csr(input_dir+str((rank-1+i)%n_workers+1))
					X_current = sps.vstack((X_current,X_temp))

				y_current[i*rows_per_worker:(i+1)*rows_per_worker]=y[((rank-1+i)%n_workers)*rows_per_worker:((rank-1+i)%n_workers+1)*rows_per_worker]

	# Initializing relevant variables
	B = np.zeros((n_workers,n_workers))
	beta=np.zeros(n_features)
	
	if( rank ):
		# Calculate dot product of assigned partitions with beta
		# X_current shape = ( ( s + 1 ) * rows_per_worker ) x n_features
		# beta shape = 1 x n_features
		# predy shape = ( ( s + 1 ) * rows_per_worker ) x 1

		# predy = X_current.dot( beta )
		predy = np.zeros( ( 1 + n_stragglers ) * rows_per_worker )


		# Calculate initial gradient
		# X^T * ( y_current / exp( ( predy * y_current ) + 1 ) )
		# Shape = N_COLS x 1
		g = -X_current.T.dot( np.divide( y_current, np.exp( np.multiply( predy, y_current ) ) + 1 ) )


		# print( "Rank {:02d}: beta: {}, X_current: {}, predy: {}, {} g: {}, {}\n\n".format( rank, beta.shape, X_current.shape, predy.shape, predy, g.shape, g ) )
		# exit( 0 )


		send_req = MPI.Request()
		recv_reqs = []

		# print( "Rank {}: Done calculating predy and initial gradient.".format( rank ) )

	else:
		B=getB(n_workers,n_stragglers)
		# A = np.zeros((int(sp.binom(n_workers,n_stragglers)),n_workers))
		# A=getA(B,n_workers,n_stragglers)

		msgBuffers = np.array([np.zeros(n_features) for i in range(n_procs-1)])
	
		g=np.zeros(n_features)

		A_row = np.zeros((1,n_procs-1))

		betaset = np.zeros((rounds, n_features))
		timeset = np.zeros(rounds)
		worker_timeset=np.zeros((rounds, n_procs-1))
		
		request_set = []
		recv_reqs = []
		send_set = []

		cnt_completed = 0
		completed_workers = np.ndarray(n_procs-1,dtype=bool)
		status = MPI.Status()

		eta0=params[2] # ----- learning rate
		alpha = params[1] # --- coefficient of l2 regularization
		utemp = np.zeros(n_features) # for accelerated gradient descent
	
	B = comm.bcast(B, root=0)

	# Setting up y_current_mod on all workers
	if( rank ):
		y_current_mod = np.zeros( ( 1 + n_stragglers ) * rows_per_worker )
		for i in range( 1 + n_stragglers ):
			y_current_mod[ i * rows_per_worker : ( i + 1 ) * rows_per_worker ] = B[ rank - 1, ( ( rank - 1 + i ) % n_workers ) ] * y_current[ i * rows_per_worker : ( i + 1 ) * rows_per_worker ]

		# print( "Rank {}: Done setting up y_current_mod.".format( rank ) )

	# Posting all Irecv requests for master and workers
	if( rank ):

		for i in range(rounds):
			# Request data 
			req = comm.Irecv([beta, MPI.DOUBLE], source=0, tag=i)
			recv_reqs.append(req)

		# print( "Rank {}: Done creating Irecv requests.".format( rank ) )
	else:

		for i in range(rounds):
			recv_reqs = []
			for j in range(1,n_procs):
				req = comm.Irecv([msgBuffers[j-1], MPI.DOUBLE], source=j, tag=i)
				recv_reqs.append(req)
			request_set.append(recv_reqs)  

	#######################################################################################################
	comm.Barrier()
	if rank==0:
		print("---- Starting Coded Iterations for " +str(n_stragglers) + " stragglers ----")
		orig_start_time= time.time()
	for i in range(rounds):
	
		if rank == 0:

			if(i%10 == 0):
				print("\t >>> At Iteration %d" %(i))

			A_row[:] = 0
			send_set[:] = []
			completed_workers[:]=False
			cnt_completed = 0

			start_time = time.time()
			
			for l in range(1,n_procs):
				sreq = comm.Isend([beta, MPI.DOUBLE], dest = l, tag = i)
				send_set.append(sreq)            
			
			# Stay here until ( n_workers - n_stragglers ) workers are done computing
			while cnt_completed < n_procs-1-n_stragglers:
				# Wait until any worker is done and has sent its data
				req_done = MPI.Request.Waitany(request_set[i], status)

				# Get the worker number
				src = status.Get_source()

				# Calculate the amount of time the worker took
				worker_timeset[i,src-1]=time.time()-start_time

				# 
				request_set[i].pop(req_done)

				# Increment the count of completed workers
				cnt_completed += 1

				# Set the worker's completed flag to True
				completed_workers[src-1] = True

			
			completed_ind_set = [l for l in range(n_procs-1) if completed_workers[l]]
			A_row[0,completed_ind_set] = np.linalg.lstsq(B[completed_ind_set,:].T,np.ones(n_workers))[0]
			g = np.squeeze(np.dot(A_row, msgBuffers))
			
			# case_idx = calculate_indexA(completed_stragglers)
			# g = np.dot(A[case_idx,ind_set],tmpBuff)
			
			grad_multiplier = eta0[i]/n_samples
			# ---- update step for gradient descent
			# np.subtract((1-2*alpha*eta0[i])*beta , grad_multiplier*g, out=beta)

			# ---- updates for accelerated gradient descent
			theta = 2.0/(i+2.0)
			ytemp = (1-theta)*beta + theta*utemp
			betatemp = ytemp - grad_multiplier*g - (2*alpha*eta0[i])*beta
			utemp = beta + (betatemp-beta)*(1/theta)
			beta[:] = betatemp

			timeset[i] = time.time() - start_time
			betaset[i,:] = beta

			ind_set = [l for l in range(n_procs-1) if not completed_workers[l]]
			for l in ind_set:
				worker_timeset[i,l]=-1

		else:
			# Wait for data to be received
			recv_reqs[i].Wait()


			# Create a straggler by waiting a set amount of time before continuing
			if( 1 <= rank <= n_stragglers ):
				time.sleep( straggle_time )

			sendTestBuf = send_req.test()
			if not sendTestBuf[0]:
				send_req.Cancel()
				#print("Worker " + str(rank) + " cancelled send request for Iteration " + str(i))

			# if rank == 1:
			# 	print( "Rank {}: beta: {}".format( rank, beta ) )

			predy = X_current.dot(beta)
			g = X_current.T.dot(np.divide(y_current_mod,np.exp(np.multiply(predy,y_current))+1))
			g *= -1
			send_req = comm.Isend([g, MPI.DOUBLE], dest=0, tag=i)

	#####################################################################################################
	comm.Barrier()
	if rank==0:
		elapsed_time= time.time() - orig_start_time
		print ("Total Time Elapsed: %.3f" %(elapsed_time))



		# Load all training data
		if not is_real_data:
			X_train = load_data(input_dir+"1.dat")
			print(">> Loaded 1")
			for j in range(2,n_procs-1):
				X_temp = load_data(input_dir+str(j)+".dat")
				X_train = np.vstack((X_train, X_temp))
				print(">> Loaded "+str(j))
		else:
			X_train = load_sparse_csr(input_dir+"1")
			for j in range(2,n_procs-1):
				X_temp = load_sparse_csr(input_dir+str(j))
				X_train = sps.vstack((X_train, X_temp))

		y_train = load_data(input_dir+"label.dat")
		y_train = y_train[0:X_train.shape[0]]

		# Load all testing data
		y_test = load_data(input_dir + "label_test.dat")
		if not is_real_data:
			X_test = load_data(input_dir+"test_data.dat")
		else:
			X_test = load_sparse_csr(input_dir+"test_data")

		n_train = X_train.shape[0]
		n_test = X_test.shape[0]

		training_loss = np.zeros(rounds)
		testing_loss = np.zeros(rounds)
		auc_loss = np.zeros(rounds)

		from sklearn.metrics import roc_curve, auc

		for i in range(rounds):
			beta = np.squeeze(betaset[i,:])
			predy_train = X_train.dot(beta)
			predy_test = X_test.dot(beta)
			training_loss[i] = calculate_loss(y_train, predy_train, n_train)
			testing_loss[i] = calculate_loss(y_test, predy_test, n_test)
			fpr, tpr, thresholds = roc_curve(y_test,predy_test, pos_label=1)
			print( "FPR: {}\n\nTPR: {}".format( fpr, tpr ) )
			auc_loss[i] = auc(fpr,tpr)
			print("Iteration %d: Train Loss = %5.3f, Test Loss = %5.3f, AUC = %5.3f, Total time taken =%5.3f"%(i, training_loss[i], testing_loss[i], auc_loss[i], timeset[i]))
		
		output_dir = input_dir + "results/"
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)

		save_vector(training_loss, output_dir+"coded_acc_%d_training_loss.dat"%(n_stragglers))
		save_vector(testing_loss, output_dir+"coded_acc_%d_testing_loss.dat"%(n_stragglers))
		save_vector(auc_loss, output_dir+"coded_acc_%d_auc.dat"%(n_stragglers))
		save_vector(timeset, output_dir+"coded_acc_%d_timeset.dat"%(n_stragglers))
		save_matrix(worker_timeset, output_dir+"coded_acc_%d_worker_timeset.dat"%(n_stragglers))
		print(">>> Done")

	comm.Barrier()