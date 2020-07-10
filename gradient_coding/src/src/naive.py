from __future__ import print_function
import sys
import random
from util import *
import os
import numpy as np
import scipy.sparse as sps
import time
from mpi4py import MPI

def naive_logistic_regression(n_procs, n_samples, n_features, input_dir, n_stragglers, straggle_time, is_real_data, params):
	# Setup MPI data
	comm = MPI.COMM_WORLD										# Create communicator to send/receive data through MPI.
	rank = comm.Get_rank()										# Worker identifying number. Rank = 0 represents the master.
	size = comm.Get_size()										# Get total number of processes (equal to n_procs).
	
	rounds = params[0]											# Set the number of iterations determined by the simulation parameters.

	beta=np.zeros(n_features)									# Initialize beta. 1xCOLS matrix of zeros (COLS is number of columns in dataset).

	# Loading the data for the workers. Only do this 
	# for workers (not for master whose rank = 0)
	if (rank):
		# Check if using a real or artificial dataset.
		if not is_real_data:									# Artifical dataset.
			X_current = load_data(input_dir+str(rank)+".dat")	# Load the assigned data from the dataset.

			y = load_data(input_dir+"label.dat")				# Load weights (y-values; either -1 or 1).
		else:													# Using real dataset.			
			X_current = load_sparse_csr(input_dir+str(rank))	# Load the assigned data from the dataset.

			y = load_data(input_dir+"label.dat")				# Load weights (y-values; either -1 or 1).


		rows_per_worker = X_current.shape[0]					# Calculate the number of rows per worker as the number of rows assigned to the current worker.
		

		# Load the appropriate weights. The dataset isn't
		# partitioned locally, so it is done here. 
		y_current = y[ ( rank - 1 ) * rows_per_worker : rank * rows_per_worker ]

	# Initializing relevant variables
	if (rank):
		try:
			predy = X_current.dot(beta)
			g = -X_current.T.dot(np.divide(y_current,np.exp(np.multiply(predy,y_current))+1))
			send_req = MPI.Request()
			recv_reqs = []
		except:
			print( "X_Current: {}\nBeta: {}\n".format( X_current.shape, beta.shape ) )
			exit( 0 )

	else:

		msgBuffers = [np.zeros(n_features) for i in range(n_procs-1)]
		g=np.zeros(n_features)
		betaset = np.zeros((rounds, n_features))
		timeset = np.zeros(rounds)
		worker_timeset=np.zeros((rounds, n_procs-1))
		
		request_set = []
		recv_reqs = []

		cnt_completed = 0

		status = MPI.Status()

		eta0=params[2] # ----- learning rate schedule
		alpha = params[1] # --- coefficient of l2 regularization
		utemp = np.zeros(n_features) # for accelerated gradient descent
   
	# Posting all Irecv requests for master and workers
	if (rank):

		for i in range(rounds):
			req = comm.Irecv([beta, MPI.DOUBLE], source=0, tag=i)
			recv_reqs.append(req)

	else:

		for i in range(rounds):
			recv_reqs = []
			for j in range(1,n_procs):
				req = comm.Irecv([msgBuffers[j-1], MPI.DOUBLE], source=j, tag=i)
				recv_reqs.append(req)
			request_set.append(recv_reqs)

	########################################################################################################
	comm.Barrier()

	if rank==0:
		orig_start_time= time.time()
		print("---- Starting Naive Iterations ----")

	for i in range(rounds):
		if rank==0:

			if( i % 10 == 0):
				print("\t >>> At Iteration %d" %(i))

			start_time = time.time()

			### Region 1
			for l in range(1,n_procs):
				comm.Isend([beta,MPI.DOUBLE],dest=l,tag=i)
			###

			### Region 2
			g[:]=0
			cnt_completed = 0

			# Stay here until ( n_workers - n_stragglers ) workers are done computing
			while cnt_completed < n_procs-1:
				# Wait until any worker is done and has sent its data
				req_done = MPI.Request.Waitany(request_set[i], status)

				# Get the worker number
				src = status.Get_source()

				# Calculate the amount of time the worker took.
				worker_timeset[i,src-1]=time.time()-start_time

				# Remove the message request from the list.
				request_set[i].pop(req_done)
				
				# Add the partial gradients.
				g+=msgBuffers[src-1]

				# Increment the count of completed workers.
				cnt_completed+=1
			###

			### Region 3
			grad_multiplier = eta0[i]/n_samples 							# 1
			# ---- update step for gradient descent
			# np.subtract((1-2*alpha*eta0[i])*beta , grad_multiplier*g, out=beta)

			# ---- updates for accelerated gradient descent
			theta = 2.0/(i+2.0)												# 2
			ytemp = (1-theta)*beta + theta*utemp							# 3
			betatemp = ytemp - grad_multiplier*g - (2*alpha*eta0[i])*beta 	# 4
			utemp = beta + (betatemp-beta)*(1/theta) 						# 5
			beta[:] = betatemp 												# 6
			
			timeset[i] = time.time() - start_time
			betaset[i,:] = beta
			###

		else:
			# Wait for data to be received
			recv_reqs[i].Wait()


			# Create a straggler by waiting a set amount of time before continuing
			if( 1 <= rank <= n_stragglers ):
				time.sleep( straggle_time )


			# sendTestBuf = send_req.test()
			# if not sendTestBuf[0]:
			#     send_req.Cancel()


			# Compute partial gradient
			predy = X_current.dot(beta)
			g = X_current.T.dot(np.divide(y_current,np.exp(np.multiply(predy,y_current))+1))
			g *= -1


			# Send data back to master
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

			training_loss[i] = calculate_loss(y_train, predy_train, n_train)	# Calculate loss on training data
			testing_loss[i] = calculate_loss(y_test, predy_test, n_test)		# Calculate loss on testing data

			# Generate the ROC curve. 
			# FPR (False Positive Rate) is the x-axis, TPR (Total Positive Rate) is the y-axis
			fpr, tpr, thresholds = roc_curve(y_test,predy_test, pos_label=1)	

			auc_loss[i] = auc(fpr,tpr)

			print("Iteration %d: Train Loss = %5.3f, Test Loss = %5.3f, AUC = %5.3f, Total time taken =%5.3f"%(i, training_loss[i], testing_loss[i], auc_loss[i], timeset[i]))
		

		# Get the output location. Create the directory if it does not exist.
		output_dir = input_dir + "results/"
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)


		# Save simulation data.
		save_vector(training_loss, output_dir+"naive_acc_training_loss.dat")
		save_vector(testing_loss, output_dir+"naive_acc_testing_loss.dat")
		save_vector(auc_loss, output_dir+"naive_acc_auc.dat")
		save_vector(timeset, output_dir+"naive_acc_timeset.dat")
		save_matrix(worker_timeset, output_dir+"naive_acc_worker_timeset.dat")
		print(">>> Done")

	# Synchronized ending.
	comm.Barrier()