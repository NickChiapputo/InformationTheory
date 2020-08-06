from __future__ import print_function
import sys
import random
from util import *
import os
import numpy as np
import scipy.sparse as sps
import time
from mpi4py import MPI

def avoidstragg_logistic_regression(n_procs, n_samples, n_features, input_dir, n_stragglers, straggle_time, is_real_data, params):

	# Setup MPI data
	comm = MPI.COMM_WORLD                           # Create communicator to send/receive data through MPI.
	rank = comm.Get_rank()                          # Worker identifying number. Rank = 0 represents the master.
	size = comm.Get_size()                          # Get total number of processes (equal to n_procs).
	
	rounds = params[0]                              # Set the number of iterations determined by the simulation parameters.

	isStraggler = np.zeros( 1 )                     # Flag to determine if worker is a straggler. Set by the master and sent with beta at each iteration.

	beta=np.zeros(n_features)

	# Loading data on workers
	if (rank):

		if not is_real_data:
			X_current = load_data(input_dir+str(rank)+".dat")
			y = load_data(input_dir+"label.dat")
		else:
			X_current = load_sparse_csr(input_dir+str(rank))
			y = load_data(input_dir+"label.dat")

		rows_per_worker = X_current.shape[0]
		y_current=y[(rank-1)*rows_per_worker:rank*rows_per_worker]
	
	# Initializing relevant variables
	if (rank):

		predy = X_current.dot(beta)
		g = -X_current.T.dot(np.divide(y_current,np.exp(np.multiply(predy,y_current))+1))
		send_req = MPI.Request()
		recv_reqs = []
		straggler_reqs = []

	else:

		msgBuffers = [np.zeros(n_features) for i in range(n_procs-1)]
		g=np.zeros(n_features)
		betaset = np.zeros((rounds, n_features))
		timeset = np.zeros(rounds)
		worker_timeset=np.zeros((rounds, n_procs-1))

		region1_timeset = np.zeros( rounds )
		region2_timeset = np.zeros( rounds )
		region3_timeset = np.zeros( rounds )
		region4_timeset = np.zeros( rounds )
		region5_timeset = np.zeros( rounds )
		
		request_set = []
		recv_reqs = []
		send_set = []


		cnt_completed = 0
		completed_workers = np.ndarray(n_procs-1,dtype=bool)

		status = MPI.Status()

		alpha = params[1] # --- coefficient of l2 regularization
		eta_sequence = params[2] # --- learning rate schedule
		utemp = np.zeros(n_features) # for accelerated gradient descent

	# Posting all Irecv requests for master and workers
	if (rank):

		for i in range(rounds):
			req = comm.Irecv([beta, MPI.DOUBLE], source=0, tag=i)
			recv_reqs.append(req)

			stragglerReq = comm.Irecv( [ isStraggler, MPI.INT ], source = 0, tag = i )
			straggler_reqs.append( stragglerReq )

	else:

		for i in range(rounds):
			recv_reqs = []
			for j in range(1,n_procs):
				req = comm.Irecv([msgBuffers[j-1], MPI.DOUBLE], source=j, tag=i)
				recv_reqs.append(req)
			request_set.append(recv_reqs)

	##########################################################################################
	comm.Barrier()

	if rank==0:
		orig_start_time = time.time()
		print("---- Starting AvoidStragg Iterations with " +str(n_stragglers) + " stragglers ----")

	for i in range(rounds):
  
		if rank==0:
			### Region 1
			regionTime = time.time()

			if(i%10 == 0):
				print("\t >>> At Iteration %d" %(i))

			start_time = time.time()

			region1_timeset[ i ] = start_time - regionTime
			###

			### Region 2
			regionTime = time.time()

			g[:]=0.0
			cnt_completed = 0
			completed_workers[:]=False

			send_set[:] = []

			# Randomly select worker to be straggler. Range [ 1, # of workers ]
			straggleRank = np.zeros( n_stragglers )
			for l in range( 0, n_stragglers ):
				straggleRank[ l ] = random.SystemRandom().randint( 1, n_procs - 1 )
				# print( "Selected rank {} to straggle.".format( straggleRank ) )
			for l in range(1,n_procs):
				isStraggler[ 0 ] = 0

				if l in straggleRank:
					isStraggler[ 0 ] = 1

				comm.Isend( [ beta, MPI.DOUBLE ], dest = l, tag = i )
				comm.Isend( [ isStraggler, MPI.INT ], dest = l, tag = i )

			region2_timeset[ i ] = time.time() - regionTime
			###

			### Region 3
			regionTime = time.time()
			
			while cnt_completed < n_procs-1-n_stragglers:
				req_done = MPI.Request.Waitany(request_set[i], status)
				src = status.Get_source()
				worker_timeset[i,src-1]=time.time()-start_time
				request_set[i].pop(req_done)

				g += msgBuffers[src-1]   # add the partial gradients
				cnt_completed += 1
				completed_workers[src-1] = True

			region3_timeset[ i ] = time.time() - regionTime
			###

			### Region 4
			regionTime = time.time()

			grad_multiplier = eta_sequence[i]/(n_samples*(n_procs-1-n_stragglers)/(n_procs-1))
			# ---- update step for gradient descent
			# np.subtract((1-2*alpha*eta_sequence[i])*beta , grad_multiplier*g, out=beta)

			# ---- updates for accelerated gradient descent
			theta = 2.0/(i+2.0)
			ytemp = (1-theta)*beta + theta*utemp
			betatemp = ytemp - grad_multiplier*g - (2*alpha*eta_sequence[i])*beta
			utemp = beta + (betatemp-beta)*(1/theta)
			beta[:] = betatemp

			timeset[i] = time.time() - start_time

			betaset[i,:] = beta
			ind_set = [l for l in range(1,n_procs) if not completed_workers[l-1]]
			for l in ind_set:
				worker_timeset[i,l-1]=-1

			region4_timeset[ i ] = time.time() - regionTime
			###

			### Region 5
			regionTime = time.time()

			# MPI.Request.Cancel
			MPI.Request.Waitall( request_set[ i ] )

			region5_timeset[ i ] = time.time() - regionTime
			###

		else:
			recv_reqs[i].Wait()
			straggler_reqs[ i ].Wait()


			# Create a straggler by waiting a set amount of time before continuing
			if( isStraggler ):
				# print( "Rank {}: straggling.".format( rank ) )
				time.sleep( straggle_time )
			# else:
			#   print( "Rank {}: not straggling.".format( rank ) )

			sendTestBuf = send_req.test()
			if not sendTestBuf[0]:
				send_req.Cancel()
				print("Worker " + str(rank) + " cancelled send request for Iteration " + str(i))


			predy = X_current.dot(beta)
			g = X_current.T.dot(np.divide(y_current,np.exp(np.multiply(predy,y_current))+1))
			g *= -1
			send_req = comm.Isend([g, MPI.DOUBLE], dest=0, tag=i)

	#########################################################################################
	comm.Barrier()

	if rank==0:
		elapsed_time= time.time() - orig_start_time
		print ("Total Time Elapsed: %.3f" %(elapsed_time))

		totalLoadTime = time.time()
		loadTime = 0.0
		vstackTime = 0.0

		# Load all training data        
		if not is_real_data:
			loadTimeStart = time.time()

			X_train = load_data( input_dir + "1.dat" )

			loadTime += time.time() - loadTimeStart

			print( ">> Loaded 1" )
			for j in range( 2, n_procs - 1 ):
				loadTimeStart = time.time()

				X_temp = load_data( input_dir + str( j ) + ".dat" )

				loadTime += time.time() - loadTimeStart


				vstackTimeStart = time.time()

				X_train = np.vstack( ( X_train, X_temp ) )

				vstackTime += time.time() - vstackTimeStart

				print( ">> Loaded " + str( j ) )
		else:
			loadTimeStart = time.time()

			X_train = load_sparse_csr( input_dir + "1" )

			loadTime += time.time() - loadTimeStart

			print( ">> Loaded 1" )
			for j in range( 2, n_procs - 1 ):
				loadTimeStart = time.time()

				X_temp = load_sparse_csr( input_dir + str( j ) )

				loadTime += time.time() - loadTimeStart


				vstackTimeStart = time.time()

				X_train = sps.vstack( ( X_train, X_temp ) )

				vstackTime += time.time() - vstackTimeStart

				print( ">> Loaded " + str( j ) )

		loadTimeStart = time.time()

		y_train = load_data( input_dir + "label.dat" )

		loadTime += time.time() - loadTimeStart

		y_train = y_train[ 0 : X_train.shape[ 0 ] ]

		# Load all testing data
		testingLoadTimeStart = time.time()
		
		y_test = load_data( input_dir + "label_test.dat" )
		if not is_real_data:
			X_test = load_data( input_dir + "test_data.dat" )
		else:
			X_test = load_sparse_csr( input_dir + "test_data" )
		
		testingLoadTime = time.time() - testingLoadTimeStart
		totalLoadTime = time.time() - totalLoadTime
		lossTime = time.time()

		n_train = X_train.shape[ 0 ]
		n_test = X_test.shape[ 0 ]

		training_loss = np.zeros( rounds )
		testing_loss = np.zeros( rounds )
		auc_loss = np.zeros( rounds )

		from sklearn.metrics import roc_curve, auc

		for i in range( rounds ):
			beta = np.squeeze( betaset[ i , : ] )
			predy_train = X_train.dot( beta )
			predy_test = X_test.dot( beta )
			training_loss[ i ] = calculate_loss( y_train, predy_train, n_train )
			testing_loss[ i ] = calculate_loss( y_test, predy_test, n_test )
			fpr, tpr, thresholds = roc_curve( y_test,predy_test, pos_label = 1 )

			auc_loss[ i ] = auc( fpr,tpr )
			print( "Iteration %d: Train Loss = %5.3f, Test Loss = %5.3f, AUC = %5.3f, Total time taken =%5.3f" % ( i, training_loss[ i ], testing_loss[ i ], auc_loss[ i ], timeset[ i ] ) )
		lossTime = time.time() - lossTime

		saveTime = time.time()
		
		output_dir = input_dir + "results/"
		if not os.path.exists( output_dir ):
			os.makedirs( output_dir )

		save_vector( training_loss, 	output_dir + "avoidstragg_acc_%d_training_loss.dat"		% ( n_stragglers ) )
		save_vector( testing_loss, 		output_dir + "avoidstragg_acc_%d_testing_loss.dat"		% ( n_stragglers ) )
		save_vector( auc_loss, 			output_dir + "avoidstragg_acc_%d_auc.dat"				% ( n_stragglers ) )
		save_vector( timeset, 			output_dir + "avoidstragg_acc_%d_timeset.dat"			% ( n_stragglers ) )

		save_vector( region1_timeset, 	output_dir + "avoidstragg_region1_timeset.dat" )
		save_vector( region2_timeset, 	output_dir + "avoidstragg_region2_timeset.dat" )
		save_vector( region3_timeset, 	output_dir + "avoidstragg_region3_timeset.dat" )
		save_vector( region4_timeset, 	output_dir + "avoidstragg_region4_timeset.dat" )
		save_vector( region5_timeset, 	output_dir + "avoidstragg_region5_timeset.dat" )

		save_matrix( worker_timeset, 	output_dir + "avoidstragg_acc_%d_worker_timeset.dat"	% ( n_stragglers ) )
		
		print( ">>> Done" )

		saveTime = time.time() - saveTime

		print( "Total Load Time = %5.3f\nTesting Data Load Time: %5.3f\nTraining Data Loading Time: %5.3f (%5.3f)\nVstack Time: %5.3f (%5.3f)\nLoss Time: %5.3f\nSave Data: %5.3f\n" % ( totalLoadTime, testingLoadTime, loadTime, loadTime / ( n_procs - 1 ), vstackTime, vstackTime / ( n_procs - 2 ), lossTime, saveTime ) )
	
	comm.Barrier()