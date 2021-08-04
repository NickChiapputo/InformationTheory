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
	initTime = time.time()

	# Setup MPI data
	comm = MPI.COMM_WORLD										# Create communicator to send/receive data through MPI.
	rank = comm.Get_rank()										# Worker identifying number. Rank = 0 represents the master.
	size = comm.Get_size()										# Get total number of processes (equal to n_procs which is used instead).
	
	rounds = params[0]											# Set the number of iterations determined by the simulation parameters.

	beta=np.zeros(n_features)									# Initialize beta. 1xCOLS matrix of zeros (COLS is number of columns in dataset).

	isStraggler = np.zeros( 1 )									# Flag to determine if worker is a straggler. Set by the master and sent with beta at each iteration.


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
			predy = X_current.dot(beta)							# Results in a zero matrix of size 1xCOLS
			g = -X_current.T.dot(np.divide(y_current,np.exp(np.multiply(predy,y_current))+1))
			send_req = MPI.Request()
			recv_reqs = []
			straggler_reqs = []
		except:
			print( "X_Current: {}\nBeta: {}\n".format( X_current.shape, beta.shape ) )
			exit( 0 )

		waitingTimeset = np.zeros( rounds )
		sleepingTimeset = np.zeros( rounds )
		computingTimeset = np.zeros( rounds )
		sendingDataTimeset = np.zeros( rounds )
		totalTimeset = np.zeros( rounds )

	else:

		msgBuffers = [ np.zeros( n_features ) for i in range( n_procs - 1 ) ]	# Initialize message buffers to send data to workers.
		g = np.zeros( n_features )												# Initialize zero gradient.
		betaset = np.zeros( ( rounds, n_features ) )							# Initialize storage for intermediate beta values to show progression.
		timeset = np.zeros( rounds )											# Initialize storage for iteration timings.
		worker_timeset = np.zeros( ( rounds, n_procs - 1 ) )					# Initialize storage for timings for each worker in each iteration.

		region1_timeset = np.zeros( rounds )
		region2_timeset = np.zeros( rounds )
		region3_timeset = np.zeros( rounds )
		region3a_timeset	= np.zeros( rounds )
		region3b_timeset 	= np.zeros( rounds )
		region3c_timeset 	= np.zeros( rounds )
		region3d_timeset 	= np.zeros( rounds )
		region4_timeset = np.zeros( rounds )
		region5_timeset = np.zeros( rounds )
		
		request_set = []
		recv_reqs = []

		cnt_completed = 0

		status = MPI.Status()

		eta0=params[ 2 ] 				# ----- learning rate schedule
		alpha = params[ 1 ] 			# --- coefficient of l2 regularization
		utemp = np.zeros( n_features ) 	# for accelerated gradient descent
   
	# Posting all Irecv requests for master and workers
	if (rank):
		for i in range(rounds):
			# Irecv format:
			# 	Irecv( [ array, count, datatype ], sourceRank, tag )

			req = comm.Irecv( [ beta, MPI.DOUBLE ], source = 0, tag = i )
			recv_reqs.append( req )

			stragglerReq = comm.Irecv( [ isStraggler, MPI.INT ], source = 0, tag = i )
			straggler_reqs.append( stragglerReq )
	else:
		for i in range(rounds):
			recv_reqs = []
			for j in range(1,n_procs):
				req = comm.Irecv([msgBuffers[j-1], MPI.DOUBLE], source=j, tag=i)
				recv_reqs.append(req)
			request_set.append(recv_reqs)

	print( "Rank {} - Time to initialize: {}.".format( rank, time.time() - initTime ) )
	# exit()
	########################################################################################################
	comm.Barrier()

	if rank==0:
		orig_start_time= time.time()
		print("---- Starting Naive Iterations ----")

	for i in range(rounds):
		if rank==0:
			### Region 1
			regionTime = time.time()

			if( i % 10 == 0):
				print("\t >>> At Iteration %d" %(i))

			start_time = time.time()

			region1_timeset[ i ] = start_time - regionTime
			###

			### Region 2
			regionTime = time.time()

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
			g[:]=0
			cnt_completed = 0

			# Stay here until ( n_workers - n_stragglers ) workers are done computing
			subRegionTime = 0
			while cnt_completed < n_procs-1:
				########
				region3d_timeset[ i ] += 0 if subRegionTime == 0 else ( time.time() - subRegionTime )

				########
				subRegionTime = time.time()

				# Wait until any worker is done and has sent its data
				req_done = MPI.Request.Waitany(request_set[i], status)

				region3a_timeset[ i ] += time.time() - subRegionTime
				########


				########
				subRegionTime = time.time()

				# Get the worker number
				src = status.Get_source()

				# Calculate the amount of time the worker took.
				worker_timeset[i,src-1]=time.time()-start_time

				# Remove the message request from the list.
				request_set[i].pop(req_done)

				region3b_timeset[ i ] += time.time() - subRegionTime
				########
				

				########
				subRegionTime = time.time()

				# Add the partial gradients.
				g+=msgBuffers[src-1]

				# Increment the count of completed workers.
				cnt_completed+=1

				region3c_timeset[ i ] += time.time() - subRegionTime
				########

				########
				subRegionTime = time.time()
			region3d_timeset[ i ] += time.time() - subRegionTime
			region3_timeset[ i ] = time.time() - regionTime
			###

			### Region 4
			regionTime = time.time()
			
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

			region4_timeset[ i ] = time.time() - regionTime
			###

			### Region 5
			regionTime = time.time()

			# MPI.Request.Cancel
			MPI.Request.Waitall( request_set[ i ] )

			region5_timeset[ i ] = time.time() - regionTime
			###

		else:
			beginTime = time.time()

			####
			startTime = beginTime

			# Wait for data to be received
			recv_reqs[ i ].Wait()
			straggler_reqs[ i ].Wait()

			waitingTimeset[ i ] = time.time() - startTime
			####


			####
			startTime = time.time()

			# Create a straggler by waiting a set amount of time before continuing
			if( isStraggler ):
				time.sleep( straggle_time )

			sleepingTimeset[ i ] = time.time() - startTime
			####


			# Compute partial gradient
			####
			startTime = time.time()

			predy = X_current.dot(beta)
			g = X_current.T.dot(np.divide(y_current,np.exp(np.multiply(predy,y_current))+1))
			g *= -1

			computingTimeset[ i ] = time.time() - startTime
			####


			# Send data back to master
			####
			startTime = time.time()

			send_req = comm.Isend([g, MPI.DOUBLE], dest=0, tag=i)
			
			sendingDataTimeset[ i ] = time.time() - startTime
			####

			totalTimeset[ i ] = time.time() - beginTime
	#####################################################################################################
	comm.Barrier()
	if rank==0:
		elapsed_time= time.time() - orig_start_time
		print ("Total Time Elapsed: %.3f" %(elapsed_time))
		print( f"Actual Running Time: {sum(timeset)}" )
		print( f"Average Running Time: {sum(timeset) / len(timeset) = }" )

		totalLoadTime = time.time()
		loadTime = 0.0
		vstackTime = 0.0

		# Load all training data		
		if not is_real_data:
			loadTimeStart = time.time()

			X_train = load_data( input_dir + "1.dat" )

			loadTime += time.time() - loadTimeStart

			print(">> Loaded 1")
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


		# Save simulation data.
		save_vector( training_loss, 	output_dir + "naive_acc_%d_training_loss.dat"	% ( n_stragglers ) )
		save_vector( testing_loss, 		output_dir + "naive_acc_%d_testing_loss.dat"	% ( n_stragglers ) )
		save_vector( auc_loss, 			output_dir + "naive_acc_%d_auc.dat"				% ( n_stragglers ) )
		save_vector( timeset, 			output_dir + "naive_acc_%d_timeset.dat"			% ( n_stragglers ) )

		save_vector( region1_timeset, 	output_dir + "naive_region1_timeset.dat" )
		save_vector( region2_timeset, 	output_dir + "naive_region2_timeset.dat" )
		save_vector( region3_timeset, 	output_dir + "naive_region3_timeset.dat" )
		save_vector( region4_timeset, 	output_dir + "naive_region4_timeset.dat" )
		save_vector( region5_timeset, 	output_dir + "naive_region5_timeset.dat" )

		save_matrix( worker_timeset, 	output_dir + "naive_acc_%d_worker_timeset.dat"	% ( n_stragglers ) )
		print(">>> Done")
		
		saveTime = time.time() - saveTime


		print( 	f"Total Load TIme:    {totalLoadTime:9.6f}\n" +
				f"Testing Load Time:  {testingLoadTime:9.6f}\n" +
				f"Training Load Time: {loadTime:9.6f}\n" +
				f"Vstack Time:        {vstackTime:9.6f}\n" +
				f"Loss Time:          {lossTime:9.6f}\n" +
				f"Save Time:          {saveTime:9.6f}\n\n" )
		print( 	f"Region 1:   	      {np.sum( region1_timeset  ):9.6f} ({np.average( region1_timeset  ):9.6f})" )
		print( 	f"Region 2:   	      {np.sum( region2_timeset  ):9.6f} ({np.average( region2_timeset  ):9.6f})" )
		print( 	f"Region 3:   	      {np.sum( region3_timeset  ):9.6f} ({np.average( region3_timeset  ):9.6f})" )
		print( 	f"Region 3a:  	      {np.sum( region3a_timeset ):9.6f} ({np.average( region3a_timeset ):9.6f})" )
		print( 	f"Region 3b:  	      {np.sum( region3b_timeset ):9.6f} ({np.average( region3b_timeset ):9.6f})" )
		print( 	f"Region 3c:  	      {np.sum( region3c_timeset ):9.6f} ({np.average( region3c_timeset ):9.6f})" )
		print( 	f"Region 3d:  	      {np.sum( region3d_timeset ):9.6f} ({np.average( region3d_timeset ):9.6f})" )
		print( 	f"Region 4:   	      {np.sum( region4_timeset  ):9.6f} ({np.average( region4_timeset  ):9.6f})" )
		print( 	f"Region 5:   	      {np.sum( region5_timeset  ):9.6f} ({np.average( region5_timeset  ):9.6f})" )
		print( 	f"Total Time: 	      {np.sum( timeset          ):9.6f} ({np.average( timeset          ):9.6f})" )
	else:
		# pass
		print( f"Rank {rank: 2d}\n" +
				f"    Total Time:        {np.sum( totalTimeset ): 5.3f}\n" +
				f"    Waiting Time:      {np.sum( waitingTimeset ): 5.3f}\n" +
				f"    Sleeping Time:     {np.sum( sleepingTimeset ): 5.3f}\n" +
				f"    Computing Time:    {np.sum( computingTimeset ): 5.3f}\n" +
				f"    Sending Data Time: {np.sum( sendingDataTimeset ): 5.3f}\n" )

	# Synchronized ending.
	comm.Barrier()