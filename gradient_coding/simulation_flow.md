# Steps for running a simulation
1. Use Makefile to select the scheme to run from the command line
	1. Set the parameters as wanted:
		* N_PROCS - 		Number of workers (plus one for the master).
		* N_STRAGGLERS - 	Number of stragglers in the scheme.
		* N_ITERATIIONS - 	Number of times to run the simulation.
		* STRAGGLER_DELAY - Delay length for stragglers in seconds.
		* N_PARTITIONS - 	For partially coded version: how many pieces of workload will one worker be handling.
		* PARTIAL_CODED - 	Switch to enable partial coded schemes.
		* DATA_FOLDER - 	Path to folder containing the data.
		* IS_REAL - 		Flag that determines whether data is real or not.
		* DATSET - 			Name of the real dataset.
		* N_ROWS - 			Number of rows in the dataset being used (real or not).
		* N_COLS - 			Number of columns in the dataset being used (real or not).
	2. Command used determines what scheme is used:
		1. Commands and the schemes use:
			* naive - 			Naive Scheme
			* cyccoded - 		Cyclic Repetition Scheme
			* repcoded - 		Fractional Repetition Scheme
			* avoidstragg - 	Avoid/Ignore Stragglers Scheme
			* partialrepcoded - Partial Repetition Scheme
			* partialcycoded - 	Partial Cyclic Scheme
		2. The aforementioned parameters are all sent to the program as command line arguments. Three additional arguments are sent that are static to tell the script which scheme is being used.
			* is_coded - 	Will be 0 if naive is being run or 1 otherwise.
			* partitions - 	Represents the number of partitions being used for partial schemes.
			* coded_ver - 	'Version number' for schemes within the same family. 
	3. When run, 'mpirun' command is used to create an MPI interface with N_PROCS number of different processes. The processes can be seen when using the 'top' command.
3. Each process runs the 'main.py' script first.
	1. Reads in the parameters and parses them from strings to integers and floats as necessary.
	2. Uses the static parameters to determine which scheme to run.
		* Runs a function from an included file.
	3. Sends parameters:
		* N_PROCS - 		Number of workers (plus one for master).
		* N_SAMPLES - 		Number of rows in dataset.
		* N_FEATURES - 		Number of columns in datset.
		* INPUT_DIR - 		Directory where the data is stored.
		* N_STRAGGLERS - 	Number of stragglers.
		* STRAGGLE_TIME - 	Delay for stragglers.
		* IS_REAL_DATA - 	Flag for using real or artificial data.
		* PARAMS - 			Parameter array, includes:
			* n_iterations - 			Number of simulations to be run.
			* alpha - 					Coefficient of l2 regularization.
			* learning_rate_schedule - 	Learning rate (used with gradient multiplier after each iteration).
4. Scheme Scripts
	1. Initialization
		* Shared Operations
			1. Creates communicator object 'comm'
			2. Creates 'rank' integer value. Represents ID number (0 = master, >0 = worker)
			3. Creates 'size' integer value. Equal to number of processes. This isn't used because the n_procs variable is passed as an argument to the function.
			4. Stores the number of iterations/rounds.
			5. Sets the number of workers as N_PROC - 1
			6. Calculates the number of rows of the dataset being used per worker.
		* Master
			1. Calculates the B matrix.
			2. Initializes gradient (all zeros).
			3. Initializes logging data (timesets, worker timesets)
			4. Broadcasts the B matrix to all workers.
			5. Generates Irecv requests for use when waiting for workers to complete. These are non-blocking requests and are stored in 'request_set'.
		* Workers
			1. Load pre-partitioned data insto X_current and labels into y_current.
			2. Initialize predy and gradient matrices
				* predy
					* Calculated as X^T * beta
					* However, beta is all zeros, so the result is just a zero matrix of size 
						* ( ( N_STRAGGLERS + 1 ) * ROWS_PER_WORKER ) x 1
				* g = X^T * ( y_current / exp( ( predy * y_current ) + 1 ) )
					* Size is N_COLS x 1
			3. Create Irecv requests for each iteration/round. These are non-blocking requests and are stored in 'recv_reqs'.
		* After each process is done with its initialization steps, it hits a Barrier() call.
			* This is a blocking call that causes the process to wait until every other process (all workers and master) have called it. Used to synchronize the system.
	2. Execution
		* System loops through each iteration. After the iterations are complete, each process hits another Barrier() call to synchronize.
		* Master
			1. Sets starting time value.
			2. Sends beta value to each process.
			3. Waits until a threshold number of processes have completed.
				* Threshold value is dependent on the scheme.
				* Steps followed when a process responds for completion:
					1. Calculate time taken for worker to complete.
					2. Sum the gradient.
					3. Increment completed count.
			4. Calculates gradient multiplier as 
				* alpha / N_ROWS
			5. Updates beta matrix values.
			6. Calculates total time taken for this iteration/round.
			7. Goes back to beginning of loop or, if loop is complete, goes to the Barrier() call at the end of the section.
		* Workers
			1. Previously created receive requests are used in a WAIT() call. This updates the beta matrix values.
			2. If the rank identifier for the worker is between 1 and N_STRAGGLERS (inclusive), time.sleep() is called to simulate the straggler delay.
			3. The updated gradient value is calculated. 
			4. The updated gradient is sent to the master using an Isend() call with source tag equal to the worker's rank identification number.
			5. The worker proceeds to the next loop (or Barrier() if the loop is complete) and pauses as the Wait() call until the master sends another set of data.
	3. Finalization
		* 
		* Master
			1. Calculates total elapsed time for all iterations/rounds.
			2. Loads training data   into X_train.
			3. Loads training labels into y_train.
			4. Loads testing  data   into X_test.
			5. Loads testing  labels into y_test.
			6. Imports roc_curve and auc functions from sklearn.metrics library.
			7. Loop through each iteration/round:
				1. Calculates predy_train for the training data
					* X_train * beta
				2. Calculates predy_test for the test data
					* X_test * beta
				3. Calculates training loss:
					* Sum( ln( 1 + exp( -1 * ( y_train * predy_train ) ) ) ) / N_TRAINING_ROWS
				4. Calculates testing loss:
					* Sum( ln( 1 + exp( -1 * ( y_test  * predy_test  ) ) ) ) / N_TEST_ROWS
				5. Calculate ROC curve using roc_curve function from sklearn.metrics.
					* Provide the true labels from y_test.
					* Provide the calculated labels from predy_test.
				6. Calculate the AUC of the ROC curve using auc function from sklearn.metrics.
					* Provide the y-values as the false-positives
					* Provide the x-values as the total positives.
			8. Output the data and save the results.
		* Workers
			1. Do nothing. Go to the Barrier() call at the end of the program and wait for master to finish before terminating.