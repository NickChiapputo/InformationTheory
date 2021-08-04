from __future__ import print_function
import sys
import os
import numpy as np
import random
import pandas as pd
from sklearn import  preprocessing
from sklearn.model_selection import train_test_split
from util import *
import itertools
import math

# This file contains code required for any preprocessing of real data, as well as splitting it into partitions 
# Currently this contains code relevant to the amazon-dataset (https://www.kaggle.com/c/amazon-employee-access-challenge)
# and dna dataset ftp://largescale.ml.tu-berlin.de/largescale/dna/

if len(sys.argv) != 7:
	print("Usage: python arrange_real_data.py n_procs input_dir real_dataset n_stragglers n_partitions partial_coded")
	sys.exit(0)

# Set the seed for the random generator.
np.random.seed(0)

# Parse the iput arguments.
n_procs, input_dir, real_dataset, n_stragglers, n_partitions, partial_coded  = [x for x in sys.argv[1:]]
n_procs, n_stragglers, n_partitions, partial_coded = int(n_procs), int(n_stragglers), int(n_partitions), int(partial_coded)

# Get the data directory.
input_dir = input_dir + real_dataset + "/"


# Prevent scientific method printing for NumPy. Not part of original source code.
np.set_printoptions( suppress = True )


# load relevant data
if real_dataset=="amazon-dataset":

	# Read in the data.
	print("Preparing data for "+real_dataset)
	trainData = pd.read_csv(input_dir + 'train.csv')


	# Take the 9 columns of every row that are not the binary labels.
	trainX = trainData.loc[:,'RESOURCE':].values


	# Take the binary action labels that determine if access was granted.
	trainY = trainData['ACTION'].values


	# Create a label encoder object to normalize the labels.
	relabeler = preprocessing.LabelEncoder()


	# Iterate through each column of the data (9 columns).
	# Number of data elements in the columns stays the same. They are just rearranged.
	# The actual values do not matter so much as the relationship between them and the 
	# labels matters, so it is okay to change them.
	print("")
	for col in range(len(trainX[0, :])):
		colSet = set( trainX[ :, col ] )
		print(  "================================================================\n" +
				"                           " + "\033[1;31m" + "Column " + str( col ) + "\n" + "\033[0;0m" +
				f"Data: {trainX[ :, col ]}, \n" +
				f"      Size = {trainX[ :, col ].size},\n" +
				f"      Unique Elements = {len( colSet )}\n" )


		# Fits the label encoder. The attribute 'classes_' of the LabelEncoder object
		# is a list of the unique values in the column (i.e., a set of the values).
		relabeler.fit(trainX[:, col])
		print(  f"Label Encoder Number of Classes = {relabeler.classes_.size}\n" +
				f"      Max Label Value: {max( relabeler.classes_ )}\n" )
		

		# Update the column and rearrange/transform it.
		# This replaces every value with its index in 
		# the LabelEncoder class list created by fit()
		trainX[:, col] = relabeler.transform(trainX[:, col])
		print(  "Transformed Column: {}\n" \
				"      Max Value in Column: {}\n" \
				"      Size = {}\n" \
				"================================================================\n".format( trainX[ :, col ], max( trainX[ :, col ] ), trainX[ :, col ].size ) )


	# Convert data into {-1, 1} binary format.
	trainY = 2*trainY - 1


	# Find all 2-tuples from the nine columns.
	# Then, get the data in the columns from the indicie of each 2-tuple (34 of them).
	# From the data, create a hash on each row for each of the tuples.
	# This results in data of size 32769 x 34.
	# 32769 = Number of rows in original data.
	# 34    = Number of ordered 2-tuples other than (5, 7) and (2, 3). Equivalent to (9 Choose 2) minus 2.
	d_all_s = interactionTermsAmazon(trainX, degree=2)  # second order


	# This was left commented out by the original creators:
	#d_all_t = interactionTermsAmazon(trainX, degree=3)  # third order
	#trainX = np.hstack((trainX, d_all_s, d_all_t))


	# Concatenate the original data and the tupled data along the second axis.
	# Adds the columns of d_all_s after the columns of trainX. Row numbers are equal.
	# Number of columns is 9 + 34 = 43. Row remains at 32769
	print(  "================================================================\n" \
			"                           " + "\033[1;31m" + "hstack\n" + "\033[0;0m" \
			"trainX: {}\nd_all_s: {}".format( trainX.shape, d_all_s.shape ) )
	trainX = np.hstack((trainX, d_all_s))
	print(  "trainX: {}\n{}\n" \
			"================================================================\n".format( trainX.shape, trainX ) )


	# For each column in the training data, get indicies for the unique values
	# and transform the data to match the indicies.
	for col in range(len(trainX[0, :])):
		relabeler.fit(trainX[:, col])
		trainX[:, col] = relabeler.transform(trainX[:, col])


	# Concatenate the rows of the transposed training data with a vector of ones.
	# Take the transpose of the trainX matrix (size 32769x43 -> 43x32769) and a vector of 32769 ones
	# and place the vector of ones in the last row (new size 44x32769). Then transpose it so that the vector of ones
	# is in the last column. New size of data is 32769x44.
	print( 	"================================================================\n" \
			"                           " + "\033[1;31m" + "vstack\n" + "\033[0;0m" \
			"trainX: {}\n{}\n\nones: {}\n{}\n".format( trainX.T.shape, trainX.T, np.ones( trainX.shape[ 0 ] ).shape, np.ones( trainX.shape[ 0 ] ) ) )
	trainX = np.vstack( [ trainX.T, np.ones( trainX.shape[ 0 ] ) ] ).T
	print( 	"trainX: {}\n{}\n" \
			"================================================================\n".format( trainX.shape, trainX ) )


	# From the training data created previously, split it into a training and testing data set.
	# The testing data is created from 20% of the data.
	X_train, X_valid, y_train, y_valid = train_test_split(trainX, trainY, test_size=0.2, random_state=0)
	print( 	"================================================================\n" \
			"                           " + "\033[1;31m" + "Splitting\n" + "\033[0;0m" \
			"Training Data: {}\nTraining Labels: {}\nTesting Data: {}\nTesting Labels: {}\n" \
			"================================================================\n".format( X_train.shape, y_train.shape, X_valid.shape, y_valid.shape ) )


	# 
	print(  "================================================================\n" \
			"                           " + "\033[1;31m" + "One Hot Encoding\n" + "\033[0;0m" \
			f"Unique: {len(np.unique(X_train))}\n" \
			"Training Data: {}\n{}\n\n" \
			"Testing Data: {}\n{}\n".format( X_train.shape, X_train, X_valid.shape, X_valid ) )
	encoder = preprocessing.OneHotEncoder(sparse=True)
	encoder.fit(np.vstack((X_train, X_valid)))			# Take the combined training and testing data and get indicies for each of the unique values.
	X_train = encoder.transform(X_train)  				# Returns a sparse matrix (see numpy.sparse)
	X_valid = encoder.transform(X_valid)				# Tranform the testing data to match the indicies for unique values.
	print(  "Training Data: {}\n{}\n\n" \
			"Testing Data: {}\n{}\n" \
			"================================================================\n".format( X_train.shape, X_train, X_valid.shape, X_valid ) )
	

	# Calculate the number of rows and columns in the new data.
	n_rows, n_cols = X_train.shape
	print("No. of training samples = %d, Dimension = %d"%(n_rows,n_cols))
	print("No. of testing samples = %d, Dimension = %d"%(X_valid.shape[0],X_valid.shape[1]))
	

	# Create output directory
	output_dir = input_dir
	if not partial_coded:
		output_dir = output_dir + str(n_procs-1) + "/"
		partitions = n_procs-1
	else:
		output_dir = output_dir + "partial/" + str((n_procs-1)*(n_partitions - n_stragglers))+"/"
		partitions = (n_procs-1)*(n_partitions - n_stragglers)

	n_rows_per_worker = n_rows//partitions


	# Check that the directory exists. If it doesn't, create it.
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)


	# Iterate through each partition and save it.
	for i in range(1, partitions+1):
		data_matrix = X_train[(i-1)*n_rows_per_worker:i*n_rows_per_worker,:]
		save_sparse_csr(output_dir+str(i),data_matrix)        
		print("\t >>> Done with partition %d" % (i))


	# Save the labels.
	save_vector(y_train, output_dir + "label.dat")
	save_vector(y_valid, output_dir + "label_test.dat")
	save_sparse_csr(output_dir + "test_data", X_valid)

elif real_dataset=="dna-dataset/dna":

	print("Preparing data for "+real_dataset)

	fin = open(input_dir + 'features.csv')
	trainData=  np.genfromtxt(itertools.islice(fin,0,500000,1), delimiter=',') 
	#np.genfromtxt(input_dir + 'features.csv',delimiter=',', max_rows=100000)
	trainX=trainData[:,1:]
	trainY=trainData[:,0]

	print("No. of positive labels = " + str(np.sum(trainY==1)))

	n,p = trainX.shape

	trainX=np.vstack([trainX.T,np.ones(trainX.shape[0])/math.sqrt(n)]).T

	X_train, X_valid, y_train, y_valid = train_test_split(trainX, trainY, test_size=0.2, random_state=0)

	encoder = preprocessing.OneHotEncoder(sparse=True)
	encoder.fit(np.vstack((X_train, X_valid)))
	X_train = encoder.transform(X_train)  # Returns a sparse matrix (see numpy.sparse)
	X_valid = encoder.transform(X_valid)
	
	n_rows, n_cols = X_train.shape
	print("No. of training samples = %d, Dimension = %d"%(n_rows,n_cols))
	print("No. of testing samples = %d, Dimension = %d"%(X_valid.shape[0],X_valid.shape[1]))
	
	# Create output directory
	output_dir = input_dir
	if not partial_coded:
		output_dir = output_dir + str(n_procs-1) + "/"
		partitions = n_procs-1
	else:
		output_dir = output_dir + "partial/" + str((n_procs-1)*(n_partitions - n_stragglers))+"/"
		partitions = (n_procs-1)*(n_partitions - n_stragglers)

	n_rows_per_worker = n_rows//partitions

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	for i in range(1, partitions+1):
		data_matrix = X_train[(i-1)*n_rows_per_worker:i*n_rows_per_worker,:]
		save_sparse_csr(output_dir+str(i),data_matrix)        
		print("\t >>> Done with partition %d" % (i))

	save_vector(y_train, output_dir + "label.dat")
	save_vector(y_valid, output_dir + "label_test.dat")
	save_sparse_csr(output_dir + "test_data", X_valid)

	fin.close()

print("Data Setup Finished.")