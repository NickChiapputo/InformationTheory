import numpy as np

# Get name of scheme.
#			Naive 	 Cyclic 	Fractional 		Ignore
schemes = [ "naive", "coded", "replication", "avoidstragg" ]
schemeNum = 2


# Get name of dataset.
datasets = [ "artificial-data/", "amazon-dataset/" ]
datasetNum = 0


# Get size of dataset.
size = "554400x100/" if datasetNum == 0 else ""


# Get number of workers.
numWorkers = 12
workers = str( numWorkers ) + "/" 


# Get the top-level directory that holds the data.
topLevelDirectory = "/home/nick/Documents/InformationTheory/gradient_coding/src/straggdata/"


# Get the number of stragglers.
stragglerNum = 1
stragglers = "" if schemeNum == 0 else "_" + str( stragglerNum )


# Get the desired data file. 
information = "_acc" + stragglers + "_timeset.dat"


# Determine the full file path.
fileName = topLevelDirectory + datasets[ datasetNum ] + size + workers + "results/" + schemes[ schemeNum ] + information
print( "Opening file '{}'".format( fileName ) )


##################################
########### Total Time ###########

time_file = open( fileName, "r" )				# Open the file in read mode.
times = time_file.read().split( " \n" )			# Split the file by space and newline. Store each data point individually in an array.

del times[ len( times ) - 1 ]					# Delete trailing empty line.
x_data = [ float( x ) for x in times ]			# Convert each data point from string to float.
numIterations = len( x_data )					# Get number of iterations in the simulation.
timeAvg = np.average( x_data )					# Calculate average of data.

# Display total average time and number of iterations.
print( "Iterations: {}\n\nAverage Total Time: {}".format( numIterations, timeAvg ) )

##################################
##################################


##################################
########## Region Times ##########

print( "\n" )

numRegions = 5									# Set number of regions.
regionTime = np.zeros( numRegions )				# Create array to hold average times.
regionPercentage = np.zeros( numRegions )		# Create array to hold percentages of total time for each average.

for i in range( 0, numRegions ):
	# Determine the full file path to region i data.
	fileName = topLevelDirectory + datasets[ datasetNum ] + size + workers + "results/" + schemes[ schemeNum ] + "_region" + str( i + 1 ) + "_timeset.dat"

	time_file = open( fileName, "r" )			# Open the file in read mode.
	times = time_file.read().split( " \n" )		# Split the file by space and newline. Store each data point individually in an array.

	del times[ len( times ) - 1 ]				# Delete trailing empty line.
	x_data = [ float( x ) for x in times ]		# Convert each data point from string to float.
	regionTime[ i ] = np.average( x_data )		# Calculate average of data.

	print( "Region " + str( i + 1 ) + ": {}".format( regionTime[ i ] ) )

print( "Region Sum: {}".format( np.sum( regionTime ) ) )

##################################
##################################


##################################
########## Worker Times ##########

print( "\n" )

# Determine the full file path to the worker data.
fileName = topLevelDirectory + datasets[ datasetNum ] + size + workers + "results/" + schemes[ schemeNum ] + "_acc" + stragglers + "_worker_timeset.dat"
time_file = open( fileName, "r" )				# Open the file in read mode.
times = time_file.read().split( "\n" )			# Split the file by newline. Store each data set individually in an array.

del times[ len( times ) - 1 ]					# Delete trailing empty line.

worker_average = np.zeros( numWorkers )			# Create average time for each worker.

# Split each iteration into individual worker
# times, then convert to floats.
worker_times = [ times[ x ].split( " " ) for x in range( 0, numIterations ) ]
for i in range( 0, numIterations ):
	worker_times[ i ] = [ float( x ) for x in worker_times[ i ] ]
worker_averages = np.average( worker_times, axis = 0 )

# Print average for each worker.
for i in range( 0, numWorkers ):
	print( "Worker {} average: {}".format( ( i + 1 ), worker_averages[ i ] ) )
print( "Average per Worker: {}".format( np.average( worker_averages ) ) )

##################################
##################################
