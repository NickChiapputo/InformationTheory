import numpy as np
import matplotlib.pyplot as plt

# Define the schemes with their string representations
naive = "naive"
cyclicRepetition = "coded"
fractionRepetition = "replication"
avoidStragglers = "avoidstragg"

# Set the number of workers and stragglers
workers = 10
stragglers = 0

# Directory for the dataset
# directory = "/home/nick/Documents/InformationTheory/gradient_coding/src/straggdata/artificial-data/55440x100/"
# directory = "/home/nick/Documents/InformationTheory/gradient_coding/src/straggdata/amazon-dataset/"
# directory = "/home/nick/Documents/InformationTheory/gradient_coding/src/data/08-13-2020/avoidstragg_straggler_test/artificial-data/"
# directory = "/home/nick/Documents/InformationTheory/gradient_coding/src/data/08-13-2020/naive_vs_avoidstragg/"
directory = "/home/nick/Documents/InformationTheory/gradient_coding/src/data/08-20-2020/avoidstragg_data-size_test/artificial-data/55440x100/"

def getData( scheme, workers, stragglers ):
	# Get AUC and time data for the given scheme with the given number of workers

	### Data in original results folder.
	# data_file = open( directory + str( workers ) + "/results/" + scheme + "_acc_" + str( stragglers ) + "_testing_loss.dat","r" )
	# time_file = open( directory + str( workers ) + "/results/" + scheme + "_acc_" + str( stragglers ) + "_timeset.dat", 	"r" )
	

	### Data in custom folder.
	data_file = open( directory + str( workers ) + "/" + scheme + "_acc_" + str( stragglers ) + "_testing_loss.dat","r" )
	time_file = open( directory + str( workers ) + "/" + scheme + "_acc_" + str( stragglers ) + "_timeset.dat", 	"r" )


	# Get an array of the data from each line
	data =  data_file.read().split( " \n" )
	times = time_file.read().split( " \n" )


	# Delete the last line because it is empty.
	del data[ len( data ) - 1 ]
	del times[ len( times ) - 1 ]


	# Parse the data into floating-point values.
	# x_data = [ float( x ) for x in times ]
	x_data = np.arange( 1, len( data ) + 1 )
	# print( x_data )


	# Calculate a cumulative sum of the data.
	i = 0;
	for i in range( 1, len( x_data ) ):
		# x_data[ i ] = x_data[ i ] + x_data[ i - 1 ]
		i = i + 1
	y_data = [ float( x ) for x in data ]


	# Return the x and y data arrays
	return x_data, y_data

# Create the figure
# fig = plt.figure( figsize = ( 9, 9 ) )


######################### FIGURE 7A #########################

# Select the first plot
# plt.subplot( 1, 2, 1 )

# x_data, y_data = getData( naive, workers, stragglers )								# Get AUC and time data for Cyclic Repetition with 10 workers.
# plt.plot( x_data, y_data, label="Naive, s=1", color="black" )		# Plot the data

# x_data, y_data = getData( cyclicRepetition, workers, stragglers )					# Get AUC and time data for Cyclic Repetition with 10 workers.
# plt.plot( x_data, y_data, label="CycRep, s=1", color="green" )		# Plot the data

# x_data, y_data = getData( fractionRepetition, workers, stragglers )					# Get AUC and time data for Fractional Repetition with 10 workers
# plt.plot( x_data, y_data, label="FracRep, s=1", color="blue" )		# Plot the data

# x_data, y_data = getData( avoidStragglers, workers, stragglers )						# Get AUC and time data for Fractional Repetition with 10 workers
# plt.plot( x_data, y_data, label="IgnoreStragg, s=0", color="red" )	# Plot the data


### AvoidStragg Worker Test Amazon Dataset
colors = [ "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan" ]
arrowprops = dict( arrowstyle="-" )
for i in range( 10 ):
    x_data, y_data = getData( avoidStragglers, workers, i )         # Get AUC and time data for Fractional Repetition with 10 workers
    plt.annotate( f"s={i}", xy=(100, y_data[ 99 ]), xytext=(100.5, y_data[ 99 ]), arrowprops=arrowprops )       
    plt.plot( x_data, y_data, label=f"s={i}", color=colors[ i ] )   # Plot the data



### AvoidStragg Worker Test Artificial Dataset
# x_data, y_data = getData( avoidStragglers, workers, 0 )						# Get AUC and time data for Fractional Repetition with 10 workers
# plt.plot( x_data, y_data, label="AvoidStragg, s=0", color="tab:blue" )		# Plot the data
# x_data, y_data = getData( avoidStragglers, workers, 3 )						# Get AUC and time data for Fractional Repetition with 10 workers
# plt.plot( x_data, y_data, label="AvoidStragg, s=3", color="tab:olive" )		# Plot the data
# x_data, y_data = getData( avoidStragglers, workers, 5 )						# Get AUC and time data for Fractional Repetition with 10 workers
# plt.plot( x_data, y_data, label="AvoidStragg, s=5", color="tab:red" )			# Plot the data
# x_data, y_data = getData( avoidStragglers, workers, 7 )						# Get AUC and time data for Fractional Repetition with 10 workers
# plt.plot( x_data, y_data, label="AvoidStragg, s=7", color="tab:purple" )		# Plot the data
# x_data, y_data = getData( avoidStragglers, workers, 9 )						# Get AUC and time data for Fractional Repetition with 10 workers
# plt.plot( x_data, y_data, label="AvoidStragg, s=9", color="tab:green" )		# Plot the data


### Naive vs AvoidStragg Amazon Dataset
# x_data, y_data = getData( naive, 			workers, 7 )					# Get AUC and time data for Fractional Repetition with 10 workers
# plt.plot( x_data, y_data, label="Naive, s=7", 		color="tab:blue" )			# Plot the data
# x_data, y_data = getData( avoidStragglers,	workers, 7 )						# Get AUC and time data for Fractional Repetition with 10 workers
# plt.plot( x_data, y_data, label="AvoidStragg, s=7", color="tab:cyan" )			# Plot the data
# x_data, y_data = getData( naive, 			workers, 9 )						# Get AUC and time data for Fractional Repetition with 10 workers
# plt.plot( x_data, y_data, label="Naive, s=9", 		color="tab:red" )			# Plot the data
# x_data, y_data = getData( avoidStragglers, 	workers, 9 )						# Get AUC and time data for Fractional Repetition with 10 workers
# plt.plot( x_data, y_data, label="AvoidStragg, s=9", color="tab:orange" )		# Plot the data


plt.title( "Data Size = 55,440 x 100" )
plt.xlabel( "Iteration Number" )
# plt.xlabel( "Time (in seconds)" )
plt.ylabel( "Testing Loss" )
# plt.title( "AUC vs Time on Amazon Employee Access Dataset" )
# plt.tight_layout()
plt.legend()


# Set plot limits
axes = plt.gca()
axes.set_xlim( [    0,   100 ] )
# axes.set_ylim( [ 0.13,  0.18 ] )

#############################################################
######################### FIGURE 7B #########################

# Select the second plot
# plt.subplot( 1, 2, 2 )


# directory = "/home/nick/Documents/InformationTheory/gradient_coding/src/data/08-20-2020/avoidstragg_data-size_test/artificial-data/554400x100/"


# x_data, y_data = getData( avoidStragglers, workers, 6 )							# Get AUC and time data for Fractional Repetition with 10 workers
# plt.plot( x_data, y_data, label="AvoidStragg, s=6", color="tab:brown" )			# Plot the data
# x_data, y_data = getData( avoidStragglers, workers, 7 )							# Get AUC and time data for Fractional Repetition with 10 workers
# plt.plot( x_data, y_data, label="AvoidStragg, s=7", color="tab:pink" )			# Plot the data
# x_data, y_data = getData( avoidStragglers, workers, 8 )							# Get AUC and time data for Fractional Repetition with 10 workers
# plt.plot( x_data, y_data, label="AvoidStragg, s=8", color="tab:gray" )			# Plot the data
# x_data, y_data = getData( avoidStragglers, workers, 9 )							# Get AUC and time data for Fractional Repetition with 10 workers
# plt.plot( x_data, y_data, label="AvoidStragg, s=9", color="tab:olive" )			# Plot the data


# plt.title( "Data Size = 554,400 x 100" )
# plt.xlabel( "Iteration Number" )
# plt.xlabel( "Time (in seconds)" )
# plt.ylabel( "Testing Loss" )
# plt.title( "AUC vs Time on Amazon Employee Access Dataset" )
# plt.tight_layout()
# plt.legend()

# #############################################################
# ######################### FIGURE 7C #########################

# # Select the third plot
# plt.subplot( 1, 3, 1 )

# x_data, y_data = getData( cyclicRepetition, 30 )					# Get AUC and time data for Cyclic Repetition with 10 workers.
# plt.plot( x_data, y_data, label="CycRep, s=1", color="green" )		# Plot the data

# x_data, y_data = getData( fractionRepetition, 30 )					# Get AUC and time data for Fractional Repetition with 10 workers
# plt.plot( x_data, y_data, label="FracRep, s=1", color="blue" )		# Plot the data

# x_data, y_data = getData( avoidStragglers, 30 )						# Get AUC and time data for Fractional Repetition with 10 workers
# plt.plot( x_data, y_data, label="IgnoreStragg, s=1", color="red" )	# Plot the data

#############################################################

# ax.autoscale( tight = True )
plt.show()
