import numpy as np
import matplotlib.pyplot as plt

# Define the schemes with their string representations
naive = "naive"
cyclicRepetition = "coded"
fractionRepetition = "replication"
avoidStragglers = "avoidstragg"

# Set the number of workers and stragglers
stragglers = 1

# Directory for the dataset
top_level_directory = "data/fig7/"

directories = [
	top_level_directory + "a/",	# Figure 7A
	top_level_directory + "b/",	# Figure 7B
	top_level_directory + "c/"	# Figure 7C
]

def getData( scheme, workers, stragglers, directory ):
	# Get AUC and time data for the given scheme with the given number of workers

	### Data in original results folder.
	# data_file = open( directory + str( workers ) + "/results/" + scheme + "_acc_" + str( stragglers ) + "_testing_loss.dat","r" )
	# time_file = open( directory + str( workers ) + "/results/" + scheme + "_acc_" + str( stragglers ) + "_timeset.dat", 	"r" )
	

	### Data in custom folder.
	# data_file = open( directory + scheme + "_acc_" + str( stragglers ) + "_testing_loss.dat","r" )
	data_file = open( directory + scheme + "_acc_" + str( stragglers ) + "_auc.dat","r" )
	time_file = open( directory + scheme + "_acc_" + str( stragglers ) + "_timeset.dat", 	"r" )


	# Get an array of the data from each line
	data =  data_file.read().split( " \n" )
	times = time_file.read().split( " \n" )


	# Delete the last line because it is empty.
	del data[ -1 ]
	del times[ -1 ]


	# Parse the data into floating-point values and accumulate
	# to get the total time by each AUC step.
	x_data = np.cumsum( [ float( x ) for x in times ] )


	# Convert the AUC values into floating-point values from strings.
	y_data = [ float( x ) for x in data ]


	# Return the x and y data arrays
	return x_data, y_data


colors = [
	[ "b", "g", "r" ],					# Figute 7A
	[ "b", "g", "r", "c", "m", "y" ],	# Figure 7B
	[ "b", "g", "r", "c", "m", "y" ]	# Figure 7C
]

labels = [
	[ "FracRep s=1", "CycRep s=1", "IgnoreStragg s=1" ],													# Figure 7A
	[ "FracRep s=3", "FracRep s=4", "CycRep s=3", "CycRep s=4", "IgnoreStragg s=3", "IgnoreStragg s=4" ],	# Figure 7B
	[ "FracRep s=4", "FracRep s=9", "CycRep s=4", "CycRep s=9", "IgnoreStragg s=4", "IgnoreStragg s=9" ]	# Figure 7C
]

schemes = [
	[ fractionRepetition, cyclicRepetition, avoidStragglers ],															# Figure 7A
	[ fractionRepetition, fractionRepetition, cyclicRepetition, cyclicRepetition, avoidStragglers, avoidStragglers ],	# Figure 7B
	[ fractionRepetition, fractionRepetition, cyclicRepetition, cyclicRepetition, avoidStragglers, avoidStragglers ],	# Figure 7C
]

workers = [
	10,	# Figure 7A
	20,	# Figure 7B
	30	# Figure 7C
]

stragglers = [
	[ 1, 1, 1 ],			# Figure 7A
	[ 3, 4, 3, 4, 3, 4 ],	# Figure 7B
	[ 4, 9, 4, 9, 4, 9 ]	# Figure 7C
]


save_location_dir = "data/fig7/"
save_location_fname = [
	"fig7a.pdf",
	"fig7b.pdf",
	"fig7c.pdf"
]


# Select the first plot
rows = 1
cols = 1
subplot_idx = 2
plt.rcParams["font.size"] = 16
plt.subplot( rows, cols, 1 )

# x_data, y_data = getData( naive, workers, stragglers )								# Get AUC and time data for Cyclic Repetition with 10 workers.
# plt.plot( x_data, y_data, label="Naive, s=1", color="black" )		# Plot the data

for i in range( len( colors[ subplot_idx ] ) ):
	# Get AUC and time data for the current scheme and worker values.
	x_data, y_data = getData( schemes[ subplot_idx ][ i ], workers[ subplot_idx ], stragglers[ subplot_idx ][ i ], directories[ subplot_idx ] )			
	plt.plot( x_data, y_data, label = labels[ subplot_idx ][ i ], color = colors[ subplot_idx ][ i ] )


# plt.title( "Data Size = 55,440 x 100" )
plt.title( f"AUC vs. Time on n={workers[ subplot_idx ]} workers" )
plt.xlabel( "Time (in seconds)" )
# plt.xlabel( "Time (in seconds)" )
plt.ylabel( "AUC" )
# plt.title( "AUC vs Time on Amazon Employee Access Dataset" )
plt.tight_layout()
plt.legend()


# Set plot limits
axes = plt.gca()
# axes.set_xlim( [    0,   100 ] )
# axes.set_ylim( [ 0.13,  0.18 ] )
plt.savefig( save_location_dir + save_location_fname[ subplot_idx ] )

# plt.show()
