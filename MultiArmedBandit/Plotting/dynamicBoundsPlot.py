import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy.stats import beta
import math


filePath = "Data/save_alpha_beta_test/ucb/"
trials = 1000000
delta = 100

numDataPoints = trials // delta

k = 2


bound1 = np.zeros( numDataPoints + 1 )
arm1_referenceData = np.zeros( numDataPoints + 1 )
bound2  = np.zeros( numDataPoints + 1 )
arm2_referenceData = np.zeros( numDataPoints + 1 )

data = np.zeros( numDataPoints + 1 )


iterationNumber = 0

# Read alpha-beta file
file = filePath + str( iterationNumber ) + "_bounds.dat"
print( "Reading from: %s" % ( file ) )

try:
	with open( file ) as f:
		# Skip the first two lines
		next( f )
		inputMu = next( f ).split()

		# Set time slot 0 as uniform
		bound1[ 0 ] = 0
		arm1_referenceData[ 0 ] = inputMu[ 0 ]
		bound2[ 0 ]  = 0
		arm2_referenceData[ 0 ] = inputMu[ 1 ]

		for i in range( 1, numDataPoints + 1 ):
			inputAB = next( f ).split()

			bound1[ i ] = float( inputAB[ 1 ] )
			arm1_referenceData[ i ] = inputMu[ 0 ]
			bound2[ i ] = float( inputAB[ 2 ] )
			arm2_referenceData[ i ] = inputMu[ 1 ]

except FileNotFoundError:
	print( "File is not found" )
	exit( 0 )
except PermissionError:
	print( "You don't have permission to access this file." )
	exit( 0 )


# Read data file
file = filePath + str( iterationNumber ) + ".dat"
print( "Reading from: %s" % ( file ) )

with open( file ) as f:
	data[ 0 ] = 0
	for i in range( 1, numDataPoints + 1 ):
		data[ i ] = next( f ).split()[ 1 ]

# Set arm 1 as best arm
if( inputMu[ 0 ] < inputMu[ 1 ] ):
	# Switch bound
	temp = bound1
	bound1 = bound2
	bound2 = temp


# Create figure with 1 row of 2 columns
fig, ax = plt.subplots( nrows = 2, ncols = 1 )


plt.subplots_adjust( bottom = 0.25 )

xData = range( 0, numDataPoints + 1 )

index = 0
arm1_plot, = ax[ 0 ].plot( xData, bound1, lw = 2, label = 'Arm 1' )
arm2_plot, = ax[ 0 ].plot( xData, bound2, lw = 2, label = 'Arm 2' )
arm1_reference, = ax[ 0 ].plot( xData, arm1_referenceData, linestyle = 'dashed' )
arm2_reference, = ax[ 0 ].plot( xData, arm2_referenceData, linestyle = 'dashed' )
ax[ 0 ].set_xscale( 'log' )
ax[ 0 ].set_xlim( 1, numDataPoints )
ax[ 0 ].set_ylim( 0, 1 )
ax[ 0 ].margins( x = 0 )
ax[ 0 ].grid()

# Plot default data
dataPlot, = ax[ 1 ].plot( xData, data, lw = 2 )
ax[ 1 ].set_xscale( 'log' )
ax[ 1 ].set_xlim( 1, numDataPoints )
ax[ 1 ].set_ylabel( 'Regret' )
ax[ 1 ].set_xlabel( 'Time' )
ax[ 1 ].grid()

axcolor = 'lightgoldenrodyellow'
axtime = plt.axes( [ 0.2, 0.1, 0.65, 0.03 ], facecolor = axcolor )

# Create slider for dynamic graph
# Range is [ 0, trials ]
# Start at 0
# Step by 1
stime = Slider( axtime, 'Time Slot', valmin = 0, valmax = math.log10( numDataPoints ), valinit = 0, valstep = 0.01 )

def update( val ):
	# Get time index
	if float( stime.val ) == 0:
		index = 0
	else:
		index = int( 10 ** float( stime.val ) )

	# Update the alpha-beta plots
	arm1_plot.set_data( xData[ 0:index ], bound1[ 0:index ] )
	arm2_plot.set_data( xData[ 0:index ], bound2[ 0:index ] )

	# Update the data plot
	dataPlot.set_data( xData[ 0:index ], data[ 0:index ] )

	# Change the text on the slider
	stime.valtext.set_text( f'{index*100:,}' )


	# Update the figure display
	fig.canvas.draw_idle()


stime.on_changed( update )
update( stime.val )


plt.show()
