import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy.stats import beta
import math


filePath = "Data/save_alpha_beta_test/ts/"
trials = 1000000
delta = 100

numDataPoints = trials // delta

k = 2


alpha1 = np.zeros( numDataPoints + 1 )
beta1  = np.zeros( numDataPoints + 1 )
alpha2 = np.zeros( numDataPoints + 1 )
beta2  = np.zeros( numDataPoints + 1 )

data = np.zeros( numDataPoints + 1 )


iterationNumber = 0

# Read alpha-beta file
file = filePath + str( iterationNumber ) + "AB.dat"
print( "Reading from: %s" % ( file ) )

try:
	with open( file ) as f:
		# Skip the first two lines
		next( f )
		inputMu = next( f ).split()

		# Set time slot 0 as uniform
		alpha1[ 0 ] = 1
		beta1[ 0 ]  = 1
		alpha2[ 0 ] = 1
		beta2[ 0 ]  = 1

		for i in range( 1, numDataPoints + 1 ):
			inputAB = next( f ).split()

			alpha1[ i ] = float( inputAB[ 1 ] )
			beta1[ i ] = float( inputAB[ 2 ] )

			alpha2[ i ] = float( inputAB[ 3 ] )
			beta2[ i ] = float( inputAB[ 4 ] )
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
	# Switch alpha
	temp = alpha1
	alpha1 = alpha2
	alpha2 = temp

	# Switch beta
	temp = beta1
	beta1 = beta2
	beta2 = temp


# Create figure with 1 row of 2 columns
fig, ax = plt.subplots( nrows = 2, ncols = 1 )

# Set parameters for alpha-beta plot (col 1)
ax[ 0 ].set_xlim( 0, 1 )
ax[ 0 ].set_ylim( 0, None )

plt.subplots_adjust( bottom = 0.25 )

x = np.linspace( 0.01, 0.99, 1000 ) 

index = 0
y1 = beta.pdf( x, alpha1[ index ], beta1[ index ] )
y2 = beta.pdf( x, alpha2[ index ], beta2[ index ] )
arm1_plot, = ax[ 0 ].plot( x, y1, lw = 2 )
arm2_plot, = ax[ 0 ].plot( x, y2, lw = 2 )
# ax[ 0 ].set_ylim( 0, 1.1 * abPlotMaxY )
ax[ 0 ].margins( x = 0 )
ax[ 0 ].grid()
# ax[ 0 ].set_aspect( 'equal' )

# Plot default data
xData = range( 0, numDataPoints + 1 )
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

	# Create pdfs for both arms
	y1 = beta.pdf( x, alpha1[ index ], beta1[ index ] )
	y2 = beta.pdf( x, alpha2[ index ], beta2[ index ] )

	# Calculate maximum value for each pdf
	y1max = y1.max()
	y2max = y2.max()

	# print( "Index: %i\nMax Y1: %f\nMax Y2: %f\n" % ( index, y1max, y2max ) )

	# Find the maximum y-axis value
	scaleUp = 2.0
	scaleDown = 2.0
	currMax = ax[ 0 ].get_ylim()[ 1 ]
	if y1max > y2max:
		ax[ 0 ].set_ylim( 0, y1max * 1.1 )
	else:
		ax[ 0 ].set_ylim( 0, y2max * 1.1 )

	# Update the alpha-beta plots
	arm1_plot.set_ydata( y1 )
	arm2_plot.set_ydata( y2 )

	# Update the data plot
	dataPlot.set_data( xData[ 0:index ], data[ 0:index ] )

	# Change the text on the slider
	stime.valtext.set_text( f'{index*100:,}' )


	# Update the figure display
	fig.canvas.draw_idle()


stime.on_changed( update )
update( stime.val )

plt.show()
