import numpy as np
import matplotlib.pyplot as plt
import math


# Simulation parameters
GAMES = 1000000
A = 5
B = 50


# Parameters for Histogram Data
HIST_RANGE = 1500
DENSITY = True
TICK_FREQ = 100
TICKS = np.arange( 0, HIST_RANGE + 1, TICK_FREQ )
TICKS[ 0 ] = 1

# Parameters for Histogram Display
	# Histogram Title
HIST_SUP_TITLE = "Rounds for One Player to Win"
HIST_SUP_TITLE_SIZE = 'x-large'

	# Histogram subtitle
BETTING_STRATEGY = "Constant Amount ($1)"
HIST_TITLE = "A = \$" + str( A ) + "   B = \$" + str( B ) + "\nBetting Strategy: " + BETTING_STRATEGY
HIST_TITLE_SIZE = 'medium'

	# Histogram X-Label
HIST_X_LABEL = "Rounds"
HIST_X_LABEL_SIZE = 'x-large'

	# Histogram Y-Label
HIST_Y_LABEL = "Density"
HIST_Y_LABEL_SIZE = 'x-large'

	# Label Tick Number Size
HIST_TICK_SIZE = 'x-small'

	# Histogram Grid
HIST_GRID_BEHIND = True
HIST_GRID_AXIS = 'y'
HIST_GRID_ALPHA = 0.5

	# Histogram Edges
HIST_EDGE_COLOR = 'black'
HIST_LINE_WIDTH = 0.5


# Data input parameters
FILE = "data/results.dat"


# Data output parameters
OUT_FILE_TYPE = ".pdf"
OUT_FILE_BETTING_NAME = "constant"
OUT_FILE = "data/" + "a-" + str( A ) + "_b-" + str( B ) + "_" + OUT_FILE_BETTING_NAME + OUT_FILE_TYPE


# Data storage for number of rounds each game takes
rounds = np.zeros( GAMES )

numberOf1 = 0

print( "Reading from: %s" % ( FILE ) )

try:
	with open( FILE ) as f:
		for i in range( 0, GAMES ):
			# Get next line
			dataInput = next( f ).split()

			# Get number of rounds
			rounds[ i ] = int( dataInput[ 1 ] )

			if( rounds[ i ] == 2 or rounds[ i ] == 3 ):
				numberOf1 = numberOf1 + 1
except FileNotFoundError:
	print( "File is not found" )
	exit( 0 )
except PermissionError:
	print( "You don't have permission to access this file." )
	exit( 0 )

# Calculate bin width and convert to integer. Add one because int truncates decimals and width must be > 0
BIN_WIDTH = int( ( 1. / ( GAMES ** ( 1. / 3. ) ) ) * np.std( rounds ) * 3.49 ) + 1
if( BIN_WIDTH <= 1 ):
	BIN_WIDTH = 2;
BINS = np.arange( 0, HIST_RANGE + 1, BIN_WIDTH )

print( "Round Std: %f" % ( np.std( rounds ) ) )
print( "Bin Width: %f\n" % ( BIN_WIDTH ) )

# Create figure with one plot
fig, ax = plt.subplots( nrows = 1, ncols = 1, constrained_layout = True )

# Plot histogram
n, bins, patches = ax.hist( x = rounds, bins = BINS, range = ( 1, HIST_RANGE ), density = DENSITY, edgecolor = HIST_EDGE_COLOR, linewidth = HIST_LINE_WIDTH )

# Turn on grid
ax.grid( axis = HIST_GRID_AXIS, alpha = HIST_GRID_ALPHA )
ax.set_axisbelow( HIST_GRID_BEHIND )

# Format title
plt.title( HIST_TITLE, fontsize = HIST_TITLE_SIZE )

# Format sup title
plt.suptitle( HIST_SUP_TITLE, fontsize = HIST_SUP_TITLE_SIZE )

# Format y-axis
plt.ylabel( HIST_Y_LABEL, fontsize = HIST_Y_LABEL_SIZE )
plt.yticks( fontsize = HIST_TICK_SIZE )

# Format x-axis
plt.xlabel( HIST_X_LABEL, fontsize = HIST_X_LABEL_SIZE )
plt.xticks( ticks = TICKS, fontsize = HIST_TICK_SIZE )

# Set autoscale to remove empty spaces on sides of figure
ax.autoscale( enable = True, axis = 'x', tight = True )
ax.set_xlim( left = 1 )

# Maximize figure window
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

# Display figure
# plt.show()

# Save figure
print( "Saving to: %s" % ( OUT_FILE ) )
plt.savefig( OUT_FILE )
