import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import os

trials = 1000000
simulations = 500
k = 2
numAvg = 1
plots = 1
delta = 100

filePath = "Data/save_alpha_beta_test/"

# Inform user what is being plotted
print( "Plotting %i iterations of %i simulations with %i trials. (NUM_AVG: %i)" % ( plots, simulations, trials, numAvg ) )
print( "---------------------------------------------------------------------------------" )

# Create matrix with 'plots' rows and 'trials' columns
data = np.zeros( ( plots, trials // delta ) )
maxData = np.zeros( trials )
minData = np.zeros( trials )
avgData = np.zeros( trials )
x = range( 0, trials, delta )
maxValues = np.zeros( plots )

fig, ax = plt.subplots( 1, 1 )

count = 0
for i in range( 0, plots ):
	print( "Gathering Data...%i\r" %( i + 1 ) , end = '' )
	file = filePath + str( i ) + ".dat"
	if os.path.exists( file ):
		with open( file ) as f:
			try:
				index = 0
				for j in range( 0, trials, delta ):
					data[ i ][ index ] = [ float( z ) for z in next( f ).split() ][ 1 ]
					index = index + 1

				ax.plot( x, data[ i ] )
			except:
				print( "Error opening %s" % ( file ) )

		print( "Data[ %i ][ %i ] = %f" % ( i + 1, trials, data[ i ][ index - 1 ] ) )
		maxValues[ i ] = data[ i ][ index - 1 ]
print( "" )
ax.margins( x = 0, y = 0 )
ax.set_xscale( 'log' )
ax.set_xlim( 100, trials )
ax.set_ylim( 0, 225 )

ax.tick_params( labelleft = False, labelright = True, left = False, right = True, axis = 'both', which = 'both', labelsize = 14 )
# ax.yaxis.set_minor_locator( MultipleLocator( 10 ) )

plt.xlabel( "Time" )
plt.ylabel( "Regret" )
plt.grid( True, which = 'both', axis = 'both', alpha = 0.35 )
plt.tight_layout()

# Used to display range plot instead of all executions
# ax.plot( x, avgData )
# ax.plot( x, maxData )
# ax.plot( x, minData )
# ax.fill_between( x, maxData, minData, alpha = 0.2 )


# Display output
plt.show()

# Save as pdf
# print( "Saving..." )
# outPath = "Plots/Thompson_Simulations/" + str( simulations ) + ".pdf"
# fig.savefig( outPath, bbox_inches = 'tight' )
# print( "Saved to " + outPath )