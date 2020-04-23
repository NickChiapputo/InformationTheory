import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import os

# Use this to copy paste the ε symbol
# ε-Greedy

trials = 1000000
simulations = 1000
vals = [ "10" ]
numAvg = 1
delta = 1
algs = [ "Thompson Sampling", "UCB", "ε-Greedy" ]
plots = len( vals ) * len( algs )
colors = [ "orange", "green", "blue" ]
filePath = "/media/nick/TONKS/MultiArmedBandit/0."

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
for i in range( 0, len( vals ) ):
	print( "Gathering Data...%i\r" % ( count + 1 ), end = '' )
	file = filePath + vals[ i ] + "/0.dat"
	if os.path.exists( file ):
		with open( file ) as f:
			# try:
			index = 0
			for j in range( 0, trials, delta ):
				z = next( f ).split()
				for k in range( 0, len( algs ) ):
					data[ len( algs ) * count + k ][ index ] = float( z[ k + 1 ] )
				index = index + 1

			# for k in range( 0, len( algs ) ):
			# 	labels[ len( algs ) * count + k ] = algs[ k ]
			# except:
			# 	print( "Error opening %s" % ( file ) )

		# print( "Data[ %i ][ %i ] = %f" % ( i + 1, trials, data[ i ][ index - 1 ] ) )
		maxValues[ len( algs ) * count + k ] = data[ len( algs ) * count + k ][ index - 1 ]
	else:
		print( "Error opening '%s'" % ( file ) )
	count = count + 1

order = np.argsort( data[ :, -1 ] )[ : : -1 ]
# data = data[ order ]

for i in range( 0, plots ):
	# label = algs[ i % len( algs ) ] + " K = " + str( vals[ i % len( vals ) ] )
	ax.plot( x, data[ i ], color = colors[ i % len( algs ) ] )

print( "" )
ax.margins( x = 0, y = 0 )
ax.set_xscale( 'log' )
ax.set_xlim( 100, trials )
ax.set_ylim( 0, 450 )

ax.tick_params( labelleft = False, labelright = True, left = False, right = True, axis = 'both', which = 'both', labelsize = 14 )
# ax.yaxis.set_major_locator( MultipleLocator( 10 ) )

plt.xlabel( "Time" )
plt.ylabel( "Regret" )
plt.grid( True, which = 'both', axis = 'both', alpha = 0.35 )
plt.tight_layout()
plt.legend( loc = 'upper left', labels = algs )

# Used to display range plot instead of all executions
# ax.plot( x, avgData )
# ax.plot( x, maxData )
# ax.plot( x, minData )
# ax.fill_between( x, maxData, minData, alpha = 0.2 )


# Display output
# print( "Plotting data..." )
plt.show()

# Save as pdf
# print( "Saving..." )
# outPath = "Plots/Combined/_" + vals[ 0 ] + ".pdf"
# fig.savefig( outPath, bbox_inches = 'tight' )
# print( "Saved to " + outPath )