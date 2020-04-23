import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import gc

k = [ "10", "100", "1000", "10000", "100000" ]
indices = [ 0, 0, 0, 0, 0 ]

trials = 1000000
delta = 1

usbFilePath = "/media/nick/CHIAPPUTO/MultiArmedBandit/Combined/Epsilon"

data = np.zeros( ( len( indices ), trials // delta ) )
maxValues = np.zeros( len( indices ) )
labels = [ "Explore Length = {}".format( x ) for x in k ] # ε - Copy this when needed

x = range( 0, trials, delta )
fig, ax = plt.subplots( 1, 1 )

for i in range( 0, len( indices ) ):
	print( "Gathering Data...%i/%i\r" %( i, len( indices ) ), end = '' )
	file = usbFilePath + str( k[ i ] ) + "/" + str( indices[ i ] ) + ".dat"
	with open( file ) as f:
		index = 0
		for j in range( 0, trials, delta ):
			data[ i ][ index ] = [ float( z ) for z in next( f ).split() ][ 1 ]
			index = index + 1

		ax.plot( x, data[ i ], label = "unknown" )	
		print( "%s: %f" % ( k[ i ], data[ i ][ index - 1 ] ) )

	# print( "Data[ %i ][ %i ] = %f\nStd dev = %f" % ( i + 1, trials, data[ i ][ index - 1 ], np.std( data[ i ] ) ) )
	maxValues[ i ] = data[ i ][ index - 1 ]
	print( "Gathering Data...%i/%i\r" %( i + 1, len( indices ) ), end = '' )

print( "" )


plt.xlabel( "Time" )
ax.set_xscale( 'log' )
ax.set_xlim( 100, trials )

plt.ylabel( "Regret" )
# ax.set_ylim( 0, np.max( data[ 0 ] ) )
# ax.yaxis.set_minor_locator( MultipleLocator( 10 ) )

ax.tick_params( labelleft = False, labelright = True, left = False, right = True, axis = 'both', which = 'both', labelsize = 14 )

ax.margins( x = 0, y = 0 )
plt.grid( True, which = 'both', axis = 'both', alpha = 0.35 )
plt.tight_layout()

# Sort labels from largest to smallest to align with the plots
# order = np.argsort( data[ :, -1 ] )[ : : -1 ]
plt.legend( labels = list( np.array( labels ) ), loc = 'upper left' )

# Display  output
plt.show()

# Save as pdf
# print( "Saving...\r", end = '' )
# outPath = "Plots/ChangingEpsilon/epsilonLinear" + ".pdf"
# fig.savefig( outPath, bbox_inches = 'tight' )
# print( "Saved to " + outPath )

gc.collect()