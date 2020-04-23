from scipy.stats import beta
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

simulations = 500
numAvg = 1
iterations = 200

filePath = "Data/Thompson_Epsilon_0.40/"

bestArmAlpha = 0.0
bestArmBeta = 0.0

worstArmAlpha = 0.0
worstArmBeta = 0.0

iterationNumber = 42
file = filePath + str( iterationNumber ) + "AB.dat"
print( "Reading from: %s" % ( file ) )
with open( file ) as f:
	# Get alpha-beta 1
	inputAB = next( f ).split()
	alpha1 = float( inputAB[ 1 ] )
	beta1 = float( inputAB[ 2 ] )

	# Get alpha-beta 2
	inputAB = next( f ).split()
	alpha2 = float( inputAB[ 1 ] )
	beta2 = float( inputAB[ 1 ] )

	next( f )
	next( f )
	inputMu = next( f ).split()	

	# Get arm averages

# Set arm 1 as best arm
if( inputMu[ 0 ] < inputMu[ 1 ] ):
	temp = alpha1
	alpha1 = alpha2
	alpha2 = temp

	temp = beta1
	beta1 = beta2
	beta2 = temp

x = np.linspace( 0.01, 0.99, 1000 )
y1 = beta.pdf( x, alpha1, beta1 )
y2 = beta.pdf( x, alpha2, beta2 )
fig, ax = plt.subplots( 1, 1 )
ax.set_xlim( 0, 1 )
ax.plot( x, y1, label = 'Arm 1 (best)' )
ax.fill_between( x, y1, 0, alpha = 0.2 )
ax.fill_between( x, y2, 0, alpha = 0.2 )

ax.plot( x, y2, label = 'Arm 2 (worst)' )
ax.margins( x = 0, y = 0 )
ax.legend( loc = 'best' )

# Display output
plt.tight_layout()
plt.show()

# Save as pdf
outPath = "Plots/ChangingEpsilon/Thompson_4AB.pdf"
fig.savefig( outPath, bbox_inches = 'tight' )
print( "Saved to " + outPath )

# for i in range( 0, iterations ):
# 	#print( "Gathering Data...%i\r" %( i + 1 ), end = '' )

# 	file = filePath + str( i ) + ".dat"

# 	with open( file ) as f:
# 		# Get alpha-beta 1
# 		inputAB = next( f ).split()
# 		print( "1: %s %s" % ( inputAB[ 1 ], inputAB[ 2 ] ) )
# 		alpha1 = float( inputAB[ 1 ] )
# 		beta1 = float( inputAB[ 2 ] )

# 		# Get alpha-beta 2
# 		inputAB = next( f ).split()
# 		print( "2: %s %s\n" % ( inputAB[ 1 ], inputAB[ 2 ] ) )
# 		alpha2 = float( inputAB[ 1 ] )
# 		beta2 = float( inputAB[ 2 ] )

# 		next( f )
# 		next( f )
# 		inputMu = next( f ).split()

# 		if inputMu[ 0 ] > inputMu[ 1 ]:
# 			incABest = alpha1 / iterations
# 			incBBest = beta1 / iterations

# 			incAWorst = alpha2 / iterations
# 			incBWorst = beta2 / iterations
# 		else:
# 			incABest = alpha2 / iterations
# 			incBBest = beta2 / iterations

# 			incAWorst = alpha1 / iterations
# 			incBWorst = beta1 / iterations

# 		bestArmAlpha = bestArmAlpha + incABest
# 		bestArmBeta = bestArmBeta + incBBest

# 		worstArmAlpha = worstArmAlpha + incAWorst
# 		worstArmBeta = worstArmBeta + incBWorst




# print( "Best Arm:" )
# print( "    Alpha = %f" % ( bestArmAlpha ) )
# print( "    Beta = %f\n" % ( bestArmBeta ) )
# print( "Worst Arm:" )
# print( "    Alpha = %f" % ( worstArmAlpha ) )
# print( "    Beta = %f" % ( worstArmBeta ) )

