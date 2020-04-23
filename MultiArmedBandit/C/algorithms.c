#include <gsl/gsl_randist.h>
#include <stdlib.h>
#include <math.h>

#include "algorithms.h"

#define BERNOULLI( P ) ( (double) rand() / RAND_MAX < P ? 1 : 0 )

double bernoulliReward( double p )
{
	return (double) rand() / RAND_MAX < p ? 1 : 0;
}

Algorithm * newAlgorithm( int algType, int trials, int simulations, int k, int distType, int numAvg, int exploreLen, gsl_rng *r, double epsilon )
{
	// Create new algorithm struct to return
	Algorithm * alg = malloc( sizeof( Algorithm ) );

	if( alg == NULL )
	{
		puts( "Error allocating alg." );
		return NULL;
	}

	// Initialize algorithm experimental parameters
	alg -> algType = algType;									// Type of algorithm
	alg -> trials = trials;										// Number of trials
	alg -> simulations = simulations;							// Number of simulations
	alg -> k = k;												// Number of arms
	alg -> distType = distType;									// Type of reward distribution
	alg -> r = r;												// GSL RNG pointer

	// Initialize arrays used during experimentation
	alg -> regret = (double*) calloc( trials, sizeof( double ) );	// Array for average regret at each time slot
	alg -> armChosenCount = (int*) calloc( k , sizeof( int ) );		// Count of how many times each arm has been drawn

	if( alg -> regret == NULL )
	{
		puts( "Error allocating regret." );
		return NULL;
	}

	if( alg -> armChosenCount == NULL )
	{
		puts( "Error allocating alg." );
		return NULL;
	}

	// Only Thompson uses armWinCount
	if( algType == TH_ALGTYPE )
	{
		alg -> armWinCount = (int*) calloc( k , sizeof( int ) );	// Count of how many times each arm has produced a win (Thompson Only)
		if( alg -> armWinCount == NULL )
		{
			puts( "Error allocating alg." );
			return NULL;
		}	

		alg -> numAvg = numAvg;

		alg -> alpha = (double**) calloc( k, sizeof( double ) );	// Array for average alpha values at each time slot
		alg -> beta  = (double**) calloc( k, sizeof( double ) );	// Array for average beta values at each time slot

		// Initialize alpha and beta saving arrays
		int i;
		for( i = 0; i < k; i++ )
		{
			( alg -> alpha )[ i ] = (double*) calloc( trials, sizeof( double ) );
			( alg -> beta )[ i ] = (double*) calloc( trials, sizeof( double ) );
		}
	}

	// Only ε-Greedy uses Empirical Mean saving
	if( algType == E_ALGTYPE )
	{
		alg -> empiricalMeanData = (double**) calloc( k, sizeof( double ) );	// Array for bound values for each arm at each time slot

		int i;
		for( i = 0; i < k; i++ )
		{
			( alg -> empiricalMeanData )[ i ] = (double*) calloc( trials, sizeof( double ) );
		}
	}

	// Only UCB1 uses Bound saving
	if( algType == UCB1_ALGTYPE )
	{
		alg -> bounds = (double**) calloc( k, sizeof( double ) );	// Array for bound values for each arm at each time slot

		int i;
		for( i = 0; i < k; i++ )
		{
			( alg -> bounds )[ i ] = (double*) calloc( trials, sizeof( double ) );
		}
	}

	// UCB, UCB1, and ε-Greedy use empiricalMeans
	if( algType == UCB_ALGTYPE || algType == UCB1_ALGTYPE || algType == E_ALGTYPE )
	{
		alg -> empiricalMeans = calloc( k, sizeof( double ) );	// Average reward from each arm (UCB, UCB1, and ε-Greedy only)
		if( alg -> empiricalMeans == NULL )
		{
			puts( "Error allocating alg." );
			return NULL;
		}		
	}

	if( algType == E_ALGTYPE )
	{
		alg -> epsilon = epsilon;
		alg -> exploreLen = exploreLen;
	}

	// Return the initialized algorithm
	return alg;
}

void chooseArm( Algorithm *alg, int t )
{
	switch( alg -> algType )
	{
		// UCB
		case UCB_ALGTYPE:
		{
			if( t < alg -> k )
			{
				alg -> chosenArm = t;
			}
			else
			{
				double bound, maxVal;
				int i;
				for( i = 0; i < alg -> k; i++ )
				{
					bound = alg -> empiricalMeans[ i ]
									+ sqrt( 2 * logf( (double) t + 1 ) / alg -> armChosenCount[ i ] );

					if( i == 0 || bound > maxVal )
					{
						maxVal = bound;
						alg -> chosenArm = i;
					}
				}
			}

			break;
		}

		// UCB1
		case UCB1_ALGTYPE:
		{
			if( t < alg -> k )
			{
				alg -> chosenArm = t;
			}
			else
			{
				double bound, maxVal;
				int i;
				for( i = 0; i < alg -> k; i++ )
				{
					bound = alg -> empiricalMeans[ i ]
									+ sqrt( 2 * alg -> empiricalMeans[ i ] * logf( (double) 1.0 / sqrt( (double) 1.0 / ( t + 1 ) ) ) / alg -> armChosenCount[ i ] )
									+ ( 2 * logf( (double) 1.0 / sqrt( (double) 1.0 / ( t + 1 ) ) ) ) / alg -> armChosenCount[ i ];

					if( i == 0 || bound > maxVal )
					{
						maxVal = bound;
						alg -> chosenArm = i;
					}

					// Update Bounds parameters for each arm at current time slot
					alg -> bounds[ i ][ t ] += bound / alg -> simulations;
				}
			}

			break;
		}

		// Thompson
		case TH_ALGTYPE: 
		{
			double sum;
			double maxVal = gsl_ran_beta( alg -> r, 1 + alg -> armWinCount[ 0 ], 1 + alg -> armChosenCount[ 0 ] - alg -> armWinCount[ 0 ] );
			alg -> chosenArm = 0;

			int i, j;
			for( i = 1; i < alg -> k; i++ )
			{
				sum = 0;
				for( j = 0; j < alg -> numAvg; j++ )
					sum += gsl_ran_beta( alg -> r, 1 + alg -> armWinCount[ i ], 1 + alg -> armChosenCount[ i ] - alg -> armWinCount[ i ] );
				
				sum /= alg -> numAvg;

				if( sum > maxVal )
				{
					maxVal = sum;
					alg -> chosenArm = i;
				}
			}

			break;
		}

		// ε-Greedy
		case E_ALGTYPE:
		{
			if( t < alg -> exploreLen )
			{
				alg -> chosenArm = (int)gsl_ran_flat( alg -> r, 0, alg -> k );
			}
			else if( gsl_ran_flat( alg -> r, 0, 1 ) < alg -> epsilon )
			{
				alg -> chosenArm = (int)gsl_ran_flat( alg -> r, 0, alg -> k );		// gsl_ran_flat gives random uniform value [a, b)
			}
			else
			{
				double maxVal;
				int i;

				for( i = 0; i < alg -> k; i++ )
				{
					if( i == 0 || alg -> empiricalMeans[ i ] > maxVal )
					{
						maxVal = alg -> empiricalMeans[ i ];
						alg -> chosenArm = i;
					}
				}
			}

			break;
		}
	}
}

void drawReward( Algorithm *alg, double u, double sigma, int t )
{
	if( alg -> distType == BERNOULLI_DIST )
		 alg -> reward = bernoulliReward( u );
	else if( alg -> distType == NORMAL_DIST )
		alg -> reward =  gsl_ran_gaussian( alg -> r, sigma ) + u;
}

void update( Algorithm *alg, int t, double u, double bestMean )
{
	switch( alg -> algType )
	{
		// UCB
		case 0:
		{
			alg -> armChosenCount[ alg -> chosenArm ]++;
			alg -> empiricalMeans[ alg -> chosenArm ] = alg -> empiricalMeans[ alg -> chosenArm ] * ( alg -> armChosenCount[ alg -> chosenArm ] - 1 ) / alg -> armChosenCount[ alg -> chosenArm ]
													+ alg -> reward / alg -> armChosenCount[ alg -> chosenArm ];

			alg -> expectedRewardSum += u;
			alg -> regret[ t ] += ( ( t + 1 ) * bestMean - alg -> expectedRewardSum ) / alg -> simulations;
			break;
		}

		// UCB1
		case 1:
		{
			alg -> armChosenCount[ alg -> chosenArm ]++;
			alg -> empiricalMeans[ alg -> chosenArm ] = alg -> empiricalMeans[ alg -> chosenArm ] * ( alg -> armChosenCount[ alg -> chosenArm ] - 1 ) / alg -> armChosenCount[ alg -> chosenArm ]
													+ alg -> reward / alg -> armChosenCount[ alg -> chosenArm ];

			alg -> expectedRewardSum += u;
			alg -> regret[ t ] += ( ( t + 1 ) * bestMean - alg -> expectedRewardSum ) / alg -> simulations;
			break;
		}

		// Thompson
		case 2:
		{
			alg -> armChosenCount[ alg -> chosenArm ]++;
			alg -> expectedRewardSum += u;
			alg -> regret[ t ] += ( ( t + 1 ) * bestMean - alg -> expectedRewardSum ) / alg -> simulations;

			if( alg -> distType == BERNOULLI_DIST )
			{
				alg -> armWinCount[ alg -> chosenArm ] += alg -> reward;
			}
			else if( alg -> distType == NORMAL_DIST )
			{
				alg -> armWinCount[ alg -> chosenArm ] += BERNOULLI( u );
			}

			// Update alpha and beta parameters for each arm at current time slot
			int i;
			for( i = 0; i < alg -> k; i++ )
			{

				alg -> alpha[ i ][ t ] += (float)( 1 + alg -> armWinCount[ i ] ) / alg -> simulations;
				alg -> beta[ i ][ t ]  += (float)( 1 + alg -> armChosenCount[ i ] - alg -> armWinCount[ i ] ) / alg -> simulations;
			}

			break;
		}

		// ε-Greedy
		case 3:
		{
			alg -> armChosenCount[ alg -> chosenArm ]++;
			alg -> empiricalMeans[ alg -> chosenArm ] = alg -> empiricalMeans[ alg -> chosenArm ] * ( alg -> armChosenCount[ alg -> chosenArm ] - 1 ) / alg -> armChosenCount[ alg -> chosenArm ]
													+ alg -> reward / alg -> armChosenCount[ alg -> chosenArm ];

			int i;
			for( i = 0; i < alg -> k; i++ )
					alg -> empiricalMeanData[ i ][ t ] += alg -> empiricalMeans[ i  ] / alg -> simulations;

			alg -> expectedRewardSum += u;
			alg -> regret[ t ] += ( ( t + 1 ) * bestMean - alg -> expectedRewardSum ) / alg -> simulations;
			break;
		}
	}
}

void reset( Algorithm *alg )
{
	int i;
	for( i = 0; i < alg -> k; i++ )
	{
		alg -> armChosenCount[ i ] = 0;

		if( alg -> algType == TH_ALGTYPE )
			alg -> armWinCount[ i ] = 0;

		if( alg -> algType == UCB_ALGTYPE || alg -> algType == UCB1_ALGTYPE || alg -> algType == E_ALGTYPE  )
			alg -> empiricalMeans[ i ] = 0;
	}

	alg -> expectedRewardSum = 0;
	alg -> numAvg = 1;
}

void freeAll( Algorithm *alg )
{
	int type = alg -> algType;

	if( type == UCB_ALGTYPE  || type == UCB1_ALGTYPE || type== E_ALGTYPE )
		free( alg -> empiricalMeans );

	if( type == TH_ALGTYPE )
		free( alg -> armWinCount );

	free( alg -> armChosenCount );
	free( alg -> regret );

	free( alg );
}