#include <gsl/gsl_randist.h>				// Used for random number distributions
#include <stdio.h>							// Used for input/output					
#include <time.h>							// Used to keep track of elapsed time

// Used only to check if directories exist
#include <dirent.h>
#include <errno.h>

#include "algorithms.h"						// Contains algorithm implementations

// Preprocessor Concatenation
#define STR_HELPER( x ) #x					// '#' replaces the argument with the literal text of the argument
#define STR( x ) STR_HELPER( x )

// Experiment Parameters
#define EXPERIMENTS 		1				// Number of experiments to run
#define EXPERIMENTOFFSET 	0				// What iteration to start from. Final iteration is ITERATIONS + ITERATIONOFFSET
#define SIMULATIONS 		1				// Number of simulations to average total regret over
#define TRIALS 				1000000			// Number of time slots per simulation
#define K 					2				// Number of arms
#define NUM_ALGS			1				// Number of algorithms to use

// Distribution Parmaters
#define DIST_TYPE 			BERNOULLI_DIST	// Distribution type. Values defined in algorithms.h
#define BEST_ARM 			0.5 			// Reward mean of the best arm
#define EPSILON_MEAN 		0.10			// Difference between best and all other arms.
#define SIGMA 				0.1				// Standard deviation of normal distribution.

// Data Saving Parameters
#define DELTA				100				// How often to update and/or save data

// Thompson Sampling Parameters
#define NUM_AVG				1				// Number of beta distribution samples to take for Thompson Sampling
#define AVG_INC_TIME		50000			// Time slot where the average sample increases

// ε-Greedy Parameters
#define EPSILON 			0.01			// Probability of random exploration versus exploitation
#define EXPLORE_LENGTH		18000

// UCB/UCB-1 Parameters
/* None */

// Folder Location. Final forward slash not necessary
#define DATA_DIR			"Data/save_alpha_beta_test/ucb" 	// Save data in this directory for all arm regrets
#define AB_DIR				"Data/save_alpha_beta_test/ts"		// Save data in this directory for Alpha-Beta values for Thompson Sampling
#define BOUNDS_DIR			"Data/save_alpha_beta_test/ucb"		// Save data in this directory for bounds for UCB1
#define E_DIR				"Data/save_alpha_beta_test/e"		// Save data in this directory for empirical means data for ε-Greedy

// Function definitions
int dirExists( char * );									// Check if directory exists
void printAlgs( Algorithm ** );								// Print details for each algorithm
void saveData( Algorithm **, int );							// Save data to file
void saveAlphaBetaData( Algorithm **, double[], int );		// Save alpha-beta data for arms in Thompson Sampling algorithms to file
void saveBoundsData( Algorithm **, double[], int );			// Save bounds data for arms in UCB1 algorithms to file
void saveEmpiricalMeanData( Algorithm **, double[], int );	// Save empirical mean data for arms in ε-Greedy algorithms to file

int main()
{
	// Check if data directory exists
	if( !dirExists( DATA_DIR ) )
	{
		printf( "Unable to detect '%s'. Please check that it exists.\n", DATA_DIR );
		exit( EXIT_FAILURE );
	}

	printf( "Data saving to: '%s/'\n", DATA_DIR );


	// Get seed for random value. Only updates once per second and is a PRNG
	srand( time( NULL ) );

	// Set up random number generator from GNU Science Library (GSL)
	const gsl_rng_type *T;
	gsl_rng *r;

	gsl_rng_env_setup();
	T = gsl_rng_default;
	r = gsl_rng_alloc( T );


	// Create algorithm instances
	Algorithm * algList[ NUM_ALGS ];

	// Thompson Sampling
	// algList[ 0 ] = newAlgorithm( TH_ALGTYPE, TRIALS, SIMULATIONS, K, DIST_TYPE, NUM_AVG, 0, r, 0 );
	
	// Upper Confidence Bound
	algList[ 0 ] = newAlgorithm( UCB1_ALGTYPE, TRIALS, SIMULATIONS, K, DIST_TYPE, 0, 0, r, 0 );
	
	// ε-Greedy
	// algList[ 0 ] = newAlgorithm( E_ALGTYPE,	TRIALS, SIMULATIONS, K, DIST_TYPE, 0, EXPLORE_LENGTH, r, EPSILON );
	// algList[ 0 ] = newAlgorithm( E_ALGTYPE,	TRIALS, SIMULATIONS, K, DIST_TYPE, 0, 10, r, EPSILON );
	// algList[ 1 ] = newAlgorithm( E_ALGTYPE,	TRIALS, SIMULATIONS, K, DIST_TYPE, 0, 1000, r, EPSILON );
	// algList[ 0 ] = newAlgorithm( E_ALGTYPE,	TRIALS, SIMULATIONS, K, DIST_TYPE, 0, 10000, r, EPSILON );
	// algList[ 3 ] = newAlgorithm( E_ALGTYPE,	TRIALS, SIMULATIONS, K, DIST_TYPE, 0, 18000, r, EPSILON );
	// algList[ 4 ] = newAlgorithm( E_ALGTYPE,	TRIALS, SIMULATIONS, K, DIST_TYPE, 0, 100000, r, EPSILON );


	int a;
	for( a = 0; a < NUM_ALGS; a++ )
	{
		if( algList[ a ] == NULL )
		{
			puts( "Error allocating." );
			exit( EXIT_FAILURE );
		}

		// If Thompson Sampling, check if Alpha/Beta directory exists
		if( algList[ a ] -> algType == TH_ALGTYPE )
		{
			if( !dirExists( AB_DIR ) )	// Directory does not exist
			{
				printf( "Unable to save to folder '%s'. Please check that it exists.\n", AB_DIR );
				exit( EXIT_FAILURE );
			}
			
			printf( "Alpha/Beta saving to: '%s/'\n\n", AB_DIR );
		}	

		// If UCB1, check if bounds directory exists
		if( algList[ a ] -> algType == UCB1_ALGTYPE )
		{
			if( !dirExists( BOUNDS_DIR ) )	// Directory does not exist
			{
				printf( "Unable to save to folder '%s'. Please check that it exists.\n", AB_DIR );
				exit( EXIT_FAILURE );
			}
			
			printf( "UCB1 Bounds saving to: '%s/'\n\n", AB_DIR );
		}			
	}

	printAlgs( algList );

	printf( "Experiments %i through %i of %i simulations with %i trials\n"
			"    and %i arms. NUM_AVG: %i. EPSILON_MEAN: %0.2f\n"
			"--------------------------------------------------------\n",
			EXPERIMENTOFFSET + 1, EXPERIMENTS + EXPERIMENTOFFSET, SIMULATIONS, TRIALS, K, NUM_AVG, EPSILON_MEAN );

	// Run EXPERIMENTS experiments
	int v;
	for( v = EXPERIMENTOFFSET; v < EXPERIMENTS + EXPERIMENTOFFSET; v++ )
	{
		printf( "Experiment %i\n", v + 1 );

		int bestArm = rand() % K;	// Randomly choose arm with highest mean
		int i, j;					// Create variables for iteration
		double u[ K ];				// Store means for arms

		// Initialize means
		for( i = 0; i < K; i++ )
			u[ i ] = ( i == bestArm ) ? ( BEST_ARM ) : ( BEST_ARM - EPSILON_MEAN );
		double bestMean = u[ bestArm ];

		clock_t start = clock();

		// Loop through each simulation
		for( i = 0; i < SIMULATIONS; i++ )
		{
			for( a = 0; a < NUM_ALGS; a++ )
				reset( algList[ a ] );

			printf( "Simulation %03i   \r", i + 1 );
			fflush( stdout );

			for( j = 0; j < TRIALS; j++ )
			{
				// Simulate trial for each algorithm
				for( a = 0; a < NUM_ALGS; a++ )
				{
					chooseArm( algList[ a ], j );
					drawReward( algList[ a ], u[ algList[ a ] -> chosenArm ], SIGMA, j );
					update( algList[ a ], j, u[ algList[ a ] -> chosenArm ], bestMean );
				}
			}
		}

		// Compute execution time for this experiment
		clock_t end = clock();
		float totalTime = ( (double)( end - start ) ) / CLOCKS_PER_SEC;
		int hours = totalTime / 3600.0;
		int minutes =( ( totalTime / 3600.0 ) - hours ) * 60.0;
		double seconds = totalTime - ( hours * 3600.0 ) - ( minutes * 60.0 );

		// Display final regret for each algorithm
		for( a = 0; a < NUM_ALGS; a++ )
			printf( "%i) %-25s: %0.3f\n", 
				a + 1,
				algList[ a ] -> algType == TH_ALGTYPE ? "Thompson Sampling" : algList[ a ] -> algType == UCB_ALGTYPE ? "Upper Confidence Bound" : algList[ a ] -> algType == UCB1_ALGTYPE ? "Upper Confidence Bound 1" : "ε-Greedy",
				algList[ a ] -> regret[ TRIALS - 1 ] );

		// Display time to complete
		printf( "%02i:%02i:%02.2f\n\n", hours, minutes, seconds );

		// Save average regret over time for this experiment
		saveData( algList, v );

		// Save alpha-beta information for each arm for Thompson Sampling algorithms
		saveAlphaBetaData( algList, u, v );

		// Save bounds data information for each arm for UCB1 algorithms
		saveBoundsData( algList, u, v );

		// Save empirical mean data information for each arm for ε-Greedy algorithms
		saveEmpiricalMeanData( algList, u, v );


		// Free all used memory
		for( a = 0; a < NUM_ALGS; a++ )
			freeAll( algList[ a ] );
	}

	// Free pointers used in GNU Science Library
	gsl_rng_free( r );

	return 0;
}

int dirExists( char * directory )
{
	DIR * dir = opendir( directory );

	if( dir )	// Directory exists
	{
		closedir( dir );
		return 1;
	}
	else 		// Directory does not exist
	{
		return 0;
	}
}

void saveData( Algorithm ** algList, int experiment )
{
	// Save average regret over time for this experiment
	char * filename = malloc( 100 * sizeof( char ) );

	// Save based on Epsilon Mean
	sprintf( filename, "%s/%i.dat", DATA_DIR, experiment );

	// Save based on K
	// sprintf( filename, "%s%i/%i.dat", DATA_DIR, K, experiment  );


	// Open file
	FILE * fp = fopen( filename, "w" );

	// Check if file was unable to open
	if( fp == NULL )
	{
		printf( "Error opening '%s'. Check that it exists.\n", filename );
		exit( EXIT_FAILURE );
	}


	// Write data to file
	int i, a;
	for( i = 0; i < TRIALS; i++ )
	{
		if( ( i % DELTA ) == 0 )
		{
			fprintf( fp, "%d", i + 1 );	// Time slot

			for( a = 0; a < NUM_ALGS; a++ )
				fprintf( fp, " %f", algList[ a ] -> regret[ i ] );	// Regret value for each algorithm
			fprintf( fp, "\n" );
		}
	}


	// Free memory
	fclose( fp );
	free( filename );
}

void saveAlphaBetaData( Algorithm ** algList, double armMeans[ K ], int experiment )
{
	char * filename = malloc( 100 * sizeof( char ) );

	// Create file pointer
	FILE * fp;

	// Loop through each algorithm and check if it is Thompson Sampling
	int a;
	for( a = 0; a < NUM_ALGS; a++ )
	{

		// If so, save the alpha beta values for each arm at each time slot
		if( algList[ a ] -> algType == TH_ALGTYPE )
		{
			// Open Alpha-Beta file
			sprintf( filename, "%s/%iAB.dat", AB_DIR, experiment );
			fp = fopen( filename, "w" );


			// Check if Alpha-Beta file is successfully opened
			if( fp == NULL )
			{
				printf( "Error opening '%s'. Check that it exists.\n", filename );
				exit( -1 );
			}


			// Save list of arm means
			fprintf( fp, "Arm Means:\n" );
			int i;
			for( i = 0; i < K; i++ )
			{
				fprintf( fp, "%f ", armMeans[ i ] );
			}

			for( i = 0; i < TRIALS; i++ )
			{
				if( ( i % DELTA ) == 0 )
				{
					// Print trial number
					fprintf( fp, "\n%6d ", i + 1 );

					// Print alpha-beta values for each arm
					int k;
					for( k = 0; k < K; k++ )
						fprintf( fp, "%7lf %7lf ", algList[ a ] -> alpha[ k ][ i ], algList[ a ] -> beta[ k ][ i ] );
				}
			}

			// Close file
			fclose( fp );
		}
	}
}

void saveBoundsData( Algorithm ** algList, double armMeans[ K ], int experiment )
{
	char * filename = malloc( 100 * sizeof( char ) );

	// Create file pointer
	FILE * fp;

	// Loop through each algorithm and check if it is UCB1
	int a;
	for( a = 0; a < NUM_ALGS; a++ )
	{

		// If so, save the bound values for each arm at each time slot
		if( algList[ a ] -> algType == UCB1_ALGTYPE )
		{
			// Open Alpha-Beta file
			sprintf( filename, "%s/%i_bounds.dat", BOUNDS_DIR, experiment );
			fp = fopen( filename, "w" );


			// Check if Alpha-Beta file is successfully opened
			if( fp == NULL )
			{
				printf( "Error opening '%s'. Check that it exists.\n", filename );
				exit( -1 );
			}


			// Save list of arm means
			fprintf( fp, "Arm Means:\n" );
			int i;
			for( i = 0; i < K; i++ )
			{
				fprintf( fp, "%f ", armMeans[ i ] );
			}

			for( i = 0; i < TRIALS; i++ )
			{
				if( ( i % DELTA ) == 0 )
				{
					// Print trial number
					fprintf( fp, "\n%6d ", i + 1 );

					// Print alpha-beta values for each arm
					int k;
					for( k = 0; k < K; k++ )
						fprintf( fp, "%7lf ", algList[ a ] -> bounds[ k ][ i ] );
				}
			}

			// Close file
			fclose( fp );
		}
	}
}

void saveEmpiricalMeanData( Algorithm ** algList, double armMeans[ K ], int experiment )
{
	char * filename = malloc( 100 * sizeof( char ) );

	// Create file pointer
	FILE * fp;

	// Loop through each algorithm and check if it is UCB1
	int a;
	for( a = 0; a < NUM_ALGS; a++ )
	{

		// If so, save the bound values for each arm at each time slot
		if( algList[ a ] -> algType == E_ALGTYPE )
		{
			// Open Alpha-Beta file
			sprintf( filename, "%s/%i_empirical.dat", E_DIR, experiment );
			fp = fopen( filename, "w" );


			// Check if Alpha-Beta file is successfully opened
			if( fp == NULL )
			{
				printf( "Error opening '%s'. Check that it exists.\n", filename );
				exit( -1 );
			}


			// Save list of arm means
			fprintf( fp, "Arm Means:\n" );
			int i;
			for( i = 0; i < K; i++ )
			{
				fprintf( fp, "%f ", armMeans[ i ] );
			}

			for( i = 0; i < TRIALS; i++ )
			{
				if( ( i % DELTA ) == 0 )
				{
					// Print trial number
					fprintf( fp, "\n%6d ", i + 1 );

					// Print alpha-beta values for each arm
					int k;
					for( k = 0; k < K; k++ )
						fprintf( fp, "%7lf ", algList[ a ] -> empiricalMeanData[ k ][ i ] );
				}
			}

			// Close file
			fclose( fp );
		}
	}
}

void printAlgs( Algorithm ** algList )
{
	int a;
	for( a = 0; a < NUM_ALGS; a++ )
	{
		printf( "Algorithm %i: \n  ", a + 1 );

		if( algList[ a ] -> algType == UCB_ALGTYPE )
		{
			printf( "Upper Confidence Bound\n\n" );
		}
		else if( algList[ a ] -> algType == UCB1_ALGTYPE )
		{
			printf( "Upper Confidence Bound 1\n\n" );
		}
		else if( algList[ a ] -> algType == TH_ALGTYPE )
		{
			printf( "Thompson Sampling\n"
					"    Samples: %i\n\n", algList[ a ] -> numAvg );
		}
		else if( algList[ a ] -> algType == E_ALGTYPE )
		{
			printf( "ε-Greedy\n"
					"    Explore Length: %i\n"
					"    ε: %0.2f\n\n", 
						algList[ a ] -> exploreLen, algList[ a ] -> epsilon );
		}
		else
		{
			printf( "Unknown Algorithm Type\n\n" );
		}
	}
}
