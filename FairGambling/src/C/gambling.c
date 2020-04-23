#include <gsl/gsl_randist.h>				// Used for random number distributions
#include <stdio.h>							// Used for input/output					
#include <time.h>							// Used to keep track of elapsed time

// Used only to check if directories exist
#include <dirent.h>
#include <errno.h>


// Define starting parameters of simulation
#define ALICE_STARTING 			5		// Starting money for Alice
#define BOB_STARTING			50		// Starting money for Bob
#define ALICE_WIN_PROBABILITY 	0.5		// Probability that Alice wins a draw
#define BET_AMOUNT				1 		// Amount of money each person bets at each round
#define NUM_GAMES 				1000000	// Number of games to play until a player loses


// Define simulation parameters
#define DELTA					1000


// Define color commands
#define RESET 					"\x1b[0m"
#define BOLD					"\x1b[1m"
#define F_RED					"\x1b[31m"
#define F_GREEN					"\x1b[32m"


// Define betting styles
#define CONSTANT				0	// Standard bet amount
#define DOUBLE 					1	// Double bet amount each round
#define MAXIMUM 				2	// Bet maximum amount
#define INCREASE_LOSE 			3	// Increase bet when losing, decrease after winning
#define RESET_LOSE 				4	// Incrementing bet amount, restart after loss
#define RANDOM 					5	// Random bet (uniform; 1 to max)


// Set betting style
#define BET_STYLE 				CONSTANT


// Define location to save result data
#define SAVE_DATA_LOC			"data/results.dat"


// Option to save the data. 1 = save, 0 = don't save
#define SAVE_DATA 				1

int main()
{
	if( SAVE_DATA )
		printf( "Saving data to '%s'\n\n", SAVE_DATA_LOC );

	// Display game parameters
	printf( "Game Parameters:\n"
			"    Alice Starting:        $%i\n"
			"    Bob Starting:          $%i\n"
			"    Alice Win Probability: %0.2f%%\n"
			"    Starting Bet Amount:   $%i\n"
			"    Number of Games:       %i\n\n", 
			ALICE_STARTING, BOB_STARTING, 100 * ALICE_WIN_PROBABILITY, BET_AMOUNT, NUM_GAMES );


	// Get seed for random value. Only updates once per second and is a PRNG
	srand( time( NULL ) );

	// Set up random number generator from GNU Science Library (GSL)
	const gsl_rng_type *T;
	gsl_rng *r;

	gsl_rng_env_setup();
	T = gsl_rng_default;
	r = gsl_rng_alloc( T );


	// Track amount of money Alice and Bob have
	int moneyA;
	int moneyB;

	// Stores the pool for each round
	int pool;

	// Stores the outcome of the random draw at each round
	double outcome;

	// Track average number of rounds it takes for one person to win
	double averageRounds = 0;

	// Track number of wins and losses for each person
	int aliceWin = 0;
	int bobWin = 0;

	// Track average money remaining
	FILE * fp;
	if( SAVE_DATA )
	{
		fp = fopen( SAVE_DATA_LOC, "w" );
		if( fp == NULL)
		{
			printf( "ERROR: Unable to open file '%s'.\nPlease make sure the directory exists and try again.\n", SAVE_DATA_LOC );
			exit( 0 );
		}
	}

	int i;
	for( i = 0; i < NUM_GAMES; i++ )
	{
		// Set up starting money for each player
		moneyA = ALICE_STARTING;
		moneyB = BOB_STARTING;

		// Count how many times it takes for one person to win
		int round = 0;

		// Keep track of previous round's result
		int aliceWonLastBet = 0;

		// Set betAmount
		int betAmount = BET_AMOUNT;

		// Keep track of previous round's betAmount
		int previousBetAmount;


		// Play game until one player has no money left
		while( moneyA != 0 && moneyB != 0 )
		{
			round++;


			// Set previous bet amount
			previousBetAmount = betAmount;


			// Calculate bet amount for both players
			if( BET_STYLE == CONSTANT )
			{
				betAmount = BET_AMOUNT;
			}
			else if( BET_STYLE == DOUBLE )
			{
				betAmount = ( aliceWonLastBet ? previousBetAmount * 2 : previousBetAmount );
			}
			else if( BET_STYLE == MAXIMUM )
			{
				betAmount = ( moneyA < moneyB ? moneyA : moneyB );
			}
			else if( BET_STYLE == INCREASE_LOSE )
			{
				betAmount = ( aliceWonLastBet ? ( previousBetAmount > 1 ? previousBetAmount - 1 : 1 ) : previousBetAmount + 1 );
			}
			else if( BET_STYLE == RESET_LOSE )
			{
				betAmount = ( aliceWonLastBet ? previousBetAmount + 1 : BET_AMOUNT );
			}
			else // BET_STYLE == RANDOM
			{
				betAmount = (int)gsl_ran_flat( r, 1, moneyA < moneyB ? moneyA : moneyB );
			}


			// Make sure bet amount isn't more than lowest player's bankroll
			if( moneyA < betAmount || moneyB < betAmount )
			{
				betAmount = ( moneyA < moneyB ? moneyA : moneyB );
			}


			// Take money from players and put it in the pool
			moneyA = moneyA - betAmount;
			moneyB = moneyB - betAmount;
			pool = 2 * betAmount;


			// Randomly draw a value from [0, 1)
			outcome = gsl_ran_flat( r, 0, 1);


			// Give money to the winner
			if( outcome <= ALICE_WIN_PROBABILITY )
			{
				// Alice Wins
				aliceWonLastBet = 1;
				moneyA = moneyA + pool;
			}
			else
			{
				// Bob wins
				aliceWonLastBet = 0;
				moneyB = moneyB + pool;
			}
		}

		// Increment the total number of rounds played so far
		averageRounds += round;

		// Increment the number of wins for the winning player
		if( moneyA )
			aliceWin++;
		else
			bobWin++;

		// Display current results. Only update every DELTA games to improve performance
		if( ( i + 1 ) % DELTA == 0 )
		{
			// Display current wins for each player
			printf( "Game: % 10i    Rounds: % 10i    %sAlice" RESET ": % 10i (% 8.4f%%)    %sBob" RESET ": % 10i( % 8.4f%%)         \r",
					i + 1,
					round,
					aliceWin < bobWin ? F_RED : F_GREEN, 
					aliceWin, 100 * (float) aliceWin / ( aliceWin + bobWin ),
					bobWin < aliceWin ? F_RED : F_GREEN,
					bobWin, 100 * (float) bobWin / ( aliceWin + bobWin) );
		}

		// Save current round information:  Game Number 	Number of Rounds		Alice Win Count		Bob Win Count
		if( SAVE_DATA )
			fprintf( fp, "%i %i %i %i\n", 		( i + 1 ), 		round, 					aliceWin, 			bobWin );
	}

	if( SAVE_DATA )
		fclose( fp );

	// Calculate average number of rounds by dividing sum by the number of games
	averageRounds /= NUM_GAMES;

	// Display the results of the simulation
	printf( "%110sAverage number of rounds needed: %.3f.\n"
			"Results:\n"
			"    " BOLD "%sAlice" RESET ":  %i (%.4f%%)\n"
			"    " BOLD "%sBob" RESET ":    %i (%.4f%%)\n",
			"\n", averageRounds, 
			aliceWin < bobWin ? F_RED : F_GREEN,
			aliceWin, 100 * (float) aliceWin / ( aliceWin + bobWin ), 
			bobWin < aliceWin ? F_RED : F_GREEN,
			bobWin, 100 * (float) bobWin / ( bobWin + aliceWin ) );

	return 0;
} 
