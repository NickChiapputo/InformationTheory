/* Author : Nicholas Chiapputo
 * E-mail : nicholaschiapputo@my.unt.edu
 *
 * This program seeks to generate a receiver and transmitter matrix in 
 * T = ceil(log_2(K+1))+L-1 time slots given K receivers and L symbols
 * such that each receiver receives all L symbols by time slot T.
 * This program generates a receiver matrix determining which receiver
 * receives which symbol at each time slot, with a * representing no symbol
 * received. Additionally, the program generates a transmitter matrix that 
 * pairs two nodes and determines which node sends a symbol to another, specific
 * node.
 *
*/

#include <math.h>
#include <time.h>
#include <stdio.h>
#include <errno.h>
#include <dirent.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sys/stat.h>
#include "generateMatrix.h"

#define LRANGE 50
#define KRANGE 100

typedef struct
{
	int minL, maxL, k;
	char display, saveResults;
}
work_t;

char s;
int max;
int solved = 0;
int work_pool_index = 0;
int work_size = 0;
int thread_count = 8;
work_t* work_pool = NULL;
pthread_mutex_t mtx_work_pool 	= PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t mtx_output 		= PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t mtx_solved 		= PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t mtx_fileOutput 	= PTHREAD_MUTEX_INITIALIZER;

void* worker_thread( void* );
int generate( int, int, char );

void start_threads( int start_l, int start_k, int max_l, int max_k, char display, char saveResults )
{
	int i; 
	int k = start_k ;
	max = ( max_k - start_k + 1 ) * ( max_l - start_l + 1 );
	work_size = ( max_k - start_k + 1 );
	pthread_mutex_init( &mtx_work_pool, NULL );
	pthread_mutex_init( &mtx_solved, NULL );
	pthread_mutex_init( &mtx_output, NULL );
	work_pool = malloc( sizeof( work_t ) * work_size );
	work_pool_index = work_size - 1;

	// fill work_pool with all metadata needed to run all generations
	for( i = work_size - 1; i >= 0; i-- )
	{
		work_pool[ i ].minL = start_l;
		work_pool[ i ].maxL = max_l;
		work_pool[ i ].k = k;
		work_pool[ i ].display = display;
		work_pool[ i ].saveResults = saveResults;

		if( k != max_k )
			k++;
	}

	// Create the thread array
	pthread_t* threads = malloc( sizeof( pthread_t ) * thread_count );

	// Spawn off all of the threads
	for( i = 0; i < thread_count; i++ )
	{
		if( pthread_create( threads + i, NULL, worker_thread, NULL ) )
		{
			perror( "Error creating thread: " );
			exit( EXIT_FAILURE );
		}
	}

	for( i = 0; i < thread_count; i++ )
	{
		if( pthread_join( threads[ i ], NULL ) )
		{
			perror( "Problem with pthread_join" );
		}
	}
}

void* worker_thread( void* _ )
{
	int l;
	const work_t* work = NULL;
	while( 1 )
		// Get next work item from pool
	{
		work = NULL;
		pthread_mutex_lock( &mtx_work_pool );
		if( work_pool_index >= 0 )
		{
			work = work_pool + work_pool_index;
			work_pool_index--;
		}
		pthread_mutex_unlock( &mtx_work_pool );
		if( work == NULL ) break;

		int totalSolved = 0;
		for( l = work -> minL; l <= work -> maxL; l++ )
			totalSolved += generate( l, work -> k, work -> saveResults );

		pthread_mutex_lock( &mtx_solved );
		solved += totalSolved;
		system( "clear" );
		printf( "%02.2f%%\n", ( (double) solved / (double) max ) * 100.0 );
		pthread_mutex_unlock( &mtx_solved );
	}
}

// Pre-Condition: n > 0
// Post-Condition: return value is ceiling of log_2(n);
int Log_2( int n )
{
	int a = 0;
	n--;
	while( n > 0 )
	{ 
		a++;		// Increment answer
		n >>= 1;	// Divide n by 2
	}
	return a;
}

void printToFile( int l, int k, int T, int** rec, int** senders )
{
	char* fileName = malloc( 48 );	// 25
	char* strL = fileName + 25;		// 12
	char* strK = strL + 12;			// 11

	int lowerK = ( ( l / KRANGE ) + 1 ) * KRANGE - KRANGE + 1;
	int lowerL = ( ( l / LRANGE ) + 1 ) * LRANGE - LRANGE + 1;


	sprintf( strL, "L%i-%i_", lowerL, lowerL + LRANGE - 2 );
	sprintf( strK, "K%i-%i", lowerK, lowerK + KRANGE - 2 );


	strcpy( fileName, "data/" );
	strcat( fileName, strL    );
	strcat( fileName, strK    );

	DIR* dir = opendir( "data" );
	if( dir )
	{
		closedir( dir );
	}
	else if( ENOENT == errno )
	{
		mkdir( "data", 0777 );	
	}
	else
	{
		puts( "Failed to open 'data' directory" );
		exit( EXIT_FAILURE );
	}

	FILE* fp = fopen( fileName, "a" );

	int i, j;
	int spaces = log10( l ) > log10( k ) ? log10( l ) : log10( k );
	fprintf( fp, "L: %i\nK: %i", l, k );
	for( i = 0; i < k; i++ )
	{
		if( i != 0 )
			fprintf( fp, "]\n[ " );
		else
			fprintf( fp, "\n[ " );

		for( j = 0; j < T; j++ )
		{
			if( rec[ i ][ j ] == 0 )
				fprintf( fp, "%*s%s(%*s%s) ", spaces, "", "*", spaces, "", "*" );
			else
				fprintf( fp, "%*i(%*i) ", spaces + 1, rec[ i ][ j ], spaces + 1, senders[ i ][ j ] );
		}
	}
	fprintf( fp, "]\n\n" );
	free( fileName );
	fclose( fp );
}

// Pres-Condition: l, k > 0
// Post-Condition: return value is 1 or 0, representing whether or not the matrix was able to be generated and verified
int generate( int l, int k, char f )
{
	int** rec;						// List of receivers and the symbols they received at each time slot
	int** senders;
	int* symbols;					// Symbols
	int* currCount;					// Used to hold the current count for each symbol
	int* count;						// Used to hold the number of symbols each receiver has
	int* moves;						// Used to determine the number of times a symbol will propagate in the current time slot
	int* received;					// Used to determine if a receiver has received a symbol in the current time slot
	int* sent;						// Used to determine if a receiver has sent     a symbol in the current time slot
	int i, j, t, phaseOneEnd;		// Used for iterations through loops
	int buffer = 0;
	int T = Log_2( k + 1 ) + l - 1;	// Calculate optimal propagation time


	// Instantiate lists
	symbols = malloc( sizeof( int ) * ( 3 * l + 3 * k )  );	// Size = l
	currCount = symbols + l;								// Size = l
	moves = currCount + l;									// Size = l
	received = moves + l;									// Size = k
	sent = received + k;									// Size = k
	count = sent + k;										// Size = k
	rec = malloc( sizeof( int* ) * 2 * k );	// Receivers
	senders = rec + k;
	for( i = 0; i < ( k > l ? k : l ); i++ )
	{
		if( i < l )
		{
			currCount[ i ] = 0;
			symbols[ i ] = i + 1;		
		}

		if( i < k )
		{
			count[ i ] = 0;
			rec[ i ] = malloc( sizeof( int ) * T * 2 );
			senders[ i ] = rec[ i ] + T;

			for( j = 0; j < T; j++ )
			{
				senders[ i ][ j ] = -1;
				rec[ i ][ j ] = 0;	// Default receivers with full empty symbols to avoid having to add later on
			}
		}
	}

	phaseOneEnd = Log_2( k );

	for( t = 0; t < T; t++ )
	{
		if( t >= T - l + 1 )
			buffer = 0;

		// Reset tallies
		for( j = 0; j < k; j++ )
		{
			received[ j ] = 0;
			sent[ j ] = 0;
		}	

		// Create moves array
		createMovesArray( l, k, t, moves, currCount );

		// Check if any more moves need to be made
		propagate( l, k, t, T, phaseOneEnd, &buffer, currCount, moves, count, received, symbols, rec, senders );

		
		// Reset received list for use in verification
		for( j = 0; j < k; j++ )
			received[ j ] = 0;

		// Verification
		if( verifyMoves( l, k, t, moves, symbols, sent, received, rec, senders ) == 0 )
		{
			return 0;
		}

	}


	// Display Receivers
	if( s == 'y' || s == 'Y' )
	{
		pthread_mutex_lock( &mtx_output );
		printf( "\n\nL = %i\nK = %i\n", l, k );
		displayReceivers( T, k, rec );
		puts("");

		printf( "\nL = %i\nK = %i\n", l, k );
		displaySenders( T, k, senders );
		pthread_mutex_unlock( &mtx_output );
	}

	if( f == 'y' || f == 'Y' )
	{
		pthread_mutex_lock( &mtx_fileOutput ); 
		printToFile( l, k, T, rec, senders );
		pthread_mutex_unlock( &mtx_fileOutput );
	}

	// Free pointers
	free( symbols );
	for( i = 0; i < k; i++ )
		free( rec[ i ] );
	free( rec );
	return 1;
}

int main()
{
	// Get inputs
	int l, k, maxL, maxK, start_l = 1, start_k = 1;
	char w, range, f;
	printf( ">> Generate range? (Y/N)\n>> " );
	scanf( " %c", &range );
	if( range == 'y' || range == 'Y')
	{

		printf( "\n>> Start L = " );
		scanf( "%i", &start_l );
		printf( ">> Start K = " );
		scanf( "%i", &start_k );
		
		printf( "\n>> max L = " );
		scanf( "%i", &maxL );
		printf( ">> max K = " );
		scanf( "%i", &maxK );
	}
	else
	{
		printf( "\n>> L = " );
		scanf( "%i", &start_l );
		printf( ">> K = " );
		scanf( "%i", &start_k );
		maxL = start_l;
		maxK = start_k;
	}

	int max = ( maxK - start_k + 1 ) * ( maxL - start_l + 1 );

	printf( "\nWork pool? (Y/N)\n>> " );
	scanf( " %c", &w );

	printf( "\n>> Show Matrix? (Y/N)\n>> " );
	scanf( " %c", &s );

	printf( "\n>> Save results? (Y/N)\n>> " );
	scanf( " %c", &f );

	// Start execution timing
	struct timespec begin, end;
	double elapsed;

	clock_gettime( CLOCK_MONOTONIC, &begin );

	if( f == 'y' || f == 'Y' )
		system( "find data | grep L | xargs rm -f" );

	system( "clear" );
	if( w == 'y' || w == 'Y' )
	{
		start_threads( start_l, start_k, maxL, maxK, s, f );
	}
	else
	{
		for( k = start_k; k <= maxK; k++ )
		{
			for( l = start_l; l <= maxL; l++ )
			{
				int temp = generate( l, k, f );
				solved += temp;
				if( temp == 0 )
					puts( "Error" );
				printf( "%02.2f%%\r", ( (double) solved / (double) max ) * 100.0 );
			}
		}
	}

	clock_gettime( CLOCK_MONOTONIC, &end );
	elapsed = ( end.tv_sec - begin.tv_sec );
	elapsed += ( end.tv_nsec - begin.tv_nsec ) / 1000000000.0;
	int generations = ( range == 'y' || range == 'Y' ) ? ( maxK - start_k + 1 ) * ( maxL - start_l + 1 ): 1;
	double percentage = ( (double) solved / (double) generations ) * 100.0;
	printf( "\n\nExecution Time: %f s\n\n%i of %i solved (%.2f%%)\n", elapsed, solved, generations, percentage );
	return 0;
}
