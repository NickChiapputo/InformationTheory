#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "generateMatrix.h"

int verifyMoves( int l, int k, int t, int* moves, int* symbols, int* sent, int* received, int** rec, int** senders )
{
	int i;
	int vBuffer = 0;
	int vBufferRec = 0;
	int sym;
	if( t >= l )
	{
		sym = symbols[ l - 1 ];
		// Get a receiver who received symbol l this time slot
		while( rec[ vBufferRec ][ t ] != sym )
			vBufferRec = ( vBufferRec + 1 ) % k;
		//printf( "0 -> %i (%i)\n", vBufferRec + 1, sym );

		senders[ vBufferRec ][ t ] = 0;
		received[ vBufferRec ] = 1;
	}

	for( i = l - 1; i >= 0; i-- )
	{
		int currMoves = moves[ i ];	// Number of propagations for symbol i
		if( i == l - 1 ) 
			currMoves--;
		sym = symbols[ i ];

		// Loop through each propagation
		for( currMoves; currMoves > 0; currMoves-- )
		{
			vBufferRec = 0;
			vBuffer = 0;
			// Get a receiver who received current symbol
			while( rec[ vBufferRec ][ t ] != sym || received[ vBufferRec ] == 1 )
			{
				vBufferRec = ( vBufferRec + 1 ) % k;
			}


			// Get receiver who can send current symbol
				// Repeat if vBuffer has already sent a symbol, doesn't have the current symbol, or is the same as vBufferRec
			int loops = 0;
			while( sent[ vBuffer ] == 1 || ( hasSym( t, vBuffer, sym, rec ) == 0 || vBuffer == vBufferRec ) )
			{
				if( loops++ > k )
					return 0;
				//printf( "Node %i - \nsent = %i\nhasSym = %i\nEqual to vBufferRec = %i\n\n", vBuffer + 1, sent[ vBuffer ], hasSym( t, vBuffer, sym, rec ), vBuffer == vBufferRec );
				vBuffer = ( vBuffer + 1 ) % k;
			}

			//printf( "%i -> %i (%i)\n", vBuffer + 1, vBufferRec + 1, sym );
			senders[ vBufferRec ][ t ] = vBuffer + 1;
			received[ vBufferRec ] = 1;
			sent[ vBuffer ] = 1;
		}
	}
	return 1;
}

int hasSym( int T, int buffer, int sym, int** rec )
{
	int i;
	for( i = 0; i < T; i++ )
	{
		if( rec[ buffer ][ i ] == sym )
			return 1;
	}
	return 0;
}	

int isEmpty( int T, int buffer, int** rec )
{
	int i;
	for( i = 0; i < T; i++ )
	{
		if( rec[ buffer ][ i ] != 0 )
			return 0;
	}
	return 1;
}

void createMovesArray( int l, int k, int t, int* moves, int* currCount )
{
	int i, currentMoves;
	for( i = 0; i < l; i++ )
	{
		currentMoves = currCount[ i ];
		/*
		*  If current count is less than or equal to half of k, then double it
		*  Otherwise, add moves to make currCount = k
		*/
		moves[ i ] = 2 * currentMoves <= k ? currentMoves : k - currentMoves;
		//printf( "Moves for %i: %i\n", i + 1, moves[ i ] );
	
}
	if( t >= l && currCount[ l - 1 ] + moves[ l - 1 ] != k )
		moves[ l - 1 ]++;
}

void displayReceivers( int T, int k, int** rec )
{
	int i;
	for( i = 0; i < k; i++ )
	{
		printf( "\n%3i - ", i + 1 );
		int j;
		for( j = 0; j < T; j++ )
		{
			if( rec[ i ][ j ] == 0 )
				printf( "  * " );
			else
				printf( "%3i ", rec[ i ][ j ] );
		}
	}
	puts("");
}
void displaySenders( int T, int k, int** senders )
{
	int i;
	for( i = 0; i < k; i++ )
	{
		printf( "\n%3i - ", i + 1 );
		int j;
		for( j = 0; j < T; j++ )
		{
			if( senders[ i ][ j ] == -1 )
				printf( "  * " );
			else
				printf( "%3i ", senders[ i ][ j ] );
		}
	}
}

/*
 *	Count currently not used
*/
void propagate( int l, int k, int t, int T, int phaseOneEnd, int* bufferPtr, int* currCount, int* moves, int* count, int* received, int* symbols, int** rec, int** senders )
{
	int i, usedCount;
	int buff = *bufferPtr;

	//printf( "%i == %i\n%i >= 3\n%i >= 9\n%i >= 4\n%i >= %i\n\nSpecial Case: %i", t, phaseOneEnd - 1, l, k, ( 1 << phaseOneEnd ) - k, k, ( 1 << ( phaseOneEnd - 1 ) ) + ( 1 << phaseOneEnd - 3 ) + 1, t == phaseOneEnd - 1 && l >= 3 && k >= 9 && ( 1 << phaseOneEnd ) - k >= 4 && k >= ( 1 << ( phaseOneEnd - 1 ) ) + ( 1 << phaseOneEnd - 3 ) + 1 );
	//puts("");
	if( t == phaseOneEnd - 1 && l >= 3 && k >= 11 )
	{
		//printf( "%i >= %i\n ", k, ( 1 << ( phaseOneEnd - 1 ) ) + phaseOneEnd - 1 );
		for( i = 0; i < l; i++ )
		{
			int index = 0;
			int sum = 0;
			int sym = symbols[ i ];
			int currMoves = moves[ i ];

			for( index; index < l; index++ )
				sum += currCount[ index ];
			if( sum >= k )
				buff = 0;
			//printf( "\nsym: %i\nsum: %i\nbuff: %ireceived: %i\nhasSym: %i\n\n", sym, sum, buff );

			//printf( "Moves for %i: %i\n", sym, currMoves );

			if( currCount[ i % l ] != k )
			{
				for( currMoves; currMoves > 0; currMoves-- )
				{
					/*		received[ buff ] == 1
					 *	current receiver has already receive a symbol this time slot
					 * 	
					 *		hasSym( T, buff, sym, rec ) == 1
					 *	current receiver already has symbol
					 *
					 *		( hasSym( T, buff, 1, rec ) != 1 && sym != 1 && sum >= k )
					 *	receiver hasn't received the first symbol before, the current symbol is not 1, and there have been more than k propagations
					 *
					 *		( isEmpty( T, buff, rec ) == 0 && sum < k )
					 *	Current receiver has no symbols and the total number of propagations is less than k
					*/
					int repeat = received[ buff ] == 1 || hasSym( T, buff, sym, rec ) == 1 || ( hasSym( T, buff, 1, rec ) != 1 && sym != 1 && sum >=  k ) || ( isEmpty( T, buff, rec ) == 0 && sum < k ) ? 1 : 0;
					//printf( "\nBuffer: %i\n%i || %i || %i || ( %i && ( %i && %i ) ) = %i\n", buff + 1, received[ buff ], hasSym( T, buff, sym, rec ), ( isEmpty( T, buff, rec ) == 0 && sum < k ), hasSym( T, buff, 1, rec ) != 1, sym != 1, sym != 2, repeat );
					int loops = 0;
					while( repeat == 1 )
					{
						buff = ( buff + 1 ) % k;
						if( loops++ > k ) printf( "Propagation Error" );
						//if( loops++ <= k )	printf( "\nBuffer: %i\nsum: %i\n%i || %i || %i || ( %i && ( %i && %i ) ) = %i\n", buff + 1, sum, received[ buff ], hasSym( T, buff, sym, rec ), ( isEmpty( T, buff, rec ) == 0 && sum < k ), hasSym( T, buff, 1, rec ) != 1, sym != 1, sym != 2, repeat );
						repeat = received[ buff ] == 1 || hasSym( T, buff, sym, rec ) == 1 || ( hasSym( T, buff, 1, rec ) != 1 && sym != 1 && sum >=  k ) || ( isEmpty( T, buff, rec ) == 0 && sum < k ) ? 1 : 0;
					}

					//printf( "%i to %i\n", sym, buff + 1 );
					received[ buff ] = 1;		// Flag reciever as having received a symbol this time slot
					rec[ buff ][ t ] = sym;		// Assign symbol to receiver
					count[ buff ]++;			// Increase symbol count of reciever
					currCount[ i % l ]++;		// Increase count of current symbol
					buff = ( buff + 1 ) % k;	// Increment buffer
					sum++;
				}
			}
		}

		buff = 0;
		if( t < l )
		{
			int sum = 0;
			int sym = symbols[ t ];
			for( i = 0; i < l; i++ )
				sum += currCount[ i ];
			while( received[ buff ] == 1 || ( hasSym( T, buff, 1, rec ) != 1 && sum >= k && sym != 1 ) || ( isEmpty( T, buff, rec ) == 0 && sum < k ) )
				buff = ( buff + 1 ) % k;

			// Assign initial value
			//printf( "\n\nSrc sends %i to %i\n", symbols[ t % l ], buff + 1 );
			senders[ buff ][ t ] = 0;				// Source node sends first of symbol
			rec[ buff ][ t ] = symbols[ t % l ];	// Receive symbol to receiver "buff"
			count[ buff ]++;						// Increase symbol count of reciever
			received[ buff ] = 1;					// Flag receiver as having received a symbol
			buff = ( buff + 1 ) % k;				// Increment buffer
			currCount[ t ]++;						// Increment count of symbol
		}
	}
	else
	{
		for( i = 0; i < l ; i++ )
		{
			int index = 0;
			int sum = 0;
			int sym = symbols[ i ];
			int currMoves = moves[ i ];
			//printf( "\nMoves for %i: %i\n", sym, currMoves );
			for( index; index < l; index++ )
				sum += currCount[ index ];
			if( sum >= k )						// If move than or equal to k symbols have been sent in total, reset the buffer at 0 for each symbol
				buff = 0;

			if( currCount[ i ] != k )
			{
				for( currMoves; currMoves > 0; currMoves-- )
				{
					int repeat = received[ buff ] == 1 || hasSym( T, buff, sym, rec ) == 1 ? 1 : 0;
					int loops = 0;
					while( repeat == 1 )
					{
						//if( loops++ < k ) printf( "buffer: %i\nreceived: %i\nhasSym: %i\n\n", buff + 1, received[ buff ], hasSym( T, buff, sym, rec ) );
						buff = ( buff + 1 ) % k;
						repeat = received[ buff ] == 1 || hasSym( T, buff, sym, rec ) == 1 ? 1 : 0;
					}

					//printf( "%i to %i\n", sym, buff + 1 );
					received[ buff ] = 1;		// Flag reciever as having received a symbol this time slot
					rec[ buff ][ t ] = sym;		// Assign symbol to receiver
					count[ buff ]++;			// Increase symbol count of reciever
					currCount[ i ]++;			// Increase count of current symbol
					buff = ( buff + 1 ) % k;	// Increment buffer
				}
			}
		}

		if( t < l )
		{
			while( received[ buff ] == 1 )
				buff = ( buff + 1 ) % k;

			// Assign initial value
			//printf( "\n\nSrc sends %i to %i\n", symbols[ t % l ], buff + 1 );
			senders[ buff ][ t ] = 0;				// Source node sends first of symbol
			rec[ buff ][ t ] = symbols[ t % l ];	// Receive symbol to receiver "buff"
			count[ buff ]++;						// Increase symbol count of reciever
			received[ buff ] = 1;					// Flag receiver as having received a symbol
			buff = ( buff + 1 ) % k;				// Increment buffer
			currCount[ t ]++;						// Increment count of symbol
		}
	}

	*bufferPtr = buff;
}