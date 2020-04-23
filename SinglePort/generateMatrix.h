#ifndef GENERATEMATRIX_H
#define GENERATEMATRIX_H

int verifyMoves( int l, int k, int t, int* moves, int* symbols, int* sent, int* received, int** rec, int** senders );

int hasSym( int T, int buffer, int sym, int** rec );

int isEmpty( int T, int buffer, int** rec );

void createMovesArray( int l, int k, int t, int* moves, int* currCount );

void displayReceivers( int T, int k, int** rec );

void displaySenders( int T, int k, int** senders );

void propagate( int l, int k, int t, int T, int phaseOneEnd, int* bufferPtr, int* currCount, int* count, int* moves, int* received, int* symbols, int** rec, int** senders );

#endif