#ifndef EPSILONGREEDY_H
#define EPSILONGREEDY_H

#define UCB_ALGTYPE  	0
#define UCB1_ALGTYPE 	1
#define TH_ALGTYPE 		2
#define E_ALGTYPE 		3

#define BERNOULLI_DIST 	0
#define NORMAL_DIST 	1

struct Algorithm;

/* 		int algType: 				Algorithm type. Possible values:
 *										0 - Upper Confidence Bound
 *										1 - Upper Confidence Bound 1
 *										2 - Thompon Sampling
 *										3 - ε-Greedy
 *
 *		int trials:					Number of trials in each experiment.
 *									Set during 'initialize' call.
 *
 * 		int experiments: 			Number of experiments to average over.
 *									Set during 'initialize' call.
 *
 *		int k:						Number of arms in simulatino.
 *									Set during 'initialize' call.
 *
 *		int distType;				Type of reward distribution. Possible value:
 *										BERNOULLI_DIST 	- Bernoulli Distribution
 *										NORMAL_DIST 	- Normal Distribution
 *
 *		int numAvg;					Number of beta distribution samples to take and
 *									average over when choosing arm. Only applies to
 *									Thompson Sampling. Value is >= 1.
 *
 *		int chosenArm;				Currently selected arm to draw from.
 *
 *		int exploreLen;				Number of time slots for pure exploration. Only
 *									applies to ε-Greedy.
 *
 *		gsl_rng *r;					GNU Science Library RNG pointer.
 *									Used to generate rewards and beta 
 *									random values for Thompson.
 *
 *		int *armChosenCount: 		List of number of times each arm has
 *									been chosen. Updated in 'update' function.
 *
 *		int *armWinCount:	 		List of number of times each arm has
 *									been chosen and returned a reward of 1
 *									in Bernoulli distribution. Updated in 
 *									'update' function.
 *
 *		double *regret: 			List of average regret values at each 
 *									time slot. Must be allocated and freed 
 *									in implementation.
 *
 *		double **alpha: 			List of average alpha values at each time slot
 *									for each arm. Must be allocated and freed 
 *									in implementation.
 *
 *		double **bounds: 			List of average bounds values at each time slot
 *									for each arm. Must be allocated and freed 
 *									in implementation.
 *
 *		double **regret: 			List of average beta values at each time slot
 *									for each arm. Must be allocated and freed 
 *									in implementation.
 *
 *		double *empiricalMeans: 	Average reward for each arm. Updated in
 *									'update' function.
 *
 *		double reward: 				Reward at current timeslot. Updated in
 *									'drawReward' function.
 *
 *		double expectedRewardSum: 	Running tally of expected reward at current
 *								 	timeslot. Updated in 'update' function.
 *
 *		double epsilon:				Epsilon value used only in Epsilon-Greedy algorithm.
 *									Value should be between 0 and 1, inclusive.
*/
typedef struct Algorithm
{
	int algType, trials, simulations, k, distType, numAvg, chosenArm, exploreLen;

	gsl_rng * r;

	int * armChosenCount;
	int * armWinCount;
	double * regret;
	double ** alpha;
	double ** beta;
	double ** bounds;
	double ** empiricalMeanData;
	double * empiricalMeans;

	double reward;
	double expectedRewardSum;

	double epsilon;
} Algorithm;

double bernoulliReward( double p );

Algorithm * newAlgorithm( int algType, int trials, int simulations, int k, int distType, int numAvg, int exploreLen, gsl_rng *r, double epsilon );

void chooseArm( Algorithm *alg, int t );

void drawReward( Algorithm *alg, double u, double sigma, int t );

void update( Algorithm *alg, int t, double u, double bestMean );

void reset( Algorithm *alg );

void freeAll( Algorithm *alg );

#endif