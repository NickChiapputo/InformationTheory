# Multi-Armed Bandit
## Description
This repository contains an implementation of the Multi-Armed Bandit algorithms Thompson Sampling, Upper Confidence Bound, Upper Confidence Bound 1, and ε-Greedy. This repository contains C source files in the `C/` directory that can be used for reconfigurable testing of these four algorithms. The `bandit.c` source file shows how the algorithms can be simulated.

Our report contains the results of evaluating the effect of changing the simulation paramaters of Thompson Sampling on the final regret value. We also compare Thompson Sampling to UCB, UCB1, and ε-Greedy to determine under what circumstances each algorithm should be chosen. Finally, we implement a potential improvement on the Thompson Sampling algorithm in which we adjust the number of samples taken from the Beta posterior distributions at each time slot. The plots used in the report can all be found in the `Plots` directory.

## Installation
The program requires the use of the GNU Science Library (GSL). This can be installed with instructions at [GNU.org](https://www.gnu.org/software/gsl/).

## Usage
The header and source file `algorithms.h` and `algorithms.c` can be used as library files for any program. These files are made to allow users to more easily simulate Multi-Armed Bandit simulations using the four implemented algorithms. The source `bandit.c` shows how these files can be used in a full program. Example usage can be taken from that source. An example creation of an algorithm object is given below.

    Algorithm * ts = newAlgorithm( TH_ALGTYPE, 1000000, 500, 2, BERNOULLI_DIST, 1, 0, r, 0 );

This example creates a Thompson Sampling algorithm in a simulation environment with 10^6 trials, 500 simulations, 2 arms, using a Bernoulli distribution for rewards, one beta sample, and using a GSL random number generator `r` as defined in the GSL documentation. The individual parameters for this call are defined in `algorithms.h`. A reset function is also given so that the algorithms can be reset back to their initial states after each simulation for implementations where multiple simulations are required. An example call to this function using the previously created Thompson Sampling algorithm is as follows:

    reset( ts );

To simulate a full simulation, simply use a for loop to go through each time slot. At each iteration, use the `callArm` `drawReward` and `update` functions to simulate each of the respective steps in a Multi-Armed Bandit simulation. Below is an example simulation using these functionalities. This example uses the previously created Thompson Sampling algorithm object and a number of trials `TRIALS`. The array `u` is of length K, the number of arms, and each value is a double representing the mean reward of that arm. The third argument to the `drawReward` function is the standard deviation of a normal distribution. If you are not using a normal distribution for the algorithm, then this value does not matter. The variable `bestMean` is the largest value in the `u` array. 

```
int i;
for( i = 0; i < TRIALS; i++ )
{
	chooseArm( ts, i );
	drawReward( ts, u[ ts -> chosenArm ], 0, i );
	update( ts, i, u[ ts -> chosenArm ], bestMean );
}
```
