# Fair Gambling
## Overview
The purpose of this project is to show that, in a fair betting game, each player's chance of winning will converge to a consistent value. The source for simulating the betting games is located in the C source file `src/C/gambling.c`. A Python plotting utility is included at `src/plotting/hist.py` to view the results of a simulation.

## How to Use
### Simulation Options
The various options for the simulation can be changed directly through the global definitions in the source file `src/C/gambling.c`. The following is a list of the available options and their meanings.

| Parameter Name        | Meaning                                                                                                |
| --------------------- | ------------------------------------------------------------------------------------------------------ |
| ALICE_STARTING        | The amount of money Alice starts out with at each game.                                                |
| BOB_STARTING          | The amount of money Bob starts out with at each game.                                                  |
| ALICE_WIN_PROBABILITY | Alice's win chance in each round. This value should be between 0 and 1. A fair game has a value of 0.5.|
| BET_AMOUNT            | The starting bet amount for each game.                                                                 |
| NUM_GAMES             | The number of games to simulate.                                                                       |
| BET_STYLE             | The betting strategy to be used. There are six defined betting strategies explained in the source.     |
| SAVE_DATA_LOC         | The file to which each game result is saved. This is used for plotting.                                |
| SAVE_DATA             | An option to turn on or off data saving. If this value is 1, then each game's results will be saved.   |

### Compilation
The basic command to compile the C source is:
`make`

To directly use `gcc`, use the following commands in order:
`gcc -c -o src/C/gambling.o src/C/gambling.c`
`gcc -o src/C/gamblersRuin src/C/gambling.o -lm -lgsl -lgslcblas`

### Execution
The following make command can be used to run the executable with or without compiling first:
`make run`

### Cleaning
Object files and executables can be automatically cleaned up using the following make command:
`make clean`

### Plotting
A Python plotting utility is provided in `src/plotting/hist.py`. The purpose of this file is to display a histogram showing how often games end after a number of rounds. The options can be changed in the source, but the default is to read from the file `data/results.dat` as written in the simulation source file. The x-axis limit can be changed by changing the `HIST_RANGE` variable and the title display can be changed by updating the `BETTING_STRATEGY` and `HIST_TITLE` attributes. 

The script was created using Python 3.6.8 and is not guaranteed to work on other versions. To execute the script, use the following command:
`python3 src/plotting/hist.py`
