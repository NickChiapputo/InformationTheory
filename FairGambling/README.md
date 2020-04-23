# Fair Gambling
## Overview
The purpose of this project is to show that, in a fair betting game, each player's chance of winning will converge to a consistent value. The source for simulating the betting games is located in the C source file `src/C/gambling.c`.

## How to Use
### Simulation Options
The various options for the simulation can be changed directly through the global definitions in the source file `src/C/gambling.c`. The following is a list of the available options and their meanings.

| Parameter Name        | Meaning                                                                                                |
| --------------------- | ------------------------------------------------------------------------------------------------------ |
| ALICE_STARTING        | The amount of money Alice starts out with at each game                                                 |
| BOB_STARTING          | The amount of money Bob starts out with at each game                                                   |
| ALICE_WIN_PROBABILITY | Alice's win chance in each round. This value should be between 0 and 1. A fair game has a value of 0.5 |
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
