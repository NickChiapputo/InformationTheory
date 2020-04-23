# Broadcasting With Port Constraints

The algorithm here contains slight differences from that described in the paper. First, each row of the symbol allocation matrix A (determining the number of symbols sent in each time slot) is determined at each time slot. This is done to save potentially significant amounts of memory. Additionally, there is little to no execution time difference. Second, determining receivers and transmitters who are available to receive or send a symbol is determined one by one instead of generating a list of available nodes. This is done to save execution time and a small amount of memory. 

Smaller, less significant differences include using 0 as a base instead of 1 and utilizing a sent and received array to determine if a node has been used for either purpose instead of using an availbility array defined in the pseudocode in the paper. Finally, both algorithms are combined into one within the program to allow for easier protocol generation.

Finally, a Python version of the generation algorithm can be found in the file "Python.ipynb".

# How to compile and run the program

Compile using `make` in terminal with gcc installed. Only tested using gcc 5.4.0 using Ubuntu 16.04.

Run using `./generate`

# Run-time options
Range refers to a range of L and K values between 1 and the inputted values.

Work pool refers to allowing the program to use multiple threads, thus increasing the execution time of the program.

Show matrix displays the matrices at each time slot for each L and K tuple.

Save results file overwrites and creates a new Results text file with all L\*K matrices


# Results

The partial zip files contain all results for L <= 500 and K <= 100. Uncompressed size of all files is 5.2 GB. Combined zip file size is 1.3 GB. This results file was generated from the Python program. Due to file size constraints, the C results could not be stored in the repository. However, it has been tested that both programs produce the same output.

Results generated from the C program are partitioned into sets of 50 L and 100 K (e.g., L = [1,49] K = [1,99], L = [50,99] K = [1,99], etc.) to create smaller results files to counter issues opening, saving, or sending them.
