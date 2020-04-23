# Define the C compiler to use
CC = gcc

# Define the Python compiler to use
PLOT = python3

# Define any compile-time flags
CFLAGS = -Wall

# Define any libraries to link to the executable
LIBS = -lm -lgsl -lgslcblas

# Define the C source files
SRCS = C/bandit.c C/algorithms.c

# Define the C object files
OBJS = $(SRCS:.c=.o)

# Define the executable file
EXEC = C/bandit

# Define arguments for the executable
ARGS = $(ARG)


.PHONY: depend clean

# Generate bandit executable
$(EXEC): clean $(OBJS)
	$(CC) $(CFLAGS) -o $(EXEC) $(OBJS) $(LIBS)

# Execute bandit
run: $(EXEC)
	clear && ./$(EXEC) $(ARGS)

# Clean directory of object and executable files. Leave only sources left
clean: 
	rm -f $(EXEC) $(OBJS)
