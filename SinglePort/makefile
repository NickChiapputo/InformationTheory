# Define the C compiler to use
CC = gcc

# Define any compile-time flags
CFLAGS =

# Define any libraries to link to the executable
LIBS = -lpthread -lm

# Define the C source files
SRCS = main.c generateMatrix.c

# Define the C object files
OBJS = $(SRCS:.c=.o)

# Define the executable file
EXEC = generate

# Define arguments for the executable
ARGS = $(ARG)


.PHONY: depend clean

# Generate executable
$(EXEC): clean $(OBJS)
	$(CC) $(CFLAGS) -o $(EXEC) $(OBJS) $(LIBS)

# Execute program
run: $(EXEC)
	clear && ./$(EXEC)

# Clean directory of object and executable files. Leave only sources left
clean: 
	rm -f $(EXEC) $(OBJS)
