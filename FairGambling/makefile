# Define the C compiler to use
CC = gcc

# Define the Python compiler to use
PLOT = python3

# Define any compile-time flags
CFLAGS = -Wall

# Define any libraries to link to the executable
LIBS = -lm -lgsl -lgslcblas

# Define the C source files
SRCS = src/C/gambling.c

# Define the plotting utility source files
PLOT_SRC = src/plotting/hist.py

# Define the C object files
OBJS = $(SRCS:.c=.o)

# Define the executable file
EXEC = src/C/gamblersRuin

# Define arguments for the executable
ARGS = $(ARG)


PDF_OUT = Report/Summary
PDF_SRC = Report/Report.tex


.PHONY: depend clean

# Generate gambling executable
$(EXEC): clean $(OBJS)
	$(CC) $(CFLAGS) -o $(EXEC) $(OBJS) $(LIBS)

# Execute simulation
run: $(EXEC)
	clear && ./$(EXEC) $(ARGS)

plot: $(PLOT_SRC)
	$(PLOT) $(PLOT_SRC)

pdf: cleanPDF $(PDF_SRC)
	pdflatex -halt-on-error -jobname=$(PDF_OUT) $(PDF_SRC) && pdflatex -halt-on-error -jobname=$(PDF_OUT) $(PDF_SRC) && make cleanPDF

cleanPDF:
	rm -f Report/Summary.log Report/Summary.aux

# Clean directory of object and executable files. Leave only sources left
clean: 
	rm -f $(EXEC) $(OBJS)
