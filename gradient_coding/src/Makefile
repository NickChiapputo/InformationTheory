# No. of workers. Add one for the master
N_PROCS ?= 13

# No. of stragglers in our coding schemes
N_STRAGGLERS ?= 1

# No. of iterations
N_ITERATIONS ?= 1

# Straggler Delay (seconds)
STRAGGLER_DELAY ?= 0

# For partially coded version: how many pieces of workload will one worker be handling.
N_PARTITIONS ?= 10

# Switch to enable partial coded schemes
PARTIAL_CODED ?= 0

# Path to folder containing the data folders
DATA_FOLDER ?= straggdata/

IS_REAL ?= 0

DATASET ?= amazon-dataset
N_ROWS ?= 554400
N_COLS ?= 100

# Note that DATASET is automatically set to artificial-data/ (n_rows)x(n_cols)/... if IS_REAL is set to 0 \
# or artificial-data/partial/ (n_rows)x(n_cols)/... if PARTIAL_CODED is also set to 1

generate_random_data:
	python ./src/generate_data.py $(N_PROCS) $(N_ROWS) $(N_COLS) $(DATA_FOLDER) $(N_STRAGGLERS) $(N_PARTITIONS) $(PARTIAL_CODED)

arrange_real_data:
	python ./src/arrange_real_data.py $(N_PROCS) $(DATA_FOLDER) $(DATASET) $(N_STRAGGLERS) $(N_PARTITIONS) $(PARTIAL_CODED)

naive:  
	@echo \
	"Naive Scheme:\n" 							\
	"    Workers:       $$(( $(N_PROCS) - 1 ))\n" 	\
	"    Samples:       $(N_ROWS)\n" 				\
	"    Stragglers:    $(N_STRAGGLERS)\n"			\
	"    Straggle Time: $(STRAGGLER_DELAY)\n"

	mpirun -np $(N_PROCS) python main.py $(N_PROCS) $(N_ROWS) $(N_COLS) $(DATA_FOLDER) $(IS_REAL) $(DATASET) 0 $(N_STRAGGLERS) $(STRAGGLER_DELAY) $(N_ITERATIONS)  0 0

cyccoded:
	@echo \
	"Cyclic Repetition Scheme:\n" 							\
	"    Workers:       $$(( $(N_PROCS) - 1 ))\n" 	\
	"    Samples:       $(N_ROWS)\n" 				\
	"    Stragglers:    $(N_STRAGGLERS)\n"			\
	"    Straggle Time: $(STRAGGLER_DELAY)\n"

	mpirun -np $(N_PROCS) python main.py $(N_PROCS) $(N_ROWS) $(N_COLS) $(DATA_FOLDER) $(IS_REAL) $(DATASET) 1 $(N_STRAGGLERS) $(STRAGGLER_DELAY) $(N_ITERATIONS)  0 0

repcoded:
	@echo \
	"Fractional Repetition Scheme:\n" 				\
	"    Workers:       $$(( $(N_PROCS) - 1 ))\n" 	\
	"    Samples:       $(N_ROWS)\n" 				\
	"    Stragglers:    $(N_STRAGGLERS)\n"			\
	"    Straggle Time: $(STRAGGLER_DELAY)\n"

	mpirun -np $(N_PROCS) python main.py $(N_PROCS) $(N_ROWS) $(N_COLS) $(DATA_FOLDER) $(IS_REAL) $(DATASET) 1 $(N_STRAGGLERS) $(STRAGGLER_DELAY) $(N_ITERATIONS)  0 1

avoidstragg:
	@echo \
	"Avoid Stragglers Scheme:\n" 				\
	"    Workers:       $$(( $(N_PROCS) - 1 ))\n" 	\
	"    Samples:       $(N_ROWS)\n" 				\
	"    Stragglers:    $(N_STRAGGLERS)\n"			\
	"    Straggle Time: $(STRAGGLER_DELAY)\n"

	mpirun -np $(N_PROCS) python main.py $(N_PROCS) $(N_ROWS) $(N_COLS) $(DATA_FOLDER) $(IS_REAL) $(DATASET) 1 $(N_STRAGGLERS) $(STRAGGLER_DELAY) $(N_ITERATIONS)  0 2

partialrepcoded:
	@echo \
	"Partial Repetition Scheme:\n" 				\
	"    Workers:       $$(( $(N_PROCS) - 1 ))\n" 	\
	"    Samples:       $(N_ROWS)\n" 				\
	"    Stragglers:    $(N_STRAGGLERS)\n"			\
	"    Straggle Time: $(STRAGGLER_DELAY)\n"

	mpirun -np $(N_PROCS) python main.py $(N_PROCS) $(N_ROWS) $(N_COLS) $(DATA_FOLDER) $(IS_REAL) $(DATASET) 1 $(N_STRAGGLERS) $(STRAGGLER_DELAY) $(N_ITERATIONS) $(N_PARTITIONS) 1

partialcyccoded:
	@echo \
	"Partial Cyclic Scheme:\n" 				\
	"    Workers:       $$(( $(N_PROCS) - 1 ))\n" 	\
	"    Samples:       $(N_ROWS)\n" 				\
	"    Stragglers:    $(N_STRAGGLERS)\n"			\
	"    Straggle Time: $(STRAGGLER_DELAY)\n"

	mpirun -np $(N_PROCS) python main.py $(N_PROCS) $(N_ROWS) $(N_COLS) $(DATA_FOLDER) $(IS_REAL) $(DATASET) 1 $(N_STRAGGLERS) $(STRAGGLER_DELAY) $(N_ITERATIONS) $(N_PARTITIONS) 0
