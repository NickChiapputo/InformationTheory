#!/bin/sh

# for i in $(seq 0 19)
# do
# 	clear
# 	make avoidstragg N_PROCS=21 N_STRAGGLERS=$i STRAGGLER_DELAY=1 N_ROWS=55440
# done


case ${1} in
	fig5)
		clear
		echo "Figure 5"

		n_strag=2
		strag_delay=0.68

		make naive 		N_PROCS=13 N_STRAGGLERS=${n_strag} N_ITERATIONS=100 STRAGGLER_DELAY=${strag_delay} IS_REAL=0 N_ROWS=554400 N_COLS=100
		make cyccoded 	N_PROCS=13 N_STRAGGLERS=${n_strag} N_ITERATIONS=100 STRAGGLER_DELAY=${strag_delay} IS_REAL=0 N_ROWS=554400 N_COLS=100
		make repcoded 	N_PROCS=13 N_STRAGGLERS=${n_strag} N_ITERATIONS=100 STRAGGLER_DELAY=${strag_delay} IS_REAL=0 N_ROWS=554400 N_COLS=100
		;;

	fig6)
		clear
		echo "Figure 6"

		#  ============================================= 
		#  ============== Subfigure One ============== 
		# Naive | FracRep s=1 | CycRep s=1 | Ignore Stragg s=1	
		n_procs=11
		strag_delay=0
		n_strag=1

		make naive			N_PROCS=${n_procs} N_STRAGGLERS=0 			N_ITERATIONS=100 STRAGGLER_DELAY=${strag_delay} IS_REAL=1 N_ROWS=26210 N_COLS=241915
		echo "================================================================================\n================================================================================\n\n"
		make repcoded		N_PROCS=${n_procs} N_STRAGGLERS=${n_strag} 	N_ITERATIONS=100 STRAGGLER_DELAY=${strag_delay} IS_REAL=1 N_ROWS=26210 N_COLS=241915
		echo "================================================================================\n================================================================================\n\n"
		make cyccoded		N_PROCS=${n_procs} N_STRAGGLERS=${n_strag} 	N_ITERATIONS=100 STRAGGLER_DELAY=${strag_delay} IS_REAL=1 N_ROWS=26210 N_COLS=241915
		echo "================================================================================\n================================================================================\n\n"
		make avoidstragg	N_PROCS=${n_procs} N_STRAGGLERS=${n_strag} 	N_ITERATIONS=100 STRAGGLER_DELAY=${strag_delay} IS_REAL=1 N_ROWS=26210 N_COLS=241915
		echo "================================================================================\n================================================================================\n\n"
		#  ============================================= 
		#  ============================================= 

		#  ============================================= 
		#  ============== Subfigure Two ============== 
		# Naive | FracRep s=3 | FracRep s=4 | CycRep s=3 | CycRep s=4 | Ignore Stragg s=3 | Ignore Stragg s=4
		n_procs=21
		n_strag=3

		make naive			N_PROCS=${n_procs} N_STRAGGLERS=0 			N_ITERATIONS=100 STRAGGLER_DELAY=${strag_delay} IS_REAL=1 N_ROWS=26210 N_COLS=241915
		echo "================================================================================\n================================================================================\n\n"
		make repcoded		N_PROCS=${n_procs} N_STRAGGLERS=${n_strag} 	N_ITERATIONS=100 STRAGGLER_DELAY=${strag_delay} IS_REAL=1 N_ROWS=26210 N_COLS=241915
		echo "================================================================================\n================================================================================\n\n"
		make cyccoded		N_PROCS=${n_procs} N_STRAGGLERS=${n_strag} 	N_ITERATIONS=100 STRAGGLER_DELAY=${strag_delay} IS_REAL=1 N_ROWS=26210 N_COLS=241915
		echo "================================================================================\n================================================================================\n\n"
		make avoidstragg	N_PROCS=${n_procs} N_STRAGGLERS=${n_strag} 	N_ITERATIONS=100 STRAGGLER_DELAY=${strag_delay} IS_REAL=1 N_ROWS=26210 N_COLS=241915
		echo "================================================================================\n================================================================================\n\n"


		n_strag=4

		make naive			N_PROCS=${n_procs} N_STRAGGLERS=0 			N_ITERATIONS=100 STRAGGLER_DELAY=${strag_delay} IS_REAL=1 N_ROWS=26210 N_COLS=241915
		echo "================================================================================\n================================================================================\n\n"
		make repcoded		N_PROCS=${n_procs} N_STRAGGLERS=${n_strag} 	N_ITERATIONS=100 STRAGGLER_DELAY=${strag_delay} IS_REAL=1 N_ROWS=26210 N_COLS=241915
		echo "================================================================================\n================================================================================\n\n"
		make cyccoded		N_PROCS=${n_procs} N_STRAGGLERS=${n_strag} 	N_ITERATIONS=100 STRAGGLER_DELAY=${strag_delay} IS_REAL=1 N_ROWS=26210 N_COLS=241915
		echo "================================================================================\n================================================================================\n\n"
		make avoidstragg	N_PROCS=${n_procs} N_STRAGGLERS=${n_strag} 	N_ITERATIONS=100 STRAGGLER_DELAY=${strag_delay} IS_REAL=1 N_ROWS=26210 N_COLS=241915
		echo "================================================================================\n================================================================================\n\n"
		#  ============================================= 
		#  ============================================= 

		#  ============================================= 
		#  ============== Subfigure Three ============== 
		# Naive | FracRep s=5 | FracRep s=9 | CycRep s=5 | CycRep s=9 | Ignore Stragg s=5 | Ignore Stragg s=9
		n_procs=31
		n_strag=5

		make naive			N_PROCS=${n_procs} N_STRAGGLERS=0 			N_ITERATIONS=100 STRAGGLER_DELAY=${strag_delay} IS_REAL=1 N_ROWS=26210 N_COLS=241915
		echo "================================================================================\n================================================================================\n\n"
		make repcoded		N_PROCS=${n_procs} N_STRAGGLERS=${n_strag} 	N_ITERATIONS=100 STRAGGLER_DELAY=${strag_delay} IS_REAL=1 N_ROWS=26210 N_COLS=241915
		echo "================================================================================\n================================================================================\n\n"
		make cyccoded		N_PROCS=${n_procs} N_STRAGGLERS=${n_strag} 	N_ITERATIONS=100 STRAGGLER_DELAY=${strag_delay} IS_REAL=1 N_ROWS=26210 N_COLS=241915
		echo "================================================================================\n================================================================================\n\n"
		make avoidstragg	N_PROCS=${n_procs} N_STRAGGLERS=${n_strag} 	N_ITERATIONS=100 STRAGGLER_DELAY=${strag_delay} IS_REAL=1 N_ROWS=26210 N_COLS=241915
		echo "================================================================================\n================================================================================\n\n"


		n_strag=9

		make naive			N_PROCS=${n_procs} N_STRAGGLERS=0 			N_ITERATIONS=100 STRAGGLER_DELAY=${strag_delay} IS_REAL=1 N_ROWS=26210 N_COLS=241915
		echo "================================================================================\n================================================================================\n\n"
		make repcoded		N_PROCS=${n_procs} N_STRAGGLERS=${n_strag} 	N_ITERATIONS=100 STRAGGLER_DELAY=${strag_delay} IS_REAL=1 N_ROWS=26210 N_COLS=241915
		echo "================================================================================\n================================================================================\n\n"
		make cyccoded		N_PROCS=${n_procs} N_STRAGGLERS=${n_strag} 	N_ITERATIONS=100 STRAGGLER_DELAY=${strag_delay} IS_REAL=1 N_ROWS=26210 N_COLS=241915
		echo "================================================================================\n================================================================================\n\n"
		make avoidstragg	N_PROCS=${n_procs} N_STRAGGLERS=${n_strag} 	N_ITERATIONS=100 STRAGGLER_DELAY=${strag_delay} IS_REAL=1 N_ROWS=26210 N_COLS=241915
		echo "================================================================================\n================================================================================\n\n"
		#  ============================================= 
		#  ============================================= 
		;;

	fig7)
		clear
		echo "Figure 7"

		#  ============================================= 
		#  ============== Subfigure One ============== 
		# FracRep s=1 | CycRep s=1 | Ignore Stragg s=1	
		n_procs=11
		strag_delay=0
		n_strag=1

		make repcoded		N_PROCS=${n_procs} N_STRAGGLERS=${n_strag} 	N_ITERATIONS=100 STRAGGLER_DELAY=${strag_delay} IS_REAL=1 N_ROWS=26210 N_COLS=241915
		echo "================================================================================\n================================================================================\n\n"
		make cyccoded		N_PROCS=${n_procs} N_STRAGGLERS=${n_strag} 	N_ITERATIONS=100 STRAGGLER_DELAY=${strag_delay} IS_REAL=1 N_ROWS=26210 N_COLS=241915
		echo "================================================================================\n================================================================================\n\n"
		make avoidstragg	N_PROCS=${n_procs} N_STRAGGLERS=${n_strag} 	N_ITERATIONS=100 STRAGGLER_DELAY=${strag_delay} IS_REAL=1 N_ROWS=26210 N_COLS=241915
		echo "================================================================================\n================================================================================\n\n"
		#  ============================================= 
		#  ============================================= 

		#  ============================================= 
		#  ============== Subfigure Two ============== 
		# FracRep s=3 | FracRep s=4 | CycRep s=3 | CycRep s=4 | Ignore Stragg s=3 | Ignore Stragg s=4
		n_procs=21
		n_strag=3

		make repcoded		N_PROCS=${n_procs} N_STRAGGLERS=${n_strag} 	N_ITERATIONS=100 STRAGGLER_DELAY=${strag_delay} IS_REAL=1 N_ROWS=26210 N_COLS=241915
		echo "================================================================================\n================================================================================\n\n"
		make cyccoded		N_PROCS=${n_procs} N_STRAGGLERS=${n_strag} 	N_ITERATIONS=100 STRAGGLER_DELAY=${strag_delay} IS_REAL=1 N_ROWS=26210 N_COLS=241915
		echo "================================================================================\n================================================================================\n\n"
		make avoidstragg	N_PROCS=${n_procs} N_STRAGGLERS=${n_strag} 	N_ITERATIONS=100 STRAGGLER_DELAY=${strag_delay} IS_REAL=1 N_ROWS=26210 N_COLS=241915
		echo "================================================================================\n================================================================================\n\n"


		n_strag=4

		make repcoded		N_PROCS=${n_procs} N_STRAGGLERS=${n_strag} 	N_ITERATIONS=100 STRAGGLER_DELAY=${strag_delay} IS_REAL=1 N_ROWS=26210 N_COLS=241915
		echo "================================================================================\n================================================================================\n\n"
		make cyccoded		N_PROCS=${n_procs} N_STRAGGLERS=${n_strag} 	N_ITERATIONS=100 STRAGGLER_DELAY=${strag_delay} IS_REAL=1 N_ROWS=26210 N_COLS=241915
		echo "================================================================================\n================================================================================\n\n"
		make avoidstragg	N_PROCS=${n_procs} N_STRAGGLERS=${n_strag} 	N_ITERATIONS=100 STRAGGLER_DELAY=${strag_delay} IS_REAL=1 N_ROWS=26210 N_COLS=241915
		echo "================================================================================\n================================================================================\n\n"
		#  ============================================= 
		#  ============================================= 

		#  ============================================= 
		#  ============== Subfigure Three ============== 
		# FracRep s=4 | FracRep s=9 | CycRep s=4 | CycRep s=9 | Ignore Stragg s=4 | Ignore Stragg s=9
		n_procs=31
		n_strag=4

		make repcoded		N_PROCS=${n_procs} N_STRAGGLERS=${n_strag} 	N_ITERATIONS=100 STRAGGLER_DELAY=${strag_delay} IS_REAL=1 N_ROWS=26210 N_COLS=241915
		echo "================================================================================\n================================================================================\n\n"
		make cyccoded		N_PROCS=${n_procs} N_STRAGGLERS=${n_strag} 	N_ITERATIONS=100 STRAGGLER_DELAY=${strag_delay} IS_REAL=1 N_ROWS=26210 N_COLS=241915
		echo "================================================================================\n================================================================================\n\n"
		make avoidstragg	N_PROCS=${n_procs} N_STRAGGLERS=${n_strag} 	N_ITERATIONS=100 STRAGGLER_DELAY=${strag_delay} IS_REAL=1 N_ROWS=26210 N_COLS=241915
		echo "================================================================================\n================================================================================\n\n"


		n_strag=9

		make repcoded		N_PROCS=${n_procs} N_STRAGGLERS=${n_strag} 	N_ITERATIONS=100 STRAGGLER_DELAY=${strag_delay} IS_REAL=1 N_ROWS=26210 N_COLS=241915
		echo "================================================================================\n================================================================================\n\n"
		make cyccoded		N_PROCS=${n_procs} N_STRAGGLERS=${n_strag} 	N_ITERATIONS=100 STRAGGLER_DELAY=${strag_delay} IS_REAL=1 N_ROWS=26210 N_COLS=241915
		echo "================================================================================\n================================================================================\n\n"
		make avoidstragg	N_PROCS=${n_procs} N_STRAGGLERS=${n_strag} 	N_ITERATIONS=100 STRAGGLER_DELAY=${strag_delay} IS_REAL=1 N_ROWS=26210 N_COLS=241915
		echo "================================================================================\n================================================================================\n\n"
		#  ============================================= 
		#  ============================================= 
		;;

	*)
		echo "Unknown Argument"
		;;
esac

exit

clear
make avoidstragg N_PROCS=11 N_STRAGGLERS=1 STRAGGLER_DELAY=1 IS_REAL=1 N_ROWS=26210 N_COLS=241915

clear
make cyccoded N_PROCS=11 N_STRAGGLERS=1 STRAGGLER_DELAY=1 IS_REAL=1 N_ROWS=26210 N_COLS=241915

clear
make repcoded N_PROCS=11 N_STRAGGLERS=1 STRAGGLER_DELAY=1 IS_REAL=1 N_ROWS=26210 N_COLS=241915



clear
make avoidstragg N_PROCS=21 N_STRAGGLERS=3 STRAGGLER_DELAY=1 IS_REAL=1 N_ROWS=26210 N_COLS=241915
clear
make avoidstragg N_PROCS=21 N_STRAGGLERS=4 STRAGGLER_DELAY=1 IS_REAL=1 N_ROWS=26210 N_COLS=241915

clear
make cyccoded N_PROCS=21 N_STRAGGLERS=3 STRAGGLER_DELAY=1 IS_REAL=1 N_ROWS=26210 N_COLS=241915
clear
make cyccoded N_PROCS=21 N_STRAGGLERS=4 STRAGGLER_DELAY=1 IS_REAL=1 N_ROWS=26210 N_COLS=241915

clear
make repcoded N_PROCS=21 N_STRAGGLERS=3 STRAGGLER_DELAY=1 IS_REAL=1 N_ROWS=26210 N_COLS=241915
clear
make repcoded N_PROCS=21 N_STRAGGLERS=4 STRAGGLER_DELAY=1 IS_REAL=1 N_ROWS=26210 N_COLS=241915



clear
make avoidstragg N_PROCS=31 N_STRAGGLERS=4 STRAGGLER_DELAY=1 IS_REAL=1 N_ROWS=26210 N_COLS=241915
clear
make avoidstragg N_PROCS=31 N_STRAGGLERS=9 STRAGGLER_DELAY=1 IS_REAL=1 N_ROWS=26210 N_COLS=241915

clear
make cyccoded N_PROCS=31 N_STRAGGLERS=4 STRAGGLER_DELAY=1 IS_REAL=1 N_ROWS=26210 N_COLS=241915
clear
make cyccoded N_PROCS=31 N_STRAGGLERS=9 STRAGGLER_DELAY=1 IS_REAL=1 N_ROWS=26210 N_COLS=241915

clear
make repcoded N_PROCS=31 N_STRAGGLERS=4 STRAGGLER_DELAY=1 IS_REAL=1 N_ROWS=26210 N_COLS=241915
clear
make repcoded N_PROCS=31 N_STRAGGLERS=9 STRAGGLER_DELAY=1 IS_REAL=1 N_ROWS=26210 N_COLS=241915

