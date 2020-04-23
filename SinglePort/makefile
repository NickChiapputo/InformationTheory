generate: main.o generateMatrix.o
	gcc -o generate main.o generateMatrix.o -O3 -lpthread -lm

main.o: main.c generateMatrix.h
	gcc -c main.c
generateMatrixc.o: generateMatrix.h
	gcc -c generateMatrix.c