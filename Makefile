all:
	g++ main.cpp -lglut -lGL -lfftw3f -lGLU -O2 -Wall -march=native
