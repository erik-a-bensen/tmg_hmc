calc_solutions: utils.cpp
	g++ -c -fPIC utils.cpp -o calc_solutions.o
	g++ -shared -o calc_solutions.so calc_solutions.o

clean:
	rm -f *.o *.so

rebuild: clean calc_solutions
