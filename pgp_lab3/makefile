CC = nvcc
CFLAGS = --std=c++11 -Werror cross-execution-space-call -lm
SOURSES = lab3.cu
BIN = lab3

all:
	$(CC) $(CFLAGS) -o $(BIN) $(SOURSES)

run:
	# ./$(BIN) < ./tests/test_4900.t 
	# ./$(BIN) < ./tests/test_49729.t  
	# ./$(BIN) < ./tests/test_499849.t 
	# ./$(BIN) < ./tests/test_4999696.t  
	# ./$(BIN) < ./tests/test_49999041.t
	# ./$(BIN) < ./tests/test_400000000.t  
	./$(BIN) < ./tests/test_silent_hill_2_remake.t  

clean:
	rm -f *.o $(BIN)

# CXX = g++
# CXXFLAGS = -std=c++11 -O2 -Wextra -Wall -Wno-sign-compare -Wno-unused-result
# SOURSES = lab3.cpp
# BIN = lab3

# all: BIN

# BIN:
# 	$(CXX) $(CXXFLAGS) -o $(BIN) $(SOURSES)

# debug:
# 	$(CXX) $(CXXFLAGS) -o $(BIN) $(SOURSES) -g3

# run:
# 	# ./$(BIN) < ./tests/test_4900.t 
# 	# ./$(BIN) < ./tests/test_49729.t  
# 	# ./$(BIN) < ./tests/test_499849.t 
# 	# ./$(BIN) < ./tests/test_4999696.t  
# 	# ./$(BIN) < ./tests/test_49999041.t
# 	# ./$(BIN) < ./tests/test_400000000.t
# 	./$(BIN) < ./tests/test_silent_hill_2_remake.t 

# clean:
# 	rm -f *.o $(BIN)