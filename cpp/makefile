CC = g++
CC_FLAGS = -g -O3 -w -std=c++11 -I . -fopenmp -DEIGEN_DONT_PARALLELIZE -DDOCTEST_CONFIG_DISABLE -Wall -Wextra

EXEC = main.exe
SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

$(EXEC): $(OBJECTS)
	$(CC) $(OBJECTS) -o $(EXEC)

%.o: %.cpp
	$(CC) $(CC_FLAGS) -c $< -o $@

clean:
	rm -f $(EXEC) $(OBJECTS)
