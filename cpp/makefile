CC = g++
CC_FLAGS = -g -O3 -std=c++11 -isystem . -DDOCTEST_CONFIG_DISABLE -Wall -Wextra -DEIGEN_DONT_PARALLELIZE

EXEC = main.exe
SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)
LIBS = -fopenmp

$(EXEC): $(OBJECTS)
	$(CC) $(OBJECTS) -o $(EXEC) $(LIBS)

%.o: %.cpp
	$(CC) $(CC_FLAGS) -c $< -o $@ $(LIBS)

clean:
	rm -f $(EXEC) $(OBJECTS)
