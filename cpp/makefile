CC = g++
CC_FLAGS = -g -std=c++11 -isystem . -DDOCTEST_CONFIG_DISABLE -Wall -Wextra -DEIGEN_DONT_PARALLELIZE
OPTIMIZATION = -O3 -march=native -mtune=native
WARNINGS = -pedantic -Wall -Wextra -Wfloat-equal -Wundef -Wcast-align -Wwrite-strings -Wlogical-op -Wredundant-decls -Wshadow -Woverloaded-virtual #-Wmissing-declarations 

EXEC = main.exe
SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)
LIBS = -fopenmp

$(EXEC): $(OBJECTS)
	$(CC) $(OPTIMIZATION) $(OBJECTS) -o $(EXEC) $(LIBS) $(WARNINGS)

%.o: %.cpp
	$(CC) $(OPTIMIZATION) $(CC_FLAGS) -c $< -o $@ $(LIBS) $(WARNINGS)

clean:
	rm -f $(EXEC) $(OBJECTS)
