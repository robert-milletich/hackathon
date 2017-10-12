#CXX = pgc++ 
CXX = g++
CC_FLAGS = -O3 -DEIGEN_DONT_PARALLELIZE  -ffast-math -g -std=c++11 -march=native -mtune=native #-acc -ta=tesla,cc60 -Minfo=accel -g -O3
INC_FLAGS = -isystem ..
WARNINGS = -pedantic -Wall -Wextra -Wfloat-equal -Wundef -Wcast-align -Wwrite-strings -Wlogical-op -Wredundant-decls -Wshadow -Woverloaded-virtual #-Wmissing-declarations 
LIBS = -fopenmp

#-Rpass-analysis=loop-vectorize -Rpass=loop-vectorize

EXEC = distance.exe
SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

$(EXEC): $(OBJECTS)
	$(CXX) $(CC_FLAGS) $(OBJECTS) -o $(EXEC) $(LIBS)

%.o: %.cpp
	$(CXX) $(CC_FLAGS) $(INC_FLAGS) -c $< -o $@ $(WARNINGS) $(LIBS)

clean:
	rm -f $(EXEC) $(OBJECTS)
