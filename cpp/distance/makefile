CXX = pgc++ 
CC_FLAGS = -O3 -std=c++11 -acc -ta=tesla,cc60 -Minfo=accel -g -O3
INC_FLAGS = -isystem ..
WARNINGS = -pedantic -Wall -Wextra -Wfloat-equal -Wundef -Wcast-align -Wwrite-strings -Wlogical-op -Wredundant-decls -Wshadow -Woverloaded-virtual #-Wmissing-declarations 

EXEC = distance.exe
SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

$(EXEC): $(OBJECTS)
	$(CXX) $(CC_FLAGS) $(OBJECTS) -o $(EXEC)

%.o: %.cpp
	$(CXX) $(CC_FLAGS) $(INC_FLAGS) -c $< -o $@ $(WARNINGS)

clean:
	rm -f $(EXEC) $(OBJECTS)
