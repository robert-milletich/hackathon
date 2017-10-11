CC = g++
CC_FLAGS = -g -pg -O0 -std=c++11 -isystem . -DDOCTEST_CONFIG_DISABLE
WARNINGS = -pedantic -Wall -Wextra -Wfloat-equal -Wundef -Wcast-align -Wwrite-strings -Wlogical-op -Wredundant-decls -Wshadow -Woverloaded-virtual #-Wmissing-declarations 

EXEC = main.exe
SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

$(EXEC): $(OBJECTS)
	$(CC) $(OBJECTS) -o $(EXEC) $(WARNINGS)

%.o: %.cpp
	$(CC) $(CC_FLAGS) -c $< -o $@ $(WARNINGS)

clean:
	rm -f $(EXEC) $(OBJECTS)
