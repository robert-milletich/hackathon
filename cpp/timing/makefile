CC = g++
CC_FLAGS = -O3 -w -std=c++11 -I ..

EXEC = distance.exe
SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

$(EXEC): $(OBJECTS)
	$(CC) $(OBJECTS) -o $(EXEC)

%.o: %.cpp
	$(CC) $(CC_FLAGS) -c $< -o $@

clean:
	rm -f $(EXEC) $(OBJECTS)
