CXX = pgc++ 
CC_FLAGS = -O3 -w -std=c++11 -acc -ta=tesla,cc60 -Minfo=accel -g -O3
INC_FLAGS = -I ../Eigen 
LIBS = 

EXEC = main.exe
SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

$(EXEC): $(OBJECTS)
	$(CXX) $(CC_FLAGS) $(INC_FLAGS) $(OBJECTS) -o $(EXEC) $(LIBS)

%.o: %.cpp
	$(CXX) $(CC_FLAGS) $(INC_FLAGS) -c $< -o $@ $(LIBS)

clean:
	rm -f $(EXEC) $(OBJECTS)
