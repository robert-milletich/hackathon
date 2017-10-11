CXX = g++ 
CC_FLAGS = -O3 -w -std=c++11 
#-acc -ta=tesla,cc60 -Minfo=accel -g -O3
INC_FLAGS = -I .. 
LIBS = 

EXEC = eigentiming.exe
SOURCES = eigentiming.cpp ../mds.cpp Timer.cpp
OBJECTS = $(SOURCES:.cpp=.o)

$(EXEC): $(OBJECTS)
	$(CXX) $(OBJECTS) $(INC_FLAGS) -o $(EXEC)

%.o: %.cpp
	$(CXX) $(CC_FLAGS) $(INC_FLAGS) -c $< -o $@

clean:
	rm -f $(EXEC) $(OBJECTS)
