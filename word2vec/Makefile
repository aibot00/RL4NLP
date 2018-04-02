###############################
#  makefile for common
#  author :  wangyan@aibot.me
#  date : 2017-01-15
###############################


WORKROOT = ../../


LIBNAME = word2vec
BIN_DIR = bin
SRC_DIR = src
LIB_DIR = lib
DATA_DIR = data
TEST_DIR = test
INCLUDE_DIR = include
OUTPUT_DIR = output


THIRD64 = $(WORKROOT)/third64
PUBLIC = $(WORKROOT)/public


#------ include  and libs ---------------------------#

INCLUDEFLAGS = -I$(INCLUDE_DIR) -I$(SRC_DIR) 

LIBFLAGS = -lpthread -lssl -lcrypto -lz -lrt -ldl



VERSION_SIGN = -D__VERSION_ID__="\"1.0.1\""
CXXFLAGS = -pipe -O2 -W -Wall -fPIC -fopenmp -finline-functions -Wno-literal-suffix -rdynamic
USERFLAGS = -g -std=c++11 
CXX = g++ $(CXXFLAGS) $(USERFLAGS) $(VERSION_SIGN)


## generate automaton
#$(shell flex++ -8 -Cf -o$(SRC_DIR)/ChunkParser.cpp $(SRC_DIR)/ChunkParser.l)
CODE_SRC_C = $(wildcard $(SRC_DIR)/*.c)
CODE_OBJS_C = $(patsubst %.c,%.o,$(CODE_SRC_C))

CODE_SRC_CPP = $(wildcard $(SRC_DIR)/*.cpp)
CODE_OBJS_CPP = $(patsubst %.cpp,%.o,$(CODE_SRC_CPP))


TEST_SOURCES = $(wildcard $(TEST_DIR)/*.cpp)
TEST_EXES = $(patsubst %.cpp,%,$(TEST_SOURCES))
BIN_EXES = $(patsubst $(TEST_DIR)/%,$(BIN_DIR)/%,$(TEST_EXES))


.PHONY: all clean lib test

#-----------------------------------------------------------------#
# make all
#-----------------------------------------------------------------#
all: tool
	if [ ! -d $(OUTPUT_DIR)/lib ]; then mkdir -p $(OUTPUT_DIR)/lib; fi
	cp lib/lib$(LIBNAME).a $(OUTPUT_DIR)/lib
	if [ ! -d $(OUTPUT_DIR)/include ]; then mkdir -p $(OUTPUT_DIR)/include; fi
	cp -rf include/* $(OUTPUT_DIR)/include


tool: lib $(TEST_EXES)

#-----------------------------------------------------------------#
# make library
#-----------------------------------------------------------------#
lib: $(CODE_OBJS_C) $(CODE_OBJS_CPP)
	if [ ! -d $(LIB_DIR) ]; then mkdir $(LIB_DIR); fi
	ar -ruv $(LIB_DIR)/lib$(LIBNAME).a $(CODE_OBJS_C) $(CODE_OBJS_CPP)

$(CODE_OBJS_C): %.o: %.c
	$(CXX) -c $< -o $@ $(INCLUDEFLAGS)

$(CODE_OBJS_CPP): %.o: %.cpp
	$(CXX) -c $< -o $@ $(INCLUDEFLAGS)


#-----------------------------------------------------------------#
# make test
#-----------------------------------------------------------------#
$(TEST_EXES): %: %.cpp
	$(CXX) $< -o $@ $(INCLUDEFLAGS) $(CODE_OBJS_C) $(CODE_OBJS_CPP) $(LIBFLAGS)
	if [ ! -d $(BIN_DIR) ]; then mkdir $(BIN_DIR); fi
	mv $@ $(BIN_DIR)/


#$(CXX) $< -o $@ $(INCLUDEFLAGS) -L$(LIB_DIR) -l$(LIBNAME) $(LIBFLAGS)


#-----------------------------------------------------------------#
# make clean
#-----------------------------------------------------------------#
clean:
	rm -f $(CODE_OBJS_C) $(CODE_OBJS_CPP) $(TEST_EXES) $(BIN_EXES) $(LIB_DIR)/*.a
	if [ -d $(OUTPUT_DIR) ]; then rm -r $(OUTPUT_DIR); fi
	if [ -d $(LIB_DIR) ]; then rm -r $(LIB_DIR); fi
	if [ -d $(BIN_DIR) ]; then rm -r $(BIN_DIR); fi
