# - make sure python has pybind11 installed, recommend using conda virtual environment


MAKEFILE_DIR := $(CURDIR)
SRC_DIR := $(MAKEFILE_DIR)/reuse_search_ext/src

CXX = g++
CXXFLAGS = -O3 -Wall -shared -std=c++14 -fPIC $(shell python -m pybind11 --includes) -I$(SRC_DIR) -lstdc++ -fopenmp -O3

SRCS = $(wildcard src/*.cc)
OBJS = $(patsubst src/%.cc, build/%.so, $(SRCS))

all: $(OBJS)

build/%.so: src/%.cc
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(OBJS)


# conda activate cuda111
