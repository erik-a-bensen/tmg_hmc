# Find available C++ compiler
CXX := $(shell command -v g++ 2> /dev/null)
ifndef CXX
    CXX := $(shell command -v clang++ 2> /dev/null)
endif
ifndef CXX
    CXX := $(shell command -v c++ 2> /dev/null)
endif
ifndef CXX
    $(error No C++ compiler found. Please install g++, clang++, or another C++ compiler)
endif

CXXFLAGS=-O3 -Wall -fPIC
target=utils.cpp
output=calc_solutions

ifeq ($(OS),Windows_NT)
    share_ext=dll
    LDFLAGS=-shared
    RM=del /Q
else
    UNAME_S := $(shell uname -s)
    RM=rm -f
    
    ifeq ($(UNAME_S),Linux)
        share_ext=so
        LDFLAGS=-shared
    endif
    
    ifeq ($(UNAME_S),Darwin)
        share_ext=dylib
        LDFLAGS=-dynamiclib -undefined dynamic_lookup
    endif
endif

.PHONY: all clean

all: $(output).$(share_ext)

$(output).$(share_ext): $(target)
	@echo "Using compiler: $(CXX)"
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(target) -o $(output).$(share_ext)

clean:
	$(RM) $(output).*