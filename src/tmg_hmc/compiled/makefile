# Find available C++ compiler
CXX := $(shell command -v g++ 2> /dev/null)
ifndef CXX
    CXX := $(shell command -v clang++ 2> /dev/null)
endif
ifndef CXX
    CXX := $(shell command -v c++ 2> /dev/null)
endif
ifndef CXX
    CXX := $(shell command -v cl 2> /dev/null)  # MSVC on Windows
endif
ifndef CXX
    $(error No C++ compiler found. Please install g++, clang++, or MSVC)
endif

# Use older C++ standard for better compatibility
CXXFLAGS=-O3 -Wall -fPIC -std=c++14
target=utils.cpp
output=calc_solutions

ifeq ($(OS),Windows_NT)
    share_ext=dll
    # Static linking on Windows
    LDFLAGS=-shared -static-libgcc -static-libstdc++
    RM=del /Q
    CXXFLAGS=-O3 -Wall -std=c++14
else
    UNAME_S := $(shell uname -s)
    RM=rm -f
    
    ifeq ($(UNAME_S),Linux)
        share_ext=so
        # Static linking on Linux - embed libstdc++ and libgcc
        LDFLAGS=-shared -static-libstdc++ -static-libgcc
    endif
    
    ifeq ($(UNAME_S),Darwin)
        share_ext=dylib
        # macOS: use libc++ (system C++ library) which is always available
        LDFLAGS=-dynamiclib -undefined dynamic_lookup
        # Use libc++ instead of libstdc++ on macOS
        CXXFLAGS=-O3 -Wall -fPIC -std=c++14 -stdlib=libc++
    endif
endif

.PHONY: all clean

all: $(output).$(share_ext)

$(output).$(share_ext): $(target)
	@echo "Using compiler: $(CXX)"
	@echo "Platform: $(UNAME_S)"
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(target) -o $(output).$(share_ext)
	@echo "Checking dependencies:"
	@if [ "$(UNAME_S)" = "Linux" ]; then ldd $(output).$(share_ext) || true; fi
	@if [ "$(UNAME_S)" = "Darwin" ]; then otool -L $(output).$(share_ext) || true; fi

clean:
	$(RM) $(output).*

rebuild: clean all