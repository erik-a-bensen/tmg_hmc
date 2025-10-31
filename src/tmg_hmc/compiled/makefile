# Detect platform
UNAME_S := $(shell uname -s 2>/dev/null)
ifeq ($(UNAME_S),)
    # Windows without uname
    PLATFORM := Windows
else
    PLATFORM := $(UNAME_S)
endif

# Find C++ compiler
CXX := $(shell which g++ 2>/dev/null || which clang++ 2>/dev/null || which c++ 2>/dev/null || echo g++)

# Common flags
CXXFLAGS = -O3 -Wall -std=c++14
TARGET = utils.cpp
OUTPUT = calc_solutions

# Platform-specific settings
ifeq ($(PLATFORM),Windows)
    # Windows with MinGW
    EXT = dll
    LDFLAGS = -shared -static-libgcc -static-libstdc++
    RM = del /F /Q
    # Remove -fPIC on Windows (not needed)
else ifeq ($(PLATFORM),Linux)
    # Linux
    EXT = so
    CXXFLAGS += -fPIC
    LDFLAGS = -shared -static-libstdc++ -static-libgcc
    RM = rm -f
else ifeq ($(PLATFORM),Darwin)
    # macOS
    EXT = dylib
    CXXFLAGS += -fPIC -stdlib=libc++
    LDFLAGS = -dynamiclib -undefined dynamic_lookup -stdlib=libc++
    RM = rm -f
    
    # Support cross-compilation for universal binaries
    ifdef ARCHFLAGS
        CXXFLAGS += $(ARCHFLAGS)
        LDFLAGS += $(ARCHFLAGS)
    endif
else
    # Default to Linux-like
    EXT = so
    CXXFLAGS += -fPIC
    LDFLAGS = -shared
    RM = rm -f
endif

# Targets
.PHONY: all clean

all: $(OUTPUT).$(EXT)

$(OUTPUT).$(EXT): $(TARGET)
	@echo "================================"
	@echo "Platform: $(PLATFORM)"
	@echo "Compiler: $(CXX)"
	@echo "Flags: $(CXXFLAGS) $(LDFLAGS)"
	@echo "================================"
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(TARGET) -o $(OUTPUT).$(EXT)
	@echo "Build complete: $(OUTPUT).$(EXT)"

clean:
ifeq ($(PLATFORM),Windows)
	-@$(RM) $(OUTPUT).$(EXT) 2>nul || exit 0
else
	$(RM) $(OUTPUT).$(EXT)
endif

# Help target
help:
	@echo "Available targets:"
	@echo "  all    - Build the shared library (default)"
	@echo "  clean  - Remove built files"
	@echo "  help   - Show this help message"