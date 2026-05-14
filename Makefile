# Compiler and Flags
CC = gcc
CFLAGS = -Wall -Wextra -O3 -D__USE_MINGW_ANSI_STDIO=1
LDFLAGS = -lm

# Directories
SRC_DIR = src/c
BIN_DIR = bin

# Library files
LIB_SRC = $(SRC_DIR)/libdnn.c
LIB_OBJ = $(BIN_DIR)/libdnn.o
LIB_HDR = $(SRC_DIR)/libdnn.h

# Source files that require libdnn.o (files that include libdnn.h)
LIB_REQ_SRCS = $(SRC_DIR)/lorenz_dnn_final.c
LIB_REQ_BINS = $(patsubst $(SRC_DIR)/%.c, $(BIN_DIR)/%.exe, $(LIB_REQ_SRCS))

# Standalone source files (all other .c files except libdnn.c and LIB_REQ_SRCS)
STANDALONE_SRCS = $(filter-out $(LIB_SRC) $(LIB_REQ_SRCS), $(wildcard $(SRC_DIR)/*.c))
STANDALONE_BINS = $(patsubst $(SRC_DIR)/%.c, $(BIN_DIR)/%.exe, $(STANDALONE_SRCS))

# Default target
all: $(BIN_DIR) $(LIB_OBJ) $(LIB_REQ_BINS) $(STANDALONE_BINS)

# Create bin directory if it doesn't exist
$(BIN_DIR):
	powershell.exe -Command "if (!(Test-Path $(BIN_DIR))) { New-Item -ItemType Directory -Path $(BIN_DIR) }"

# Compile the library object
$(LIB_OBJ): $(LIB_SRC) $(LIB_HDR) | $(BIN_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Rule for files that REQUIRE libdnn.o
$(LIB_REQ_BINS): $(BIN_DIR)/%.exe: $(SRC_DIR)/%.c $(LIB_OBJ)
	$(CC) $(CFLAGS) $< $(LIB_OBJ) -o $@ $(LDFLAGS)

# Rule for STANDALONE files
$(STANDALONE_BINS): $(BIN_DIR)/%.exe: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# Clean target
clean:
	powershell.exe -Command "if (Test-Path $(BIN_DIR)/*.exe) { Remove-Item $(BIN_DIR)/*.exe -Force }"
	powershell.exe -Command "if (Test-Path $(LIB_OBJ)) { Remove-Item $(LIB_OBJ) -Force }"

# Help target
help:
	@echo "Available targets:"
	@echo "  all     : Build the library and all C programs (default)"
	@echo "  clean   : Remove all compiled binaries and objects"
	@echo "  help    : Show this help message"
	@echo ""
	@echo "Source files are in $(SRC_DIR)"
	@echo "Binaries are output to $(BIN_DIR)"
