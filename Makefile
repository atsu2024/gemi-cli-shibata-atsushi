# Compiler and Flags
CC = gcc
CFLAGS = -Wall -Wextra -O3 -D__USE_MINGW_ANSI_STDIO=1
LDFLAGS = -lm

GOC = go build
GOFLAGS = -o

# Directories
SRC_C_DIR = src/c
SRC_GO_DIR = src/go
BIN_DIR = bin

# Library files
LIB_SRC = $(SRC_C_DIR)/libdnn.c
LIB_OBJ = $(BIN_DIR)/libdnn.o
LIB_HDR = $(SRC_C_DIR)/libdnn.h

# C Source files that require libdnn.o
LIB_REQ_SRCS = $(SRC_C_DIR)/lorenz_dnn_final.c
LIB_REQ_BINS = $(patsubst $(SRC_C_DIR)/%.c, $(BIN_DIR)/%.exe, $(LIB_REQ_SRCS))

# Standalone C source files (src/c)
STANDALONE_C_SRCS = $(filter-out $(LIB_SRC) $(LIB_REQ_SRCS), $(wildcard $(SRC_C_DIR)/*.c))
STANDALONE_C_BINS = $(patsubst $(SRC_C_DIR)/%.c, $(BIN_DIR)/%.exe, $(STANDALONE_C_SRCS))

# Root level C source files
ROOT_C_SRCS = $(wildcard *.c)
ROOT_C_BINS = $(patsubst %.c, $(BIN_DIR)/%.exe, $(ROOT_C_SRCS))

# Go source files (src/go)
SRC_GO_FILES = $(wildcard $(SRC_GO_DIR)/*.go)
SRC_GO_BINS = $(patsubst $(SRC_GO_DIR)/%.go, $(BIN_DIR)/%_go.exe, $(SRC_GO_FILES))

# Root level Go source files
ROOT_GO_SRCS = $(wildcard *.go)
ROOT_GO_BINS = $(patsubst %.go, $(BIN_DIR)/%_root.exe, $(ROOT_GO_SRCS))

# Default target
all: $(BIN_DIR) $(LIB_OBJ) $(LIB_REQ_BINS) $(STANDALONE_C_BINS) $(ROOT_C_BINS) $(SRC_GO_BINS) $(ROOT_GO_BINS)

# Create bin directory if it doesn't exist
$(BIN_DIR):
	powershell.exe -Command "if (!(Test-Path $(BIN_DIR))) { New-Item -ItemType Directory -Path $(BIN_DIR) }"

# Compile the library object
$(LIB_OBJ): $(LIB_SRC) $(LIB_HDR) | $(BIN_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Rule for files that REQUIRE libdnn.o
$(LIB_REQ_BINS): $(BIN_DIR)/%.exe: $(SRC_C_DIR)/%.c $(LIB_OBJ)
	$(CC) $(CFLAGS) $< $(LIB_OBJ) -o $@ $(LDFLAGS)

# Rule for STANDALONE files in src/c
$(STANDALONE_C_BINS): $(BIN_DIR)/%.exe: $(SRC_C_DIR)/%.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# Rule for ROOT C files
$(ROOT_C_BINS): $(BIN_DIR)/%.exe: %.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# Rule for Go files in src/go
$(SRC_GO_BINS): $(BIN_DIR)/%_go.exe: $(SRC_GO_DIR)/%.go
	$(GOC) $(GOFLAGS) $@ $<

# Rule for root Go files
$(ROOT_GO_BINS): $(BIN_DIR)/%_root.exe: %.go
	$(GOC) $(GOFLAGS) $@ $<

# Clean target
clean:
	powershell.exe -Command "if (Test-Path $(BIN_DIR)/*.exe) { Remove-Item $(BIN_DIR)/*.exe -Force }"
	powershell.exe -Command "if (Test-Path $(LIB_OBJ)) { Remove-Item $(LIB_OBJ) -Force }"

# Help target
help:
	@echo "Available targets:"
	@echo "  all     : Build all C and Go programs"
	@echo "  clean   : Remove all compiled binaries"
	@echo "  help    : Show this help message"
