CC = nvcc
FLAGS = -lm

SRC = imgblur.cu
TARGET = output

all: compile run clean

compile: $(SRC)
	$(CC) $(SRC) -o $(TARGET) $(FLAGS)

run: compile
	./$(TARGET)

clean:
	@rm -f $(TARGET)