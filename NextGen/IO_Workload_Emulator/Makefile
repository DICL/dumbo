CC=gcc
LDFLAGS = -lpthread

.PHONY: clean all

all: main.o
	$(CC) -o replayer main.o $(LDFLAGS)

clean:
	rm -f replayer main.o
