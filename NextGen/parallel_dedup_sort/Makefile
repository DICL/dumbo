CC = gcc
MPICC = mpicc

CFLAGS += -Wall 
INCLUDES = 
LIBS = 

PROGS = s4d s4dmpi samstat sink d2s

.c.o:
	$(CC) -c $(CLFAGS) $(INCLUDES) $< -o $@

all:$(PROGS) 

s4d: split4dedup.o
	$(CC) $(INCLUDES) $(LIBS) -o $@ $^

s4dmpi: split4dedup.c
	$(MPICC) -o $@ -DMPI_CODED $^

samtest: samtest.o
	$(CC) $(INCLUDES) $(LIBS) -o $@ $^

sink: sink.o
	$(CC) $(INCLUDES) $(LIBS) -o $@ $^

d2s: d2s.o
	$(CC) $(INCLUDES) $(LIBS) -o $@ $^

clean:
	rm -f $(PROGS) *.o
