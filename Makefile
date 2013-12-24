CC     = gcc
CFLAGS = -O3
EFLAGS =  
EFILE  = EMGMM
LIBS   = -lm -lgsl -lgslcblas -pthread
OBJS   = EMGMM.o 

$(EFILE) : $(OBJS)
	@echo "linking..."
	$(CC) $(EFLAGS) -o $(EFILE) $(OBJS) $(LIBS)

$(OBJS) : EMGMM.c EMGMM.h
	$(CC) $(CFLAGS) -c $*.c 

clean:
	rm -rf *.o EMGMM
