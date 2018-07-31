#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <mpi.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>

#define FILENAME_LEN	256
#define LINE_MAX	1024
#define STRIPE_SIZE	8388608

int main(int argc, char *argv[]) 
{
	int rank, size, nameLen, lineNum = 0;
	char fileName[FILENAME_LEN], line[LINE_MAX];
	char buffer[STRIPE_SIZE];
	char *prefix, *outFileName;
	FILE *inFile, *outFile;
	long writeSize, start, end, toWrite, remain, written;
	int outfd;

	if( argc < 2 ) {
		printf("Usage: mpirun -np <split count> -hostfile <hostfile> ./merge <prefix> <output sam file>\n");
		return -1;
	}
	prefix = argv[1];
	outFileName = argv[2];

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// open inputfile in parallel
	sprintf(fileName,"%s%d.sam\0",prefix,rank);
	inFile = fopen(fileName,"r");

	if( inFile == NULL ) {
		printf("The input file %s doesn't exist.\n", fileName);
		MPI_Finalize();
		return 0;
	}

	// get the length of input file
	fseek(inFile,0L,SEEK_END);
	writeSize = ftell(inFile);
	fseek(inFile,0L,SEEK_SET);

	// skip '@' started header lines excluding the first file. reduce the total writl size
	while( fgets(line, LINE_MAX - 1, inFile) != NULL ) {
		if( rank == 0 || line[0] != '@') {
	//	if( line[0] != '@') {
			break;
		}
		writeSize -= strlen(line);
	//	printf("[%d.%d][%d][%d]%s", rank, lineNum++, writeSize, strlen(line), line);
	}

	// calculate the start position on output file
	MPI_Scan(&writeSize, &end, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
	start = end - writeSize;

//	printf("%s %ld bytes (%ld ~ %ld)\n", fileName, writeSize, start, end);

	outFile = fopen(outFileName, "w");
	outfd = fileno(outFile);
	lockf(outfd, F_ULOCK, 0);
//	printf("fileName= %s outfd  = %d errno = %d\n",outFileName, outfd, errno);

	// truncate the file size to the maximum.
	if( rank == (size - 1) ) {
		if( ftruncate(outfd, (off_t)end) != 0 ) {
			perror("Error on truncate output file :");
			goto close;
		}
//		printf("Merged file size : %ld\n", end); 
	}
	MPI_Barrier(MPI_COMM_WORLD);

	// locate the file position to write
	if( fseek(outFile, start, SEEK_SET) != 0 ) {
		perror("Error on seek :");
		goto close;
	}

	//write first line
	fputs(line, outFile);
	toWrite = writeSize - strlen(line);

	// aligned write of remaining bytes 
	while(toWrite > 0 ) {
		remain = STRIPE_SIZE - (end - toWrite)%STRIPE_SIZE;
		remain = fread(buffer, 1, remain, inFile);
		written = 0;
		while(remain - written > 0) {
			written += fwrite(buffer + written, 1, remain - written, outFile);
		}
		toWrite -= remain;
	}
	//do {
	//	write(outfd, line, strlen(line));
	//	fputs(line, outFile);
	//} while( fgets(line, LINE_MAX - 1, inFile) != NULL ); 

	fsync(outfd);

close:
	fclose(inFile);
	fclose(outFile);

	MPI_Finalize();
	return 0;
}
