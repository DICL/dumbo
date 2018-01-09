#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <mpi.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <stdlib.h>

#define FILENAME_LEN	256
#define LINE_MAX	1024
#define STRIPE_SIZE	8388608

int partFileClone(int src, off_t s_offset, int dst ,off_t t_offset, size_t len);
ssize_t forceWrite(int target, void *buf, int len);
ssize_t forceRead(int target, void *buf, int len);

int main(int argc, char *argv[]) 
{
	int rank, size;
	char fileName[FILENAME_LEN];
	unsigned char header[12],tail[28];
	unsigned char eofBlock[28] = {0x1f,0x8b,0x08,0x04,0,0,0,0,0,0xff,0x06,0,0x42,0x43,0x02,0,0x1b,0,0x03,0,0,0,0,0,0,0,0,0};
	char *prefix, *outFileName;
	int inFd, outFd;
	off_t s_off, t_off;
	long len = 0, end = 0;
	int err = 0, errsum = 0, ret = 0, i, cur;
	size_t xlen = 0;
	ssize_t headerSize = -1;
	char *xbuf;

	if( argc < 2 ) {
		printf("Usage: mpirun -np <split count> -hostfile <hostfile> ./merge_bam <prefix> <output bam file>\n");
		return -1;
	}
	prefix = argv[1];
	outFileName = argv[2];


	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);


	// open inputfile in parallel
	sprintf(fileName,"%s%d.bam\0",prefix,rank);
	inFd = open(fileName, O_RDONLY);
//	inFd = open(prefix, O_RDONLY);	//to be deleted

	if( inFd == -1 ) {
		perror(fileName);
		err = 1;
		goto check;
	}

	// get the length of input file
	len = lseek(inFd,0L,SEEK_END);

	// TODO:1)file size check more than 28
	// 		2)check tail 
	//      3)gzip header check, first 12 bytes magic number
	//      4)read extra len from gzip header and read contents
	//      5)find extra BAM header, and get header block length
	if( len < 28 ) {
		err = 2;
		goto check;
	}

	lseek(inFd, -28,SEEK_END);
	if( forceRead(inFd, tail, 28) == -1 ) {
		err = 3;
		goto check;
	}
	for(i=0; i < 28 ;i++) {
		if( tail[i] != eofBlock[i] ) {
			err = 6;
			goto check;
		}
	}

	lseek(inFd, 0, SEEK_SET);
	if( forceRead(inFd, header, 12) == -1 ) {
		err = 3;
		goto check;
	}
	if( header[0] != 31 || header[1] != 139 ) {
		err = 4;
		goto check;
	}

	printf("[%d] header = ",rank);
	for(i = 0 ; i < 12 ; i++) {
		printf("%d ",(size_t)header[i]);
	}
	printf("\n");

	xlen = header[11] * 256 + header[10];
//	xlen = (size_t)header[10];
	printf("[%d] xlen = %d\n",rank,xlen); 
	if( xlen > 0 ) {
		xbuf = malloc(xlen);

		if( forceRead(inFd, xbuf, xlen) == -1 ) {
			err = 5;
			goto check;
		}
		cur = 0;
		while( cur+4 <= xlen ) {
			if( xbuf[cur] == 66 && xbuf[cur+1] == 67 && xbuf[cur+2] == 2 && xbuf[cur+3] == 0 ) {
				headerSize = (ssize_t)xbuf[cur+5] *256 + (ssize_t)xbuf[cur+4] + 1;
				break;
			}
			cur += 2 + (int)xbuf[cur+3] << 8 + (int)xbuf[cur+2];
		}
	}
	printf("[%d] headerSize = %d\n",rank,headerSize); 
	if( headerSize == -1 ) {
		err = 7;
		goto check;
	}

	if( rank > 0 ) {
		s_off = headerSize;
		len -= s_off;
	}
	if( rank < (size - 1) ) {
		len -= 28;
	}

check:
	if (err > 0) {
		fprintf(stderr, "[%d] invalid input file %s, err code = %d\n",rank,fileName,err);
	}
	MPI_Allreduce(&err, &errsum, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD);
	if (errsum > 0) {
		MPI_Finalize();
		return -1;
	}

    MPI_Scan(&len, &end, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
	t_off = (off_t)(end - len);
	
	outFd = open(outFileName, O_WRONLY | O_CREAT | O_TRUNC, 0644);
	lockf(outFd, F_ULOCK, 0);

	printf("[%d/%d] s_off = %ld , t_off = %ld, len = %d\n",rank,size,s_off,t_off,len);

//	goto close;

	// truncate the file size to the maximum.
	if( rank == (size - 1) ) {
		if( ftruncate(outFd, (off_t)end) != 0 ) {
			perror("Error on truncate output file :");
			goto close;
		}
//		printf("Merged file size : %ld\n", end); 
	}
	MPI_Barrier(MPI_COMM_WORLD);

	ret = partFileClone(inFd, s_off, outFd, t_off, len);

close:
	close(inFd);
	close(outFd);

	MPI_Finalize();
	return ret;
}

int partFileClone(int src, off_t s_offset, int dst ,off_t t_offset, size_t len)
{
	if( lseek(src, s_offset, SEEK_SET) < 0 ) {
		perror("leek src in partFileClone");
		return -1;
	}
	if( lseek(dst, t_offset, SEEK_SET) < 0 ) {
		perror("leek dst in partFileClone");
		return -1;
	}

	void *buf = malloc(STRIPE_SIZE);
	int readSize = 0, write = 0;

	while( len > 0 ) {
		readSize = ( len < STRIPE_SIZE ) ? len : STRIPE_SIZE;
		readSize = read(src, buf, readSize);
		if( readSize == -1) {
			if (errno == EINTR)
				continue;
			perror("read in partFileClone");
			free(buf);
			return -1;
		}
		write = forceWrite(dst, buf, readSize);
		if ( write < 0 ) {
			perror("write in partFileClone");
			free(buf);
			return -1;
		}
		len -= readSize;
	}
	free(buf);
	fsync(dst);
	return len;
}

ssize_t forceRead(int fd, void *buf, int len)
{
	int ret;
	while( len > 0 && (ret = read (fd, buf, len)) != 0) {
		if (ret == -1) {
			if (errno == EINTR)
				continue;
			perror("forceRead");
			return -1;
		}
		len -= ret;
		buf += ret;
	}
	return len;
}

ssize_t forceWrite(int fd, void *buf, int len)
{
	int ret;
	while( len > 0 && (ret = write (fd, buf, len)) != 0) {
		if (ret == -1) {
			if (errno == EINTR)
				continue;
			perror("forceWrite");
			return -1;
		}
		len -= ret;
		buf += ret;
	}
	return len;
}
