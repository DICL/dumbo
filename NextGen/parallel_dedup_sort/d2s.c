#include<sys/stat.h>
#include<fcntl.h>
#include<sys/types.h>
#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<unistd.h>
#include<string.h>
#include<assert.h>
#include<errno.h>
#include"uthash.h"

#define MAX_NUM_CHR		128	
#define	RNAME_LEN		32


struct chr_entry {
	char name[RNAME_LEN];
	size_t offset;
	UT_hash_handle hh;
};

struct chr_entry *chrmap = NULL;

int main(int argc, char **argv) 
{
	int rank = 0, comsize = 1;
	char infname[256] = {0,};
	char fnbuf[1024];
	char *prefix;
	int i;

	if( argc < 4 ) {
		printf("Usage : ./a.out <rank> <comsize> <directory of tempfile>\n");
		return -1;
	}

	rank = atoi(argv[1]);
	comsize = atoi(argv[2]);
	prefix = argv[3];

	//allocate bulk buffer which is aligned to the page size
	char *buffer;
	size_t PAGE_SIZE = sysconf(_SC_PAGESIZE);
	if( posix_memalign((void **)&buffer, PAGE_SIZE, PAGE_SIZE) != 0 ) {
		fprintf(stderr, "Not enough memory\n");
		return -errno;
	}

	size_t readsize = 0;

	FILE *inf = stdin;

	if( prefix[strlen(prefix)-1] == '/' ) prefix[strlen(prefix)-1] = '\0';
	sprintf(fnbuf,"%s/fsplit_tmp.header.%05d",prefix,rank);
	FILE *outhf = fopen(fnbuf, "w");

	//reading header from input file
	struct chr_entry *newent;
	size_t ln;
	size_t offset = 0;
	
	while( fgets(buffer, PAGE_SIZE, inf) ) {
		if( *buffer == '@' ) {	//header
			if( *(buffer+1) == 'S' && *(buffer+2) == 'Q') {	//read ref. Sequence dictionary
				newent = calloc(1, sizeof(struct chr_entry));
				newent->offset = offset;
				sscanf(buffer,"@SQ SN:%s LN:%lu\n", newent->name, &ln);
				offset += (ln+1);
				HASH_ADD_KEYPTR(hh, chrmap, newent->name, RNAME_LEN, newent);
			}
			fputs(buffer, outhf);
		} else break;
	}
	fclose(outhf);

	FILE **outfs = (FILE **)malloc(comsize * sizeof(FILE *));
	for(i = 0 ; i < comsize ; i++) {
	//	sprintf(fnbuf,"%s/fsplit_tmp_%05d",prefix, i);
	//	mkdir(fnbuf ,0755);
		sprintf(fnbuf,"%s/fsplit_tmp_data.%05d.%05d",prefix, i, rank);
		outfs[i] = fopen(fnbuf,"w");
	}

	size_t width = (offset + comsize) / comsize; // position width that each sink receives 

	//variables to store alignment data got from each line
	char rname[RNAME_LEN];
	int pos; 
	size_t gpos;	//global position of a alignment
	
	char *cpt, *spt;	//temporary pointer to parsing alignment line

	//loop while there is remaining byted to read 
	//current read line is got from header-reading part
	do {
		//parse line to get value
		cpt = buffer;
		while( *cpt != '\t' ) *cpt++;	//skip QNAME
		cpt++;
		while( *cpt != '\t' ) *cpt++;	//skip flag
		cpt++;
		//read RNAME
		memset(rname, 0x00, RNAME_LEN);
		spt = rname;
		while( *cpt != '\t' ) *spt++ = *cpt++;
		cpt++;
		//read POS
		pos = 0;
		while( *cpt != '\t' ) {
			pos = pos * 10 + *cpt - '0';
			cpt++;
		}
		// calculate global aligned position in whole reference genome
		if( *rname == '*' || pos == 0 ) {
			gpos = offset;
		} else {
			HASH_FIND(hh, chrmap, rname, RNAME_LEN, newent);
			gpos = newent->offset + pos;
		}

		fputs(buffer, outfs[gpos/width]);
	} while ( fgets(buffer, PAGE_SIZE, inf) );

	for(i = 0 ; i < comsize ; i++) {
		fclose(outfs[i]);
	}

	return 0;
}
