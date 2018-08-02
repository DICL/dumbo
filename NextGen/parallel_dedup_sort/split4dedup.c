#include<sys/socket.h>
#include<sys/stat.h>
#include<arpa/inet.h>
#include<netinet/in.h>
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

#ifdef MPI_CODED
#include<mpi.h>
#endif

#define BUF_PAGES	1000000	//524288 

#define MAX_NUM_CHR		128	
#define	RNAME_LEN		32
#define QNAME_LEN		256


#define IS_UNMAPPED(x)		((x) & 0x4)
#define IS_SMAPPED(x)		((x) & 0x8)
#define IS_FIRSTSEG(x)		((x) & 0x40)
#define IS_LASTSEG(x)		((x) & 0x80)
#define IS_SECONDARY(x)		((x) & 0x100)
#define IS_SUPPLEMENTARY(x)	((x) & 0x800)
#define IS_DUPLICATE(x)		((x) & 0x400)
#define IS_PRIMARY(x)		(((x) & 0x900) == 0)
#define IS_REVERSE(x)		((x) & 0x10)

struct chr_entry {
	char name[RNAME_LEN];
	size_t offset;
	UT_hash_handle hh;
};

struct chr_entry *chrmap = NULL;

#ifdef SINGLE_SHUFFLE
struct toent {
	int index;
	struct toent *next;
};

struct toent *tepool = NULL;

struct toent *insert_toent(struct toent *ori, int idx) {
	struct toent *res = ori;
	while( res ) {
		if( idx == res->index ) return ori;		//if index already exists, return original list
		res = res->next;
	}

	// create new to-entry and put it at the head of the list
	if( tepool ) {
		res = tepool;
		tepool = tepool->next;
	} else {
		res = malloc(sizeof(struct toent));
	}
	res->index = idx;
	res->next = ori;
	return res;
}
#endif

void forceWrite(int fd, const void* buf, size_t len);

int main(int argc, char **argv) 
{
	int rank = 0, comsize = 1;
	char infname[256] = {0,};

	int i;

	if( argc < 3 ) {
		printf("Usage : ./a.out <input sam file | prefix | -> [hostfile]\n \"-\" means to read from standard input \n");
		return -1;
	}

#ifdef MPI_CODED
	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD, &comsize);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	sprintf(infname, "%s%d.sam\0", argv[1], rank);
#else 
	sprintf(infname, "%s\0", argv[1]);
#endif
	
	//allocate bulk buffer which is aligned to the page size
	char *buffer;
	size_t PAGE_SIZE = sysconf(_SC_PAGESIZE);
	if( posix_memalign((void **)&buffer, PAGE_SIZE, PAGE_SIZE * BUF_PAGES) != 0 ) {
		fprintf(stderr, "Not enough memory\n");
#ifdef MPI_CODED
		MPI_Finalize();
#endif
		return -errno;
	}
	char *lastvp = buffer + PAGE_SIZE*(BUF_PAGES-1);
	char *curp = buffer;
	size_t readsize = 0;

	//opening input file
	FILE *inf;
	if( infname[0] == '-' ) {
		inf = stdin; 
	} else {
		inf = fopen(infname,"r");
		if( inf == NULL ) {
			fprintf(stderr,"error on opening file %s\n",infname);
#ifdef MPI_CODED
			MPI_Finalize();
#endif
			return -errno;
		}
	}

	//reading header from input file
	size_t header_len = 0;
	struct chr_entry *newent;
	size_t ln;
	size_t offset = 0;
	
	while( fgets(curp, PAGE_SIZE, inf) ) {
		readsize = strlen(curp);

		if( *curp == '@' ) {	//header
			if( *(curp+1) == 'S' && *(curp+2) == 'Q') {	//read ref. Sequence dictionary
				newent = calloc(1, sizeof(struct chr_entry));
				newent->offset = offset;
				sscanf(curp,"@SQ SN:%s LN:%lu\n", newent->name, &ln);
				offset += (ln+1);
				HASH_ADD_KEYPTR(hh, chrmap, newent->name, RNAME_LEN, newent);
			}
			header_len += readsize;
			curp += readsize;
		} else break;
	}


	// open connection to sinks and send header to them
	int num_sinks = 0;				// the number of total worker processes
	int outfds[128];				// array of out sink socket fds
	FILE *hf = fopen( argv[2], "r");
	char linebuf[32], sinkaddrbuf[16];	//buffer for reading host list file 
	struct sockaddr_in addr;

	while( fgets(linebuf, 32, hf) ) {
		sscanf(linebuf, "%s",sinkaddrbuf);

		bzero(&addr, sizeof(addr));
		addr.sin_family = AF_INET;
		addr.sin_addr.s_addr = inet_addr(sinkaddrbuf);
		addr.sin_port = htons(1814+1+num_sinks);

		// open connection to each sink
		outfds[num_sinks] = socket(AF_INET, SOCK_STREAM, 0);
		if( connect(outfds[num_sinks], (struct sockaddr *)&addr, sizeof(struct sockaddr_in) ) < 0 ) {
			fprintf(stderr, "Error on connecting to %s:%d\n",sinkaddrbuf, 1814+num_sinks);
#ifdef MPI_CODED
			MPI_Finalize();
#endif
			return -errno;
		};

		// send header
		forceWrite(outfds[num_sinks], &header_len, sizeof(header_len));
		forceWrite(outfds[num_sinks], buffer, header_len);
		num_sinks++;
	}
	fclose(hf);

	size_t width = (offset + num_sinks) / num_sinks; // position width that each sink receives 

	//string to store the qname of previous completly got read-group.
	char *pname = calloc(QNAME_LEN,sizeof(char)); 
	char *swptmp;
	//variables to store alignment data got from each line
	char *qname = calloc(QNAME_LEN,sizeof(char));
	int flag;
	char rname[RNAME_LEN];
	int pos; 
	size_t gpos;	//global position of a alignment
	
	char *rgp = NULL;	//pointer to current read-group with the same QNAME
	
	char *cpt, *spt, *cigar;	//temporary pointer to parsing alignment line

	size_t rg_len = 0;
	int fd;
	int sclip, raLen, eclip, cigar_len, first;	//for handle CIGAR string
	char cigar_code;
#ifdef SINGLE_SHUFFLE
	struct toent *tolist = NULL, *cur_te, *tmp_te;
#else 
	size_t mingpos;
#endif 

	//loop while there is remaining byted to read 
	//current read line is got from header-reading part
	while( readsize > 0 ) {
		//parse line to get value
		// parse QNAME
		cpt = curp;
		spt = qname;
		while( *cpt != '\t' ) *spt++ = *cpt++;
		*spt = '\0';
		cpt++;		//skip tab character
		// pasre FLAG
		flag = 0;
		while( *cpt != '\t' ) {
			flag = flag * 10 + *cpt - '0';
			cpt++;
		}
		cpt++;		// skip tab
		// parse RNAME
		memset(rname, 0x00, RNAME_LEN);
		spt = rname;
		while( *cpt != '\t' ) *spt++ = *cpt++;
		cpt++;		//skip tab
		// parse POS
		pos = 0;
		while( *cpt != '\t' ) {
			pos = pos * 10 + *cpt - '0';
			cpt++;
		}
		cpt++;		//skip tab
		while( *cpt != '\t' ) cpt++;	//skip  MAPQ
		cigar = cpt + 1;		// pointer to CIGAR

		if( strcmp(qname, pname) != 0 ) {	//new read-group !!
			if( rgp ) {	//if this read is not the first read, send completed read-group to proper sink(s)
			#ifdef SINGLE_SHUFFLE
				cur_te = tolist;
				while( cur_te ) {	//for all sockets in tolist, write read-group data
					fd = outfds[cur_te->index];
					forceWrite(fd, &rg_len, sizeof(rg_len));
					forceWrite(fd, rgp, rg_len);
					
					//release toent to tepool
					tmp_te = cur_te->next;	
					cur_te->next = tepool;
					tepool = cur_te;
					cur_te = tmp_te;
				}
			#else
				// write this read-group to minimum global position among primary alignments
				fd = outfds[mingpos/width];
				forceWrite(fd, &rg_len, sizeof(rg_len));
				forceWrite(fd, rgp, rg_len);
			#endif
			}
			// reset data for new read-group
			rgp = curp;
			rg_len = 0;

			swptmp = pname;
			pname = qname;
			qname = swptmp;

		#ifdef SINGLE_SHUFFLE
			tolist = NULL;		//for single shuffle
		#else
			mingpos = offset;	//for double shuffle
		#endif
		}
		// increase read-group length
		rg_len += readsize;

		// calculate global aligned position in whole reference genome
	#ifdef SINGLE_SHUFFLE
		if( *rname == '*' || pos == 0 ) {
			gpos = offset;
		} else {
			HASH_FIND(hh, chrmap, rname, RNAME_LEN, newent);
			gpos = newent->offset + pos;
		}
		tolist = insert_toent(tolist, gpos/width);	//for single shuffle

	#else
		// update minimul global position of primary alignment
		//for double shuffle
		if( IS_PRIMARY(flag) && !IS_UNMAPPED(flag) ) {
			HASH_FIND(hh, chrmap, rname, RNAME_LEN, newent);
			gpos = newent->offset + pos;

			// recalculate position according to CIGAR offset
			if( IS_REVERSE(flag) ) {
				raLen = 0;
				eclip = 0;
				cigar_len = 0;
				cigar_code = *cigar;
				first = 1;
				do {
					if( cigar_code >= '0' && cigar_code <= '9' ) {
						cigar_len = cigar_len * 10 + (cigar_code - '0');
					} else if (cigar_code == 'M' || cigar_code == '=' || cigar_code == 'X' ) {
						raLen += cigar_len;
						cigar_len = 0;
						first = 0;
					} else if ( cigar_code == 'S' || cigar_code == 'H' ) {
						if ( first == 0 ) eclip += cigar_len;
						cigar_len = 0;
					} else if ( cigar_code == 'D' || cigar_code =='N' ) {
						raLen += cigar_len;
						cigar_len = 0;
					} else if ( cigar_code != 'I' && cigar_code != 'P' ) {
						fprintf(stderr, "Unknown opcode '%c' in CIGAR string\n", cigar_code);
						exit(-1);
					}
					cigar++;
					cigar_code = *cigar;
				} while( cigar_code != '\t' ) ;

				gpos += (raLen + eclip - 1);
			} else {
				sclip = 0;
				while( *cigar >= '0' && *cigar <= '9' ) {	//is DIGIT
					sclip = sclip * 10 + (*cigar - '0');
					cigar++;
				}
				if( *cigar == 'S' ) gpos -= sclip;
			}

			mingpos = (gpos < mingpos) ? gpos : mingpos;
		}
	#endif

		//handle for current line is done, make cursur pointer move forward.
		curp += readsize;

		if( curp > lastvp ) {	 //if cursor is reached to the bottom of buffer
			//move current read-group data to the top of buffer
			memcpy(buffer, rgp, rg_len);
			//and reset rgp and curp
			rgp = buffer;
			curp = buffer + rg_len;
		}

		if( fgets(curp, PAGE_SIZE, inf) == NULL ) {	//EOF reached 
			//write last read-group and stop the loop
		#ifdef SINGLE_SHUFFLE
			while( tolist ) {	//for all sockets in tolist, write read-group data
				fd = outfds[tolist->index];
				forceWrite(fd, &rg_len, sizeof(rg_len));
				forceWrite(fd, rgp, rg_len);
				tolist = tolist->next;
			}
		#else 
			fd = outfds[mingpos/width];
			forceWrite(fd, &rg_len, sizeof(rg_len));
			forceWrite(fd, rgp, rg_len);
		#endif
			break;
		}
		readsize = strlen(curp);
	}

	// close sockets to all sinks 
	for( i = 0 ; i < num_sinks ; i++ ) {
		close(outfds[i]);
	}

#ifdef MPI_CODED
	MPI_Finalize();
#endif

	return 0;
}

void forceWrite(int fd, const void* buf, size_t len) {
	ssize_t size;
	while( len > 0 && (size = write(fd, buf, len)) != 0 ) {
		if( size < 0 ) {
			if( errno == EINTR ) 
				continue;
			perror("write error");
			exit(-1);
		}
		len -= size;
		buf += size;
	}
}
