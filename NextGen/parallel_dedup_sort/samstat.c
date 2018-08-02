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
#include"uthash.h"

#define LINE_BUF_SIZE	0x10000
#define MAX_NUM_CHR		256	

#define IS_UNMAPPED(x)		((x) & 0x4)
#define IS_SMAPPED(x)		((x) & 0x8)
#define IS_FIRSTSEG(x)		((x) & 0x40)
#define IS_LASTSEG(x)		((x) & 0x80)
#define IS_SECONDARY(x)		((x) & 0x100)
#define IS_SUPPLEMENTARY(x)	((x) & 0x800)
#define IS_DUPLICATE(x)		((x) & 0x400)
#define IS_PRIMARY(x)		(((x) & 0x900) == 0)

struct chr_entry {
	char name[32];
	size_t offset;
	UT_hash_handle hh;
};

struct chr_entry *chrmap = NULL;

int main(int argc, char **argv) 
{
	if(argc < 2 ) {
		printf("Usage : ./samstat [input sam file]\n");
		exit(1);
	}

	FILE *insamf = fopen(argv[1], "r");
	char lbuf[LINE_BUF_SIZE];
	
	struct chr_entry *newent;
	size_t ln;
	size_t offset = 0;

	size_t total = 0, primary = 0, dup = 0, secondary = 0, supple = 0, umap = 0, smap = 0, cunmap = 0, drgs = 0;

	char *qname = calloc(256,sizeof(char));
	char *pname = calloc(256,sizeof(char));
	char *ctmp;
	char r;
	int flag, prevflag = 0;
	int pos; 

	char *startp, *curp;
	
	int first = 1;

	while( fgets(lbuf,LINE_BUF_SIZE,insamf) ) {
		if( lbuf[0] == '@' ) {	//header
			if( lbuf[1] == 'S' && lbuf[2] == 'Q') {
				newent = calloc(1, sizeof(struct chr_entry));
				newent->offset = offset;
				sscanf(lbuf,"@SQ SN:%s LN:%lu\n", newent->name, &ln);
				offset += ln;
				HASH_ADD_KEYPTR(hh, chrmap, newent->name, 32, newent);
			}
		} else {		//alignment read
			curp = lbuf;
			startp = qname;
			while( *curp!= '\t' ) *startp++ = *curp++;
			curp++;
			flag = 0;
			while( *curp != '\t' ) {
				flag = flag * 10 + *curp - '0';
				curp++;
			}
			curp++;
			r = *curp;
			while( *curp != '\t' ) curp++;
			curp++;
			pos = 0;
			while( *curp != '\t' ) {
				pos = pos * 10 + *curp - '0';
				curp++;
			}
			curp++;

			if( r == '*' ) {
				cunmap++;
			}

			if( strcmp(qname, pname) != 0 ) {
				if( first == 0 ) first = 0;
				else if( IS_DUPLICATE(prevflag) ) drgs++;
				ctmp = pname;
				pname = qname;
				qname = ctmp;
			}
			prevflag = flag;

			if( IS_PRIMARY(flag) ) primary++;
			if( IS_SECONDARY(flag) ) secondary++;
			if( IS_SUPPLEMENTARY(flag) ) supple++;
			if( IS_UNMAPPED(flag) ) umap++;
			if( IS_DUPLICATE(flag) ) dup++;
			if( IS_SMAPPED(flag) ) smap++;
			total++;
		}
	}

	if( IS_DUPLICATE(prevflag) ) drgs++;

	printf("total bases = %lu\n",offset);
	printf("PRI= %lu ,SEC= %lu ,CHIMERIC= %lu ,UNMAPPED= %lu (%.2f%%) ,SINGLEMAPPED= %lu (%.2f%%) on total %lu reads\n",primary, secondary, supple, umap,umap*100.0/(double)total, smap, smap*100.0/(double)total, total);
	printf("duplicate marked = %lu(%.2f%%)\n",dup,dup*100.0/(double)total);
	printf("completly unmapped read = %lu\n",cunmap);
	printf("the number of deduped read-groups = %lu\n",drgs);

	fclose(insamf);
	return 0;
}
