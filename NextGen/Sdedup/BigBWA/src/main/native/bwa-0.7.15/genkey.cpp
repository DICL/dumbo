#include <iostream>
#include <sstream>
#include <iomanip>
#include <stdint.h>

#include "genkey.h"
#include "malloc_wrap.h"
#include <stdlib.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#include <map>
struct splitLine
{
    // General fields for any split line.
    splitLine_t * next;
    char * buffer;
    int bufLen;
    size_t maxBufLen;
    char **fields;
    int numFields;
    int maxFields;
    //bool split;
    // Special SAM fields that we need to access as other than strings.
    // It this were a class, these would be in a subclass.
    int   flag;
    pos_t pos;
    int   seqNum;
    pos_t binPos;
    int   binNum;
    int   SQO;
    int   EQO;
    int   sclip;
    int   eclip;
    int   rapos;
    int   raLen;
    int   qaLen;
    //bool  CIGARprocessed;
    //bool  discordant;
    //bool  splitter;
    //bool  unmappedClipped;
};

#define HASHNODE_PAYLOAD_SIZE 3
struct hashTable;
typedef struct hashTable hashTable_t;
typedef hashTable_t sigSet_t;

struct less_str
{
   bool operator()(char const *a, char const *b) const
   {
      return strcmp(a, b) < 0;
   }
};

// This stores the map between sequence names and sequence numbers.
typedef std::map<const char *, int, less_str> seqMap_t;

inline void addSeq(seqMap_t * seqs, char * item, int val)
{
    (*seqs)[item] = val;
}
struct hashNode;
typedef struct hashNode hashNode_t;
struct hashNode
{
    hashNode_t * next;
    UINT64 values[HASHNODE_PAYLOAD_SIZE];
};

hashNode_t * getHashNode();
void disposeHashNode(hashNode_t * node);

///////////////////////////////////////////////////////////////////////////////
// Hash Table
///////////////////////////////////////////////////////////////////////////////

struct hashTable
{
    UINT64 * table;
    UINT32   size;
    UINT32   entries;
    //~hashTable();
};

hashTable_t * makeHashTable();
void deleteHashTable(hashTable_t * ht);
bool hashTableInsert(hashTable_t * ht, UINT64 value);
void hashTableInit(hashTable_t * ht, int size=0);
void freeHashTableNodes();
void unsplitSplitLine(splitLine_t * line);
struct state_struct
{
    char *         inputFileName;
    FILE *         inputFile;
    char *         outputFileName;
    FILE *         outputFile;
    FILE *         discordantFile;
    char *         discordantFileName;
    FILE *         splitterFile;
    char *         splitterFileName;
    FILE *         unmappedClippedFile;
    char *         unmappedClippedFileName;
    sigSet_t *     sigs;
    seqMap_t       seqs;
    UINT32 *       seqLens;
    UINT64 *       seqOffs;
    splitLine_t ** splitterArray;
    int            splitterArrayMaxSize;
    UINT32         sigArraySize;
    int            binCount;
    int            minNonOverlap;
    int            maxSplitCount;
    int            minIndelSize;
    int            maxUnmappedBases;
    int            minClip;
    int            unmappedFastq;
    bool           acceptDups;
    bool           excludeDups;
    bool           removeDups;
    bool           addMateTags;
    bool           compatMode;
    bool           ignoreUnmated;
    bool           quiet;

    int           count;
    UINT64        totalLen;
};
typedef struct state_struct state_t;

#define QNAME  0
#define FLAG   1
#define RNAME  2
#define POS    3
#define MAPQ   4
#define CIGAR  5
#define RNEXT  6
#define PNEXT  7
#define TLEN   8
#define SEQ    9
#define QUAL  10
#define TAGS  11


// Define SAM flag accessors.
#define MULTI_SEGS     0x1
#define FIRST_SEG      0x40
#define SECOND_SEG     0x80
inline bool checkFlag(splitLine_t * line, int bits) { return ((line->flag & bits) != 0); }

inline void setFlag(splitLine_t * line, int bits) { line->flag |= bits; }

inline bool isPaired(splitLine_t * line) { return checkFlag(line, MULTI_SEGS); }

inline bool isConcordant(splitLine_t * line) { return checkFlag(line, 0x2); }

inline bool isDiscordant(splitLine_t * line) { return !isConcordant(line); }

inline bool isUnmapped(splitLine_t * line) { return checkFlag(line, 0x4); }

inline bool isNextUnmapped(splitLine_t * line) { return checkFlag(line, 0x8); }

inline bool isNextMapped(splitLine_t * line) { return !isNextUnmapped(line); }

inline bool isMapped(splitLine_t * line) { return !isUnmapped(line); }

inline bool isReverseStrand(splitLine_t * line) { return checkFlag(line, 0x10); }

inline bool isForwardStrand(splitLine_t * line) { return !isReverseStrand(line); }

inline bool isFirstRead(splitLine_t * line) { return checkFlag(line, FIRST_SEG); }

inline bool isSecondRead(splitLine_t * line) { return checkFlag(line, SECOND_SEG); }

#define BIN_SHIFT 27 //bin window is 27 bits wide
#define BIN_MASK ((1 << 27)-1) //bin window is 27 bits wide
#define MAX_SEQUENCE_LENGTH 250 // Current illumina paired-end reads are at most 150 + 150

inline int padLength(int length)
{
    return length + (2 * MAX_SEQUENCE_LENGTH);
}

inline int padPos(int pos)
{
    return pos + MAX_SEQUENCE_LENGTH;
}
inline bool moreCigarOps(char *ptr)
{
    return (ptr[0] != 0);
}
inline int str2int (char * str)
{
    return strtol(str, NULL, 0);
}

// Need to change this if pos is unsigned.
inline pos_t str2pos (char * str)
{
    return strtol(str, NULL, 0);
}
void calcOffsets(splitLine_t * line);
splitLine_t * getSplitLine();
splitLine_t * makeSplitLine();
bool needSwap(splitLine_t * first, splitLine_t * second);
void swapPtrs(splitLine_t ** first, splitLine_t ** second);

int getSeqNum(splitLine_t * line, int field, state_t * state);

void splitSplitLine(splitLine_t * line, int maxSplits);

//void s_splitNKey(char * vline, void * vstate)
splitLine_t * s_splitNKey(char * vline, void * vstate)
{
  splitLine_t * line = (s_line2SplitLine(vline));
  s_makeKey(line, vstate);

  return line;
}

splitLine_t * s_line2SplitLine(char * vline)
{
  splitLine_t * sline = getSplitLine();  
  memcpy(sline->buffer, vline, (strlen(vline)));  
  sline->bufLen = (strlen(vline)+1);
  
  splitSplitLine(sline, 12);
  
  //printf("FLAG :: %s\n", sline->fields[FLAG]);
  //printf("DEBUG sline :: %s\n", sline->buffer);

  return sline;
}

void s_setKey(char * vSeqID, int vSeqNum, void * vstate)
{
  state_t * state = (state_t *)vstate;
  // make state => 아래 두 항목 추가
  int count = state->count;
  UINT64 totalLen = state->totalLen;
  
  // ex) @SQ     SN:chr1 LN:248956422
  // seqid  => chr1
  // seqnum => 248956422
  char * seqID = vSeqID;
  int seqNum = 0;

  UINT32 seqLen = (UINT32)padLength(vSeqNum);
  UINT64 seqOff = 0;

  seqNum = count;
  count += 1;

  state->count = count;

  //seqLen = (UINT32)padLength(str2int());
  seqOff = totalLen;
  totalLen += (UINT64)(seqLen+1);

  state->totalLen = totalLen;

  // Unless we are marking dups, we don't need to use sequence numbers.
  // grow seqLens and seqOffs arrays
  // printf("Key setted : %s No : %d", seqID, count);

  if(seqNum % 32768 == 1)
  {
    state->seqLens = (UINT32*)realloc(state->seqLens, (seqNum+32768)*sizeof(UINT32));
    state->seqOffs = (UINT64*)realloc(state->seqOffs, (seqNum+32768)*sizeof(UINT64));
  }

  state->seqs[strdup(seqID)] = seqNum;
  state->seqLens[seqNum] = seqLen;
  state->seqOffs[seqNum] = seqOff;
  //printf("DBG Seq : Num : %lld, Len : %lld, Off : %lld\n", seqNum, seqLen, seqOff);
}

void s_makeKey(splitLine_t * line, void * vstate)
{
  state_t * state = (state_t *)vstate;
  line->flag = strtol(line->fields[FLAG], NULL, 0);
  calcOffsets(line);
  UINT64 seqOff;

  if(strcmp(line->fields[2],"*") == 0) {
    line->seqNum = 0;
    UINT64 seqOff = 0;
  }
  else {
    line->seqNum = getSeqNum(line, 2, state);
    seqOff = state->seqOffs[line->seqNum];
  }


  //int seqOff = state->seqOffs[line->seqNum]; //genome relative position
  
  //int t1 = (seqOff + line->pos) >> BIN_SHIFT;
  //int t2 = (seqOff + line->pos) &  BIN_MASK;
  line->binNum = (seqOff + line->pos) >> BIN_SHIFT;
  line->binPos = (seqOff + line->pos) &  BIN_MASK;
  //printf("DEBUG seqoff : %d, pos : %ld, binpos: %ld \n", seqOff, line->pos, line->binPos);
}
splitLine_t * getSplitLine()
{
    splitLine_t * line;
    line = makeSplitLine();
    
    line->next = NULL;
    //line->CIGARprocessed = false;    
    //line->discordant = false;
    //line->splitter = false;
    //line->unmappedClipped = false;
    return line;
}

splitLine_t * makeSplitLine()
{
    splitLine_t * line = (splitLine_t *)malloc(sizeof(splitLine_t));
    line->bufLen = 0;
    line->maxBufLen = 2000;
    line->buffer = (char *)malloc(line->maxBufLen);
    line->numFields = 0;
    line->maxFields = 100;
    line->fields = (char **)malloc(line->maxFields * sizeof(char *));
    return line;
}

void splitSplitLine(splitLine_t * line, int maxSplits)
{
    line->numFields = 0;
    int fieldStart = 0;
    // replace the newline with a tab so that it works like the rest of the fields.
    line->buffer[line->bufLen-1] = '\t';

    //printf("SAM : ");
    for (int i=0; i<line->bufLen; ++i)
    {
	//printf("%c", line->buffer[i]);

        if (line->buffer[i] == '\t')
        {
            line->fields[line->numFields] = line->buffer + fieldStart;
            line->numFields += 1;
            if (line->numFields == maxSplits) break;
            line->buffer[i] = 0;
            // Get ready for the next iteration.
            fieldStart = i+1;
        }
    }
    // replace the tab at the end of the line with a null char to terminate the final string.
    line->buffer[line->bufLen-1] = 0;
    //line->split = true;
}

int parseNextInt(char **ptr)
{
    int num = 0;
    for (char curChar = (*ptr)[0]; curChar != 0; curChar = (++(*ptr))[0])
    {
        int digit = curChar - '0';
        if (digit >= 0 && digit <= 9) num = num*10 + digit;
        else break;
    }
    return num;
}

inline char parseNextOpCode(char **ptr)
{
    return ((*ptr)++)[0];
}

void calcOffsets(splitLine_t * line)
{
    //if (line->CIGARprocessed) return;
    char * cigar = line->fields[CIGAR];
    line->raLen = 0;
    line->qaLen = 0;
    line->sclip = 0;
    line->eclip = 0;
    bool first = true;
    while (moreCigarOps(cigar))
    {
        int opLen = parseNextInt(&cigar);
        char opCode = parseNextOpCode(&cigar);
        if      (opCode == 'M' || opCode == '=' || opCode == 'X')
        {
            line->raLen += opLen;
            line->qaLen += opLen;
            first = false;
        }
        else if (opCode == 'S' || opCode == 'H')
        {
            if (first) line->sclip += opLen;
            else       line->eclip += opLen;
        }
        else if (opCode == 'D' || opCode == 'N')
        {
            line->raLen += opLen;
        }
        else if (opCode == 'I')
        {
            line->qaLen += opLen;
        }
        else
        {
/*
            fprintf(stderr, "Unknown opcode '%c' in CIGAR string: '%s'\n", opCode, line->fields[CIGAR]);
	    for (int i=0; i<line->bufLen; ++i) printf("%c", line->buffer[i]);
	    printf("\n");
*/
        }
    }
    line->rapos = strtol(line->fields[POS], NULL, 0);
    if (isForwardStrand(line))
    {
        line->pos = line->rapos - line->sclip;
        line->SQO = line->sclip;
        line->EQO = line->sclip + line->qaLen - 1;
    }
    else
    {
        line->pos = line->rapos + line->raLen + line->eclip - 1;
        line->SQO = line->eclip;
        line->EQO = line->eclip + line->qaLen - 1;
    }
    // Need to pad the pos in case it is negative
    line->pos = padPos(line->pos);
    // Let's not calculate these again for this line.
    // line->CIGARprocessed = true;
}

void * s_makeState ()
{
    state_t * s = new state_t();
    s->inputFile = stdin;
    s->inputFileName = (char *)"stdin";
    s->outputFile = stdout;
    s->outputFileName = (char *)"stdout";
    s->discordantFile = NULL;
    s->discordantFileName = (char *)"";
    s->splitterFile = NULL;
    s->splitterFileName = (char *)"";
    s->unmappedClippedFile = NULL;
    s->unmappedClippedFileName = (char *)"";
    s->sigs = NULL;
    s->minNonOverlap = 20;
    s->maxSplitCount = 2;
    s->minIndelSize = 50;
    s->maxUnmappedBases = 50;
    s->minClip = 20;
    s->acceptDups = false;
    s->excludeDups = false;
    s->removeDups = false;
    s->addMateTags = false;
    s->compatMode = false;
    s->ignoreUnmated = false;
    s->quiet = false;
    // Start this as -1 to indicate we don't know yet.
    // Once we are outputting our first line, we will decide.
    s->unmappedFastq = -1;
    // Used as a temporary location for ptrs to splitter for sort routine.
    s->splitterArrayMaxSize = 1000;
    s->splitterArray = (splitLine_t **)(malloc(s->splitterArrayMaxSize * sizeof(splitLine_t *)));
   
    s->count = 1; 
    s->totalLen = 0;
    return (void *)s;
}

void unsplitSplitLine(splitLine_t * line)
{
    // First make sure we are still split.
    //if (!line->split) return;
    // First undo the splits.
    // We will undo the splits backwards from the next field to avoid having to calculate strlen each time.
    for (int i=1; i<line->numFields; ++i)
    {
        line->fields[i][-1] = '\t';
    }
    // Now put the newline back in.
    //line->buffer[line->bufLen] = '\0';
    line->buffer[line->bufLen-1] = '\0';
    // Mark as no longer split.
    //line->split = false;
}

sgn_t calcSig(splitLine_t * first, splitLine_t * second)
{
    if(isMapped(first) && isUnmapped(second) || isMapped(second) && isUnmapped(first))
    {
        swapPtrs(&first, &second);
        //printf("first line unmapped.\n");
        first->binPos = 0;
    }
    else if(needSwap(first, second)) swapPtrs(&first, &second);

    UINT64 t1 = first->binPos;
    UINT64 t2 = t1 << 32;

    UINT64 final = t2 | second->binPos;
    //printf("firstline : %s\t%s\t%s\t%s\n", first->buffer, first->fields[1], first->fields[2], first->fields[3]);
    //printf("secondline : %s\t%s\t%s\t%s\n", second->buffer, second->fields[1], second->fields[2], second->fields[3]);    
    //printf("first binpos: %llu, second binpos: %llu, final : %llu\n", first->binPos, second->binPos, final);
    return (sgn_t)final;
}


void deleteSplitLine(splitLine_t * line)
{
    free(line->buffer);
    free(line->fields);
    free(line);
}
int getSeqNum(splitLine_t * line, int field, state_t * state)
{
    return state->seqs.find(line->fields[field])->second;
}
bool needSwap(splitLine_t * first, splitLine_t * second)
{
    // Sort first by ref offset.
    if (first->pos > second->pos) return true;
    if (first->pos < second->pos) return false;
    // Now by seq number.
    if (first->seqNum > second->seqNum) return true;
    if (first->seqNum < second->seqNum) return false;
    // Now by strand.
    // If they are both the same strand, it makes no difference which on is first.
    if (isReverseStrand(first) == isReverseStrand(second)) return false;
    if (isReverseStrand(first) && isForwardStrand(second)) return true;
    return false;
}
void swapPtrs(splitLine_t ** first, splitLine_t ** second)
{
    splitLine_t * temp = *first;
    *first = *second;
    *second = temp;
}
