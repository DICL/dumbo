#include <stdint.h>

typedef uint64_t UINT64;
typedef UINT64 sgn_t;
typedef uint64_t UINT64;
typedef uint32_t UINT32;
typedef int32_t   INT32;
typedef INT32 pos_t;
typedef UINT64 sgn_t;

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
    // bool split;
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

typedef struct splitLine splitLine_t;

