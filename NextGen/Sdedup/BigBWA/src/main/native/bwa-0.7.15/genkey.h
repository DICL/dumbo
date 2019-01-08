/**
 * gotta use stdint so we can use the same type in C++/C
 * could make the example a bit more complicated with some
 * #defines to fix but it's simpler this way.
 */
#include <stdint.h>
#ifdef __cplusplus
typedef uint64_t UINT64;
typedef uint32_t UINT32;
typedef int32_t   INT32;
typedef INT32 pos_t; 
typedef UINT64 sgn_t;
#endif

#ifdef __cplusplus
extern "C" 
{
#endif
typedef struct splitLine splitLine_t;
typedef uint64_t UINT64;
typedef UINT64 sgn_t;

void*  getFoo( int32_t a );
void   destroyFoo( void *foo );
void   printString( void *foo );
splitLine_t * s_line2SplitLine(char * vline);
void * s_makeState();
void s_makeKey(splitLine_t * line, void * vstate);
void unsplitSplitLine(splitLine_t * line);
splitLine_t * s_splitNKey(char * vline, void * vstate);
void deleteSplitLine(splitLine_t * line);
void s_setKey(char * vSeqID, int vSeqNum, void * state);
sgn_t calcSig(splitLine_t * first, splitLine_t * second);
#ifdef __cplusplus
}
#endif


