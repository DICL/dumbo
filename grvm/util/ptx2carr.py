#!/usr/bin/python3
import sys
import os

varname=sys.argv[1]
ifile=sys.argv[2]
ofile=sys.argv[3]
ofileh=sys.argv[4]
#print("#%s" % varname)
flen=os.stat(ifile).st_size
print("%d bytes in %s" % (flen, ifile))
with open(ifile) as fi, open(ofile, 'w') as fo:
    lines=fi.readlines()
    print("const char %s[%d] = {" % (varname, flen), file=fo)
    for line in lines:
        chrseq=', '.join(map(lambda x: "'\\x%02x'" % ord(x), line))
        print("  //%s" % (line[:-1]), file=fo)
        print("  %s," % (chrseq),file=fo)
    print("};", file=fo)
    
with open(ofileh, 'w') as foh:
    print("extern const char %s[%d];" % (varname, flen), file=foh)
