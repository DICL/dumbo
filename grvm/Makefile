.PHONY: all clean test

JSRC = $(shell find ptxcmp/src -name *.java)
JSRC += $(shell find examples -name *.java)
JSRC += $(shell find sbsvm/org/sbsvm/ -name *.java)
CPS= lib/jikesrvm3.1.4.jar

all: bin/jitptx.jar bin/libsbsvm.so

bin/jitptx.jar: $(JSRC) Makefile
	mkdir -p classes bin
	javac $(JFLAGS) -d classes -classpath $(CPS) -sourcepath ptxcmp/src:examples  -source 6 -target 6 $(JSRC)
	jar cf $@ -C bin .

bin/libsbsvm.so: Makefile
	mkdir -p bin
	$(MAKE) -C sbsvm
	cp sbsvm/libsbsvm.so bin/

clean:
	$(MAKE) -C sbsvm clean
	rm -rf bin/jitptx.jar bin/libsbsvm.so classes

test:  bin/jitptx.jar bin/libsbsvm.so
	PTXJIT_VERBOSE=1 rvm -Djava.library.path=bin -classpath bin/jitptx.jar:classes DMMStream

