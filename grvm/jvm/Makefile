.PHONY: all rvm libs

JIKESRVM_REVISION=133772

JDK_SOURCE_URL=http://hg.openjdk.java.net/jdk8u/jdk8u/jdk/archive/bd24ee3a9a04.zip
JDK_SOURCE_URL_UTIL= $(JDK_SOURCE_URL)/src/share/classes/java/util
JDK_SOURCE_URL_LANG= $(JDK_SOURCE_URL)/src/share/classes/java/lang

JDK_SOURCE_DIR=jdk8lib/jdk-bd24ee3a9a04/src/share/classes
JDK_SOURCE_FILES = $(JDK_SOURCE_DIR)/java/lang/AutoCloseable.java	\
$(JDK_SOURCE_DIR)/java/util/function/BiConsumer.java			\
$(JDK_SOURCE_DIR)/java/util/function/BiFunction.java			\
$(JDK_SOURCE_DIR)/java/util/function/BiPredicate.java			\
$(JDK_SOURCE_DIR)/java/util/function/BinaryOperator.java		\
$(JDK_SOURCE_DIR)/java/util/function/BooleanSupplier.java		\
$(JDK_SOURCE_DIR)/java/util/function/Consumer.java			\
$(JDK_SOURCE_DIR)/java/util/function/DoubleBinaryOperator.java		\
$(JDK_SOURCE_DIR)/java/util/function/DoubleConsumer.java		\
$(JDK_SOURCE_DIR)/java/util/function/DoubleFunction.java		\
$(JDK_SOURCE_DIR)/java/util/function/DoublePredicate.java		\
$(JDK_SOURCE_DIR)/java/util/function/DoubleSupplier.java		\
$(JDK_SOURCE_DIR)/java/util/function/DoubleToIntFunction.java		\
$(JDK_SOURCE_DIR)/java/util/function/DoubleToLongFunction.java		\
$(JDK_SOURCE_DIR)/java/util/function/DoubleUnaryOperator.java		\
$(JDK_SOURCE_DIR)/java/util/function/Function.java			\
$(JDK_SOURCE_DIR)/java/util/function/IntBinaryOperator.java		\
$(JDK_SOURCE_DIR)/java/util/function/IntFunction.java			\
$(JDK_SOURCE_DIR)/java/util/function/IntPredicate.java			\
$(JDK_SOURCE_DIR)/java/util/function/IntSupplier.java			\
$(JDK_SOURCE_DIR)/java/util/function/IntToDoubleFunction.java		\
$(JDK_SOURCE_DIR)/java/util/function/IntToLongFunction.java		\
$(JDK_SOURCE_DIR)/java/util/function/IntUnaryOperator.java		\
$(JDK_SOURCE_DIR)/java/util/function/LongBinaryOperator.java		\
$(JDK_SOURCE_DIR)/java/util/function/LongConsumer.java			\
$(JDK_SOURCE_DIR)/java/util/function/LongFunction.java			\
$(JDK_SOURCE_DIR)/java/util/function/LongPredicate.java			\
$(JDK_SOURCE_DIR)/java/util/function/LongSupplier.java			\
$(JDK_SOURCE_DIR)/java/util/function/LongToDoubleFunction.java		\
$(JDK_SOURCE_DIR)/java/util/function/LongToIntFunction.java		\
$(JDK_SOURCE_DIR)/java/util/function/LongUnaryOperator.java		\
$(JDK_SOURCE_DIR)/java/util/function/ObjDoubleConsumer.java		\
$(JDK_SOURCE_DIR)/java/util/function/ObjIntConsumer.java		\
$(JDK_SOURCE_DIR)/java/util/function/ObjLongConsumer.java		\
$(JDK_SOURCE_DIR)/java/util/function/Predicate.java			\
$(JDK_SOURCE_DIR)/java/util/function/Supplier.java			\
$(JDK_SOURCE_DIR)/java/util/function/ToDoubleBiFunction.java		\
$(JDK_SOURCE_DIR)/java/util/function/ToDoubleFunction.java		\
$(JDK_SOURCE_DIR)/java/util/function/ToIntBiFunction.java		\
$(JDK_SOURCE_DIR)/java/util/function/ToIntFunction.java			\
$(JDK_SOURCE_DIR)/java/util/function/ToLongBiFunction.java		\
$(JDK_SOURCE_DIR)/java/util/function/ToLongFunction.java		\
$(JDK_SOURCE_DIR)/java/util/function/UnaryOperator.java			\
$(JDK_SOURCE_DIR)/java/util/function/package-info.java			\
$(JDK_SOURCE_DIR)/java/util/stream/AbstractShortCircuitTask.java	\
$(JDK_SOURCE_DIR)/java/util/stream/AbstractSpinedBuffer.java		\
$(JDK_SOURCE_DIR)/java/util/stream/AbstractTask.java			\
$(JDK_SOURCE_DIR)/java/util/stream/BaseStream.java			\
$(JDK_SOURCE_DIR)/java/util/stream/Collector.java			\
$(JDK_SOURCE_DIR)/java/util/stream/Collectors.java			\
$(JDK_SOURCE_DIR)/java/util/stream/DistinctOps.java			\
$(JDK_SOURCE_DIR)/java/util/stream/DoublePipeline.java			\
$(JDK_SOURCE_DIR)/java/util/stream/DoubleStream.java			\
$(JDK_SOURCE_DIR)/java/util/stream/FindOps.java				\
$(JDK_SOURCE_DIR)/java/util/stream/ForEachOps.java			\
$(JDK_SOURCE_DIR)/java/util/stream/IntStream.java			\
$(JDK_SOURCE_DIR)/java/util/stream/LongPipeline.java			\
$(JDK_SOURCE_DIR)/java/util/stream/LongStream.java			\
$(JDK_SOURCE_DIR)/java/util/stream/MatchOps.java			\
$(JDK_SOURCE_DIR)/java/util/stream/Node.java				\
$(JDK_SOURCE_DIR)/java/util/stream/Nodes.java				\
$(JDK_SOURCE_DIR)/java/util/stream/PipelineHelper.java			\
$(JDK_SOURCE_DIR)/java/util/stream/ReduceOps.java			\
$(JDK_SOURCE_DIR)/java/util/stream/ReferencePipeline.java		\
$(JDK_SOURCE_DIR)/java/util/stream/Sink.java				\
$(JDK_SOURCE_DIR)/java/util/stream/SliceOps.java			\
$(JDK_SOURCE_DIR)/java/util/stream/SortedOps.java			\
$(JDK_SOURCE_DIR)/java/util/stream/SpinedBuffer.java			\
$(JDK_SOURCE_DIR)/java/util/stream/Stream.java				\
$(JDK_SOURCE_DIR)/java/util/stream/StreamShape.java			\
$(JDK_SOURCE_DIR)/java/util/stream/StreamSpliterators.java		\
$(JDK_SOURCE_DIR)/java/util/stream/StreamSupport.java			\
$(JDK_SOURCE_DIR)/java/util/stream/TerminalOp.java			\
$(JDK_SOURCE_DIR)/java/util/stream/TerminalSink.java			\
$(JDK_SOURCE_DIR)/java/util/stream/package-info.java

MOD_SOURCE_FILES = modified/java/util/function/IntConsumer.java	\
modified/java/util/stream/AbstractPipeline.java			\
modified/java/util/stream/IntPipeline.java			\
modified/java/util/stream/StreamOpFlag.java			\
modified/java/util/stream/Streams.java				\
modified/java/util/stream/Tripwire.java				\
modified/java/util/Spliterator.java

all: rvm libs

rvm: jikesrvm/dist/production_x86_64_m64-linux/JikesRVM

../lib/JikesRVM: jikesrvm/dist/production_x86_64_m64-linux/JikesRVM
	cp -f jikesrvm/dist/production_x86_64_m64-linux/rvm ../lib
	cp -f jikesrvm/dist/production_x86_64_m64-linux/JikesRVM ../lib
	cp -f jikesrvm/dist/production_x86_64_m64-linux/lib*.so ../lib
	cp -f jikesrvm/dist/production_x86_64_m64-linux/*.jar ../lib
	cp -f jikesrvm/dist/production_x86_64_m64-linux/*.image ../lib
	cp -f jikesrvm/dist/production_x86_64_m64-linux/*.map ../lib

jikesrvm/dist/production_x86_64_m64-linux/JikesRVM: jikesrvm
	( cd jikesrvm;ant -Dhost.name=x86_64_m64-linux -Dconfig.name=production;)

jikesrvm:
	mkdir -p $@
	printf "checking out Jikes RVM %s\n" $(JIKESRVM_REVISION)
	svn co -r $(JIKESRVM_REVISION) https://github.com/JikesRVM/JikesRVM.git/trunk jikesrvm
	echo "applying patch"
	(cd jikesrvm; patch -p0 < ../patches/jikesrvm.r$(JIKESRVM_REVISION).patch;)

gen-patch: patches/jikesrvm.r$(JIKESRVM_REVISION).patch

patches/jikesrvm.r$(JIKESRVM_REVISION).patch: jikesrvm
	(cd jikesrvm; LANG=en svn diff;) > patches/jikesrvm.r$(JIKESRVM_REVISION).patch

libs: jdk8rt.jar

jdk8rt.jar: $(JDK_SOURCE_FILES) $(MOD_SOURCE_FILES)
	mkdir -p classes
	javac -d classes $(JDK_SOURCE_FILES) $(MOD_SOURCE_FILES)
	jar cf $@ -C classes .
	rm -rf classes

$(JDK_SOURCE_FILES): jdk8lib.jar
	@if [ ! -e jdk8lib ];then mkdir -p jdk8lib;fi;
	@if [ ! -f $@ ];then (cd jdk8lib;jar xf ../jdk8lib.jar;);fi;

jdk8lib.jar: jdk8util.zip jdk8lang.zip
	mkdir -p ziptmp
	unzip -d ziptmp -o -u jdk8util.zip
	unzip -d ziptmp -o -u jdk8lang.zip
	jar cf $@ -C ziptmp  .
	rm -rf ziptmp

jdk8util.zip:
	wget $(JDK_SOURCE_URL_UTIL) -O $@

jdk8lang.zip:
	wget $(JDK_SOURCE_URL_LANG) -O $@

clean:
	rm -f jdk8rt.jar
	rm -rf jdk8lib

clobber: clean
	rm -f jdk8lib.jar
	rm -f jdk8util.zip
	rm -f jdk8lang.zip
	rm -rf jikesrvm
