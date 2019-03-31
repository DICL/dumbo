base_dir=$(dirname $0)

if [ -z "$INCLUDE_TEST_JARS" ]; then
  INCLUDE_TEST_JARS=false
fi

# Exclude jars not necessary for running commands.
regex="(-(test|src|scaladoc|javadoc)\.jar|jar.asc)$"
should_include_file() {
  if [ "$INCLUDE_TEST_JARS" = true ]; then
    return 0
  fi
  file=$1
  if [ -z "$(echo "$file" | egrep "$regex")" ] ; then
    return 0
  else
    return 1
  fi
}
# Memory options
if [ -z "$KAFKA_HEAP_OPTS" ]; then
  KAFKA_HEAP_OPTS="-Xmx256M"
fi

# Which java to use
if [ -z "$JAVA_HOME" ]; then
  JAVA="java"
else
  JAVA="$JAVA_HOME/bin/java"
fi


# classpath addition for release
for file in $base_dir/libs/*;
do
  if should_include_file "$file"; then
    CLASSPATH="$CLASSPATH":"$file"
  fi
done

for file in $base_dir/*;
do
  if should_include_file "$file"; then
    CLASSPATH="$CLASSPATH":"$file"
  fi
done

if [ -z "$YARN" ]; then
  YARN="yarn"
fi

$JAVA $KAFKA_HEAP_OPTS -cp $CLASSPATH "org.kisti.moha.MOHA_KafkaStop"
sleep 5s
exec $YARN jar MOHA.jar org.kisti.moha.MOHA_KafkaStart -manager_memory 8120 -broker_memory 8120 -num_brokers 3 -jar MOHA.jar -kafka_tgz kafka_2.11-0.10.1.0.tgz

