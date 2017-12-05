package kr.ac.ut.csdb


import org.apache.kafka.common.serialization.StringDeserializer
import org.apache.spark.{HashPartitioner, SparkConf}
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.apache.spark.streaming.kafka010._
import org.apache.spark.streaming.kafka010.LocationStrategies.PreferConsistent
import org.apache.spark.streaming.kafka010.ConsumerStrategies.Subscribe
object  Sdedup {

  def main(args: Array[String]): Unit = {
    if (args.length < 4) {
      System.err.println("Usage : [Kafka_servers] [Kafka_topic] [Spark_workers] [mapping method (1-qname, 2-rname, 3-pos, 4-rname+pos)]")
      System.err.println("Example : 172.31.7.168:9092,172.31.10.122:9092,172.31.14.59:9092 head,read 172.31.5.87,172.31.11.122,172.31.12.146 1")
      System.exit(1)
    }

    def mappingMethod(x:Int) = (sam:SAM4) => x match {
      case 1 => sam.qname
      case 2 => sam.rname
      case 3 => sam.pos
      case 4 => sam.rname + sam.pos

      case _ => sam.pos
    }

    val BootstrapServers = args(0)
    val Topics = args(1).split(",")
    val Workers = args(2).split(",")
    val Mapping = mappingMethod(args(3).toInt)

    println("Kafka Bootstrap Servers : " + BootstrapServers)
    println("Kafka Topics : " + Topics.mkString(","))
    println("Spark Workers : " + Workers)
    println("Partitioning Method : " + Mapping)

    val sparkConf = new SparkConf().setAppName("Sdedup")
    val ssc = new StreamingContext(sparkConf, Seconds(5))

    val kafkaParams = Map[String, Object](
      "bootstrap.servers" -> BootstrapServers,
      "key.deserializer" -> classOf[StringDeserializer],
      "value.deserializer" -> classOf[StringDeserializer],
      "group.id" -> "use_a_separate_group_id_for_each_stream",
      "auto.offset.reset" -> "latest",
      "enable.auto.commit" -> (false: java.lang.Boolean)
    )

    val topics = Topics
    val stream = KafkaUtils.createDirectStream[String, String](
      ssc,
      PreferConsistent,
      Subscribe[String, String](topics, kafkaParams)
    )

    val servers = Workers

    val lines = stream.map(record => (record.key, record.value))
    val line = lines.map(x => x._2).flatMap(x => x.split("\n"))

    val header = line.filter(x => x.startsWith("@") || x.startsWith("""^"""))
    val read = line.filter(x => !x.startsWith("@") && !x.startsWith("""^"""))

    header.foreachRDD{x =>
      if(!x.isEmpty()) {
        println(s"HEAD  IN :: ${java.time.LocalTime.now.getMinute} : ${java.time.LocalTime.now.getSecond}")

        val str = x.collect().mkString("\n")
        servers.foreach(addr => (new socketNet(10007, str, addr)).run())


        println(s"HEAD OUT :: ${java.time.LocalTime.now.getMinute} : ${java.time.LocalTime.now.getSecond}\n")
      }
    }

    val patitioner = new HashPartitioner(50)
    read
      .map{x => String2SAM.parseToSAM(x)}
      .map(x => (x.qname, x.srcData))
      .reduceByKey(_+"\n"+_)
      .map(x => String2SAM.parseToSAM(x._2))
      .map(x => ((x.rname + x.pos), x.srcData))
      .foreachRDD{x =>
        if(!x.isEmpty()) {
          println(s"READ  IN :: ${java.time.LocalTime.now.getMinute} : ${java.time.LocalTime.now.getSecond}")

          val s = x.partitionBy(patitioner).map(x => x._2).persist()
          s.foreachPartition{x =>
            val str = x.mkString("\n")
            (new socketNet(10007, str)).run()
          }

          println(s"READ OUT :: ${java.time.LocalTime.now.getMinute} : ${java.time.LocalTime.now.getSecond}\n")
        }
      }
    ssc.start()
    ssc.awaitTermination()
  }
}
