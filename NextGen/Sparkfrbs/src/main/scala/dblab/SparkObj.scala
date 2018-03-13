package dblab

import org.apache.spark.SparkConf
import org.apache.spark.streaming.{Seconds, StreamingContext}

object SparkObj {
  val sparkConf = new SparkConf().setAppName("app")
  val ssc = new StreamingContext(sparkConf, Seconds(1))
  val sc = ssc.sparkContext
}
