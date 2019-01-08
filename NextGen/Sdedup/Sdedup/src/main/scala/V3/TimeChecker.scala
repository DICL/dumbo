package V3

import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer

class TimeChecker {
  private val timer_id = 0
  private val timers : ArrayBuffer[Long] = new ArrayBuffer[Long]()
  private val timer_labels : ArrayBuffer[String] = new ArrayBuffer[String]()

  def checkTime(): Long = {
    timers += System.currentTimeMillis
    timer_labels += "Timer "+ timers.length.toString
    timers(timers.length - 1)
  }

  def checkTime(label :String): Long = {
    timers += System.currentTimeMillis
    timer_labels += label
    timers(timers.length - 1)
  }

  def checkTimeWithCollect(rdd: RDD[_]): Long = {
    rdd.collect()
    timers += System.currentTimeMillis
    timer_labels += "Timer "+timers.length.toString
    timers(timers.length - 1)
  }

  def checkTimeWithCollect(rdd: RDD[_], label :String): Long = {
    rdd.collect()
    timers += System.currentTimeMillis
    timer_labels += label
    timers(timers.length - 1)
  }

  def printInterval(): Unit ={
    println("=== Intervals between timers ===")
    for(i <- 1 until timers.length){
      println(timer_labels(i) + ": " + (timers(i)-timers(i-1)))
    }
  }

  def printElapsedTime(): Unit ={
    println("=== Elapsed Time until checkTime ===")
    for(i <- 1 until timers.length){
      println(timer_labels(i) + ": " + (timers(i)-timers(0)))
    }
  }
}
