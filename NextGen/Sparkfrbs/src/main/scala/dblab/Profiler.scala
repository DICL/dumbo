package dblab
import org.apache.spark.rdd.RDD
import scala.collection.mutable.ArrayBuffer
/*
 * Copyright (c) 2016, Mincheol Shin
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this list of conditions
 *   and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
 *   and the following disclaimer in the documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse
 *   or promote products derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 * USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
 * WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
  * Created by hb-laptop on 2018-01-26.
  */
class Profiler {
  val timer_id = 0
  val timers : ArrayBuffer[Long] = new ArrayBuffer[Long]()
  val timer_labels : ArrayBuffer[String] = new ArrayBuffer[String]()

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
      val ss = (timer_labels(i) + ": " + (timers(i)-timers(0)))
      println(ss)
    }
  }
}
