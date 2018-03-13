package dblab

import java.io.{BufferedReader, InputStreamReader}
import java.net.{ServerSocket}
import org.apache.spark.streaming._

object mns2 {
  val ssc = SparkObj.ssc
  val sc = SparkObj.sc

  var Header = sc.emptyRDD[(String)]
  var Body = sc.emptyRDD[(String, Seq[String])]

  var path = "/home/dblab/sparkfrbs.out"

  def stop = {
    val tmp1 = Header.filter(x => x.startsWith("@SQ")).sortBy(x => x.split("\\s++")(1))
  .collect().foreach(str => scala.tools.nsc.io.File(path).appendAll(str+"\n"))
    val tmp2 = Header.filter(x => x.startsWith("@RG")).sortBy(x => x)
      .collect().foreach(str => scala.tools.nsc.io.File(path).appendAll(str+"\n"))
    val tmp3 = Header.filter(x => x.startsWith("@PG")).sortBy(x => x)
      .collect().foreach(str => scala.tools.nsc.io.File(path).appendAll(str+"\n"))

    Body.collect().foreach{record =>
      val sorted = record._2.sortBy{x =>
        val spl = x.split("\\s++")
        (spl(2)+spl(3))
      }
      val str = sorted.mkString("\n")
      scala.tools.nsc.io.File(path).appendAll(str+"\n")
    }

    ssc.stop()
  }
  class Handler(port: Int, ssc: StreamingContext) extends Runnable {
    println(s"Socket created! ${port}")
    val serverSocket = new ServerSocket(port)
    def run() {
      val socket = serverSocket.accept()
      val reader = new InputStreamReader(socket.getInputStream)
      val br = new BufferedReader(reader)
      if (br.readLine() == "kill") {
        stop
      }
      br.close();
    }
  }
  private def start = {
    val stream = ssc.textFileStream("file:///mnt").flatMap(_.split("\n"))

    stream.foreachRDD{ rdd =>
      if(!rdd.isEmpty()) {
        val header = rdd.filter(x => x.startsWith("@"))
        val body   = rdd.filter(x => !x.startsWith("@")).map{x => (x.split("\\s++")(0), Seq(x))}

        println(rdd.count())

        if (!header.isEmpty()) Header = Header.union(header).distinct().persist()
        if (!body.isEmpty()) {
          Body = Body.union(body).reduceByKey(_++_).sortByKey().persist()
        }
      }
    }
    ssc.start()
    val kill = new Thread(new Handler(19999, ssc)).start()
    ssc.awaitTermination()
  }

  def Go(Path:String) = {
    path = Path
    start
  }
}


