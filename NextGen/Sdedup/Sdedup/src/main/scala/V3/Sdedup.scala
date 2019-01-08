package V3


import java.io._
import java.net.Socket
import java.nio.charset.StandardCharsets
import java.util.Date

import htsjdk.samtools.SAMFileHeader
import org.apache.spark.{HashPartitioner, RangePartitioner, SparkConf}
import org.apache.spark.internal.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.apache.spark.streaming.receiver.Receiver
import org.zeromq.ZMQ
import org.zeromq.ZMQ.{Context, Socket}
import java.time.{LocalDate, LocalTime}

import org.apache.hadoop.mapred.lib.HashPartitioner
import org.apache.spark
import org.apache.spark.streaming.zeromq.ZeroMQUtils

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

object Sdedup {
  //  def getPercent(a:Float, )
  def getString(sam:SAM4) = {
    val arr:Array[String] = Array(sam.qname, sam.flag, sam.rname, sam.pos.toString) ++ sam.others
    val res = s"${arr.mkString("\t")}"
    res
  }
  def main(args: Array[String]) {
    if (args.length < 3) {
      System.err.println("Usage: APP <ZMQ BIND URL> <DSTREAM INTERVAL SIZE> <PARTITION SIZE> <FILE OUT PATH>")
      System.exit(1)
    }
    val zmqBindURL = args(0)
    val dstreamIntevalSize = args(1).toInt
    val defaultPartitionSize = args(2).toInt
    val outPath =
      if (args(3).endsWith("/")) args(3).slice(0, args(3).length-1)
      else args(3)


    val sparkConf = new SparkConf().setAppName("Sdedup")
    sparkConf.set("spark.shuffle.blockTransferService", "nio")
    val ssc = new StreamingContext(sparkConf, Seconds(dstreamIntevalSize))
    val sc = ssc.sparkContext
    val lines = ZeroMQUtils.createTextStream(
      ssc, zmqBindURL, true, Seq("foo".getBytes), StorageLevel.MEMORY_AND_DISK
    ).repartition(defaultPartitionSize)

    var streamCnt = 0
    var dedupCnt = 0L
    var streamStart = false

    val hashPtnr = new spark.HashPartitioner(defaultPartitionSize)
    var keyData = sc.emptyRDD[(Long, (Int, Array[SAM4]))].partitionBy(hashPtnr)
    val source = lines.flatMap(_.split("\n"))
    val headerStream = source.filter(_.startsWith("@")).flatMap(_.split("\n"))

    val dedupCntAccu = sc.longAccumulator

    headerStream.foreachRDD{rdd =>
      if(!streamStart & !rdd.isEmpty()) {
        val str = rdd.collect()
        ppp.setDic(str)
      }
    }

    val readsStream = source.filter(x => !(x.startsWith("@") || x.startsWith("##"))).map(_.split("XXQQ")).map { spl =>
      //      val key = spl(0).toLong
      val data = spl.slice(1, spl.length)
      val res = data.map(x => String2SAM.parseToSAM(x))
      res
    }

    var mappedCnt = 0L
    var umappedCnt= 0L
    var totalCnt = 0L

    val endSignalStream = source.filter(_.startsWith("##"))


    readsStream.foreachRDD{ prdd =>
      if(!prdd.isEmpty()) {
        //val cnt = prdd.count()
        //totalCnt += cnt
        //        mappedCnt += cnt
        val rdd = prdd

        val broadcastMap = ssc.sparkContext.broadcast(ppp.getDict)
        val broadcastHeader = ssc.sparkContext.broadcast(ppp.getHead)

        if(!streamStart) streamStart = true

        val pairedSam = rdd.map{x =>
          val paired = x.map{ sam =>
            val bcMap = broadcastMap.value
            val idxedRname = bcMap.get(sam.rname).getOrElse(0)*1000
            val dividedPos = (sam.pos/100000000).toInt
            val key =idxedRname + dividedPos
            val res = (key, sam)
            res
          }
          paired
        }.flatMap(x => x).partitionBy(hashPtnr)
        
        pairedSam.persist(StorageLevel.MEMORY_AND_DISK)

        val formattedStreamCnt = "%05d".format(streamCnt)
        val makePartitionInfoFile = pairedSam.map(x => x._2.rname).mapPartitionsWithIndex{(idx , iter) =>
          if(!iter.isEmpty){
            val rnames = iter.toSeq.distinct.mkString(",").replace("*","unmapped")
            val partitionNo = "%05d".format(idx)
            val streamDir = new File(s"${outPath}/stream-${formattedStreamCnt}")
            streamDir.mkdirs()
            val partInfoFile = new FileWriter(s"${outPath}/stream-${formattedStreamCnt}/part.info", true)
            partInfoFile.append(s"part-${partitionNo}\t${rnames}\n")
            partInfoFile.close()
          }
          iter
        }
        makePartitionInfoFile.count() //do action

        val sortedSam = pairedSam.map(_._2)
          .mapPartitions{ samIter =>
            val head = Seq(broadcastHeader.value.mkString("\n")).iterator
            val sorted = samIter.toSeq.sortBy(_.pos).map(x => getString(x)).iterator
            val res = if(sorted.isEmpty) sorted
            else head++sorted
            res
          }
        sortedSam.saveAsTextFile(s"file://${outPath}/stream-${formattedStreamCnt}")
        pairedSam.unpersist()
        streamCnt += 1
      }
      println(s"total received : ${totalCnt}")
      val currentTime = LocalTime.now()
    }
    endSignalStream.foreachRDD{ rdd =>
      if(!rdd.isEmpty()) {
        println(s"End Signal Received at ${LocalTime.now()}")
        ssc.stop()
      }
    }

    ssc.start()
    ssc.awaitTermination()
  }
}

