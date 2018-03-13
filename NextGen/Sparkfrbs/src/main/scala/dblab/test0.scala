package dblab

import java.io.FileWriter

import org.apache.spark.{HashPartitioner, SparkConf, SparkContext, TaskContext}

import scala.collection.mutable.ArrayBuffer
import scala.io.Source
object test0 {
  def main(args : Array[String]): Unit = {
    if(args.length < 6) {
      println("usage : FreebayesPath ReferencefilePath SamfilePath OutputDitPath ExecutorNumb RegionSize")
      println("usage : /home/dblab/sparkfrbs/freebayes /home/dblab/sparkfrbs/datafile/reference/hg38.fa " +
        "/home/dblab/sparkfrbs/datafile/kitty.sam /home/dblab/sparkfrbs/outputfile/ 10 100")
    }
    def Start = {
      val sc = SparkObj.sc

      val FreebayesPath = args(0)
      val Ref = args(1) //ReferencefilePath
      val SrcFile = args(2) //Sam or Bam file
      val OutputDirPath = args(3).endsWith("/") match {
        case true => args(3)
        case false => args(3)+"/"
      }

      val ExecutorNumb = args(4).toInt
      val regionSize = args(5).toInt
      //fasta!
      case class myfasta(name:String, idx:Int, start:Int, end:Int)

      var cmdList = new ArrayBuffer[myfasta]()

      val faiFile = Ref+".fai"

      val helper = new Profiler()
      helper.checkTime()

      var cnt = 0
      for (str <- Source.fromFile(faiFile).getLines) {
        var regionStart = 0
        val spl = str.split("\t")
        val chromName = spl(0)
        val chromLengh  = spl(1).toInt

        while(regionStart < chromLengh) {
          var start = regionStart
          var end = regionStart + regionSize
          if(end > chromLengh) end = chromLengh

          val fasta = myfasta(chromName, cnt, start, end)
          cmdList.+=(fasta)
          regionStart = end
          cnt = cnt + 1
        }
      }
      val res = sc.parallelize(cmdList, ExecutorNumb*10).foreachPartition{ (iter) =>
        import scala.sys.process._
        import java.io.File
        val pi = TaskContext.getPartitionId()
        iter.foreach { fasta =>
          ((s"$FreebayesPath -f $Ref -L $SrcFile -r ${fasta.name}:${fasta.start}-${fasta.end}")
            #> new File(s"$OutputDirPath${fasta.idx.formatted("%06d")}.${fasta.name}:${fasta.start.formatted("%013d")}-${fasta.end.formatted("%013d")}.$pi.out")).!
        }
      }

      helper.checkTime()
      helper.printElapsedTime()
    }
    Start
  }
}

