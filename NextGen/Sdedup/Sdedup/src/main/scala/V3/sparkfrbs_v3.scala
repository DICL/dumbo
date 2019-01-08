package V3


import org.apache.spark.{HashPartitioner, SparkConf, SparkContext, TaskContext}
import scala.sys.process._
import java.io.File
import scala.collection.mutable.ArrayBuffer
import scala.io.Source
object sparkfrbs_v3 {
  def main(args : Array[String]): Unit = {
    if(args.length < 7) {
      println("usage : FreebayesPath SamtoolsPath ReferencefilePath BamDir1 BamDir2 VcfOutputDirPath PartitionNumb RegionSize")
    }

    val sparkConf = new SparkConf().setAppName("sparkfrbs")
    val sc = new SparkContext(sparkConf)

    val FreebayesPath = args(0)
    val SamtoolsPath  = args(1)
    val Ref = args(2) //ReferencefilePath
    val WorkDir = args(3) //Sam or Bam file
    val WorkDir2 = args(4) //Sam or Bam file

    val OutputDirPath = args(5).endsWith("/") match {
      case true => args(5)
      case false => args(5)+"/"
    }

    val ExecutorNumb = args(6).toInt
    val regionSize = args(7).toInt
    //fasta!
    case class myfasta(name:String, idx:Int, start:Long, end:Long)

    val cmdList = ArrayBuffer[myfasta]()
    val faiFile = Ref+".fai"
    val helper = new TimeChecker()
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
    val fileList = (s"find $WorkDir -name *.bam -type f -ls").!!.split("\n")
    val fileList2 = (s"find $WorkDir2 -name *.bam -type f -ls").!!.split("\n").map{file =>
      val fullpath = file.split("\\s+")(10)
      val rname = fullpath.substring(0, fullpath.length-4).split("/").last
      (rname, fullpath)
    }


//    val fileListx = fileList++fileList2


    val cmdList2 = fileList.map{ file =>
      val fullpath = file.split("\\s+")(10)
      val rname = fullpath.substring(0, fullpath.length-4).split("/").last
      val fullpath2 = fileList2.filter(_._1 == rname).map(_._2).mkString(" ")
      val fullpath3 = (fullpath +" "+ fullpath2).trim()

//      val cmd = cmdList.filter(_.name==rname).map(x => (fullpath, x))
      val cmd = cmdList.filter(_.name==rname).map(x => (fullpath3, x))
      println(fullpath3)
      cmd
    }.flatMap(x => x)



    println(s"run freebayes ...")
    println(s"total number of commands : "+cmdList2.length)

    val res = sc.parallelize(cmdList2, ExecutorNumb*10).mapPartitionsWithIndex{ (idx, iter) =>
      val pi = TaskContext.getPartitionId()
      iter.foreach { data =>
        val filename = data._1
        val fasta = data._2
        val vcfout = (s"$OutputDirPath${fasta.idx.formatted("%06d")}.${pi.formatted("%010d")}.${fasta.name}.${fasta.start.formatted("%013d")}-${fasta.end.formatted("%013d")}.vcf")
        if(!(new File(s"$filename.bai")).exists()) s"${SamtoolsPath} index $filename".!!
        val outputdir = (new File(s"$OutputDirPath"))
        if(!outputdir.exists())  outputdir.mkdirs()

        ((s"$FreebayesPath -v $vcfout --pooled-discrete --pooled-continuous --min-alternate-fraction 0.1 --genotype-qualities " +
          s"--report-genotype-likelihood-max --allele-balance-priors-off -f $Ref $filename -r ${fasta.name}:${fasta.start}-${fasta.end}")).!
      }
      iter
    }
    res.count()

    helper.checkTime()
    helper.printElapsedTime()
  }
}



