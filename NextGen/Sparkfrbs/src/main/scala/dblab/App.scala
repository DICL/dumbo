//package dblab
//
//import java.io.FileWriter
//
//
//import org.apache.spark.{HashPartitioner, SparkConf, SparkContext}
//object App {
//  def main(args : Array[String]): Unit = {
//    if(args.length < 5) {
//      println("usage : FreebayesPath ReferencefilePath SamfilePath OutputDitPath ExecutorNumb")
//      println("usage : /home/dblab/sparkfrbs/freebayes /home/dblab/sparkfrbs/datafile/reference/hg38.fa " +
//        "/home/dblab/sparkfrbs/datafile/kitty.sam /home/dblab/sparkfrbs/outputfile/ 10")
//    }
//
//    val sparkConf = new SparkConf().setAppName("app")
//    val sc = new SparkContext(sparkConf)
//
//    val FreebayesPath = args(0)
//    val ReferencefilePath = args(1)
//    val SamfilePath = args(2)
//    val OutputDitPath = args(3)
//    val ExecutorNumb = args(4).toInt
//
//    val loadedSamfile = sc.textFile(SamfilePath)
//    val header = loadedSamfile.filter(x => x.startsWith("@"))
//    val body = loadedSamfile.filter(x => !x.startsWith("@"))
//
//    val partitioned = body.map{x => (x.split("\t")(2),x)}.repartitionAndSortWithinPartitions(new HashPartitioner(ExecutorNumb)).persist()
//
//    val cHeader = header.collect().mkString("\n")
//    val samString = partitioned.map(_._2)
//
//    val res = samString
//    val res0 = res.mapPartitionsWithIndex
//    { (idx, iter) =>
//      import scala.sys.process._
//      import java.io.File
//
//      val mergedString = (List(cHeader) ++ iter.toList).mkString("\n")
//      val writeFilePath = OutputDitPath+"part"+idx+".out"
//      val file = new File(writeFilePath)
//      val fw = new FileWriter(file)
//      fw.write(mergedString)
//      fw.close()
//
//
//      //      ("/home/dblab/sparkfrbs/freebayes -f
//      // /home/dblab/sparkfrbs/datafile/reference/hg38.fa /home/dblab/sparkfrbs/datafile/kitty.sam" #>
//      // new File(s"/home/dblab/sparkfrbs/outputfile/freebayes.test$idx.out")).!
//
//      ((s"$FreebayesPath -f $ReferencefilePath $writeFilePath") #> new File(s"$OutputDitPath/fb#$idx.out")).!
//
//      iter.toIterator
//    }
//    println(res0.first())
//
//  }
//}
