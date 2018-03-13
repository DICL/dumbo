//package dblab
//
//import java.io.File
//import scala.io.Source
//import scala.sys.process._
//
//object mns {
//
//  val vcfDir = ""
//  val outDir = "out"
//  val samDir = "samtools"
//
//  val res = s"ls -t ${vcfDir}".!!
//
//  var Cnt = 0
//
//  var Header = scala.collection.mutable.ArrayBuffer[String]()
//  var Body = scala.collection.mutable.ArrayBuffer[String]()
//
//  val tmp1 = res.split("\n").filter(x => x.endsWith("sam"))
//  tmp1.foreach{ file =>
//    for (line <- Source.fromFile(file).getLines) {
//      if(Cnt == 0)
//        scala.tools.nsc.io.File(outDir+".sam").appendAll(line+"\n")
//      else if(!line.startsWith("@"))
//        scala.tools.nsc.io.File(outDir+".sam").appendAll(line+"\n")
//    }
//    Cnt = Cnt + 1
//  }
////| samtools view  -Sb - | samtools sort - sorted && samtools index sorted.bam
//  (s"cat $outDir.sam" #| s"$samDir view -Sb" #> new File(s"$outDir.bam")).!
//  (s"$samDir sort -o $outDir.sorted.bam $outDir.bam").!
//  (s"$samDir index $outDir.sorted.bam").!
//}
