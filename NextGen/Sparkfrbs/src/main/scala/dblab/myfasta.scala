//package dblab
//import scala.io.Source
//import scala.sys.process._
//
///**
//  * Created by hb-laptop on 2018-01-25.
//  */
//object myfasta {
//  def main(args: Array[String]): Unit = {
//    if(args.length < 2) {
//      println("usage vcfdir outdir")
//      System.exit(0)
//    }
//    val vcfDir = args(0)
//    val outDir = args(1)
//
//    val res = s"ls -t ${vcfDir}".!!
//    val tmp1 = res.split("\n").filter(x => x.endsWith("out")).map(x => x.split("\\s++")).map(_(8))
//    tmp1.foreach{ file =>
//      for (line <- Source.fromFile(file).getLines) {
//        if(!line.startsWith("#")) {
//          scala.tools.nsc.io.File(outDir).appendAll(line+"\n")
//        }
//      }
//    }
//  }
//}
