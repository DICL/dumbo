package V3

import java.io.File

import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import scala.sys.process._

object SMerger2 {
  def dirSlash(dir:String) = {
    if(dir.endsWith("/")) dir.substring(0, dir.length-1)
    else dir
  }
  def setCmdWithDir1(dirname:String, partname:String, samtools:String, rmap:mutable.HashMap[String, Array[String]], sambam:String="bam"):mutable.Iterable[String] = {
    println(s"serch partname : ${partname}")
    val rname = rmap.get(partname).get(0)
    val files1 = s"find $dirname -type f ! -size 0 -name $partname".!!.split("\n")

    val fileLen = files1.length
    val arr = ArrayBuffer[String]()
    var i = 0;
    val size = 5
    var loop = 1;
    var last = 0;
    var exitv = false

    if(fileLen > 5) {
      while(!exitv) {
        val lsize =
          if(size*loop < fileLen) size*loop
          else {
            exitv = true
            fileLen
          }
        val filex = files1.slice(last, lsize).mkString(" ")
        val cmd = s"$samtools merge -@ 4 -O ${sambam} $dirname/$partname$loop-$rname.merged $filex"
        //println(s"size info : ${fileLen}, slice from ${last} to ${lsize}")
        //println(files1.slice(last, lsize).mkString("\n"))
//        println(s"")

        last =  lsize
        loop += 1

        arr.+=(cmd)
      }
    }
    else {
      val files = files1.mkString(" ")
      val cmd = s"$samtools merge -@ 4 -O ${sambam} $dirname/$partname-$rname.merged $files"
      arr.+=(cmd)
    }

    arr
  }
  def setCmdWithDir2(rnameWithPath:String, files:String, samtools:String, sambam:String="bam") = {
    val cmd = s"$samtools merge -@ 10 -O ${sambam} $rnameWithPath.${sambam} $files"
    cmd
  }
  def main(args: Array[String]): Unit = {
    if(args.length < 3) {
      println("usage sparkfrbs [dirPath] [samtoolsPath] [sam or bam]")
      System.exit(1)
    }
    val dirPath = dirSlash(args(0))
    val samtools = args(1)
    val sambam = args(2)

    val tc = new TimeChecker

    val dirs =  s"ls $dirPath".!!.split("\\s+").filter(_.startsWith("stream"))

    val cmdArr = ArrayBuffer[String]()
    val fileInfo = mutable.HashMap[String, Int]()
    val rnameMap = mutable.HashMap[String, Array[String]]()

    dirs.foreach{dir =>
      val path = s"$dirPath/$dir"
      val proc1 = s"find $path -name part-* ! -size 0 -ls".!!.split("\n")

      proc1.foreach{file =>
        val spl = file.split("\\s+")
        val size = spl(6).toInt
        val fullname = spl(10).split("/")
        val fname = fullname(fullname.length-1)

        if(!fileInfo.isDefinedAt(fname)) fileInfo.put(fname, size)
        else {
          val old = fileInfo.get(fname).get
          val newSize = size+old
          fileInfo.update(fname, newSize)
        }
      }
    }

    val proc2 = s"find $dirPath -name part.info".!!.split("\n")
    proc2.foreach{info =>
      val filename = info

      for (line <- Source.fromFile(filename).getLines) {
        if(!line.isEmpty) {
          val spl = line.split("\\s+")
          val partname = spl(0)
          val rnames = spl(1).split(",")
          if(rnames.length > 1) {
            println(s"INFO :: ${partname} has more than 2 rname : ${rnames.mkString(", ")}")
          }
          rnames.foreach{ rname =>
            if(!rnameMap.isDefinedAt(partname)) {
              //println(s"map setted key : ${partname} value : ${rname}")
              rnameMap.+=((partname, Array(rname)))
            }
            else {
              val oldData = rnameMap.get(partname).get
              if(!oldData.contains(rname)) {
                val newData = oldData ++ Array(rname)
                rnameMap.update(partname, newData)
              }
            }
          }
        }
      }
    }


    //    val uPath = s"$dirPath/unmapped"
    //    println(s"Upath : $uPath")
    //    println(s"cmd : find $uPath -name part-* ! -size 0 -ls")
    //    val proc3 = s"find $uPath -name part-* ! -size 0 -ls".!!.split("\n")
    //    proc3.foreach{ file =>
    //      val spl = file.split("\\s+")
    //      val size = spl(6).toInt
    //      val fullname = spl(10)
    //      println(s"cmd : cat $fullname >> $uPath/header")
    //      val cmd = (s"cat $fullname" #>> new File(s"$uPath/header")).!!
    //    }
    //    s"cp $uPath/header $dirPath/unMapped.sam".!!

    var i = 0
    var toggle = true

    val fsize = fileInfo.size


    while(i < fsize) {
      val cmd =
        if(toggle) {
          fileInfo.maxBy(_._2)
        }
        else {
          fileInfo.minBy(_._2)
        }
      fileInfo.remove(cmd._1)

      val partName = cmd._1

      val res = setCmdWithDir1(dirPath, partName, samtools, rnameMap, sambam)

      res.foreach(x => cmdArr.+=(x))
//      cmdArr.+=(res)
      toggle = !toggle
      i = i+1
    }

    val sparkConf = new SparkConf().setAppName("SMerger")
    val sc = new SparkContext(sparkConf)

    //println(cmdArr.mkString("\n"))

    val cmdList = sc.makeRDD(cmdArr)

    cmdList.repartition(1024)



    tc.checkTime("start")
    println("Merge step 1 start ...")
    val act = cmdList.mapPartitionsWithIndex{(idx, itr) =>
      import scala.sys.process._

      itr.foreach{cmd =>
        cmd.!
      }
      itr
    }
    act.count()

    println("Merge step 1 done ...")



    println("Merge step 2 start ...")
//    println("step2")
    val proc4 = (s"ls $dirPath" #| "grep merged").!!.split("\n")

    var map3 = mutable.Map[String, Array[String]]()
    proc4.foreach{fname =>
      val rname = fname.split("-")(2).dropRight(7)
      val path = s"$dirPath/$fname"

      if(map3.isDefinedAt(rname)) {
        val old = map3.get(rname).get
        val newVal = (old ++ Array(path))
        map3 = map3.updated(rname, newVal)
      }
      else {
        map3.put(rname, Array(path))
      }
    }

    val mergeSeq = map3.filter(_._2.length > 1).toSeq
    val moveSeq = map3.filter(_._2.length == 1).toSeq

    val cmdList2 = mergeSeq.map{ x =>
      val rname = x._1
      val files = x._2.mkString(" ")
      val rnameWithPath = s"$dirPath/$rname"
      val res = setCmdWithDir2(rnameWithPath, files, samtools, sambam)
      res
    }

    val cmdList3 = moveSeq.foreach{ x =>
      val rname = x._1
      val file = x._2.mkString(" ")
      val rnameWithPath = s"$dirPath/$rname"
      val cmd = s"mv $file $rnameWithPath.${sambam}"
      //println(cmd)
      cmd.!
    }
    println("Merge step 2 done ...")
    println("Merge step 3 start ...")

    val mergeList = sc.makeRDD(cmdList2).repartition(1024)
    val act2 = mergeList.mapPartitionsWithIndex{(idx, itr) =>
      import scala.sys.process._
      itr.foreach{cmd =>
        cmd.!
      }
      itr
    }
    act2.count()

    println("Merge step 3 done ...")


    tc.checkTime("end")
    tc.printElapsedTime()
  }
}

