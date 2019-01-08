package kr.ac.ut.csdb

case class SAM4(qname:String, flag:String, rname:String, pos:String, srcData:String)

object String2SAM {
  def parseToSAM(str:String) : SAM4 = {

    val srcData = str
    val spl = str.split("\t")

    if(spl.length > 4) {
      val qname = spl(0)
      val flag = spl(1)
      val rname = spl(2)
      val pos = spl(3)

      SAM4(qname, flag, rname, pos, srcData)
    }
    else {
      SAM4("qname", "flag", "rname", "pos", srcData)
    }
  }
}

