package V3

object ppp {
  var dict_seq = scala.collection.mutable.Map[String, Int]()
  var header = scala.collection.mutable.ArrayBuffer[String]()
  var lastv = 0L
  var rg = false
  var pg = false

  def setDic(lines:Array[String]) = {
    dict_seq.put("*", dict_seq.size)

    lines.foreach { line =>
      if(line.startsWith("@SQ")) {
        val spl = line.split("\\s++")
        val rname = spl(1).substring(3)
        if(!dict_seq.isDefinedAt(rname)){
          val res = dict_seq.size
          dict_seq.put(rname, res)
          header.+=(line)
        }
      }
      else if(line.startsWith("@RG")) {
        if(!rg) {
          header.+=(line)
          rg = true
        }
      }
      else if(line.startsWith("@PG")) {
        if(!pg) {
          header.+=(line)
          pg = true
        }
      }
    }
  }

  def getHead = header
  def getDict = dict_seq
}

