package kr.ac.ut.csdb

import java.net.{ServerSocket, Socket}
import java.net._
import java.io._

class socketNet(port: Int, msg:String, address:String="localhost") extends java.io.Serializable with Runnable {
  override def run(): Unit = {
    val s = new Socket(InetAddress.getByName(address), port)
    val out = new PrintStream(s.getOutputStream(),true, "UTF-8")
    out.write(msg.getBytes("UTF-8"))
    s.close()
  }
}
