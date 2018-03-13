//package dblab
//
//import java.io.{BufferedReader, InputStreamReader}
//import java.net.{ServerSocket, Socket}
//
//import org.apache.spark.streaming.StreamingContext
//
//object KillServer {
//
//  class NetworkService(port: Int, ssc: StreamingContext) extends Runnable {
//    val serverSocket = new ServerSocket(port)
//
//    def run() {
//      Thread.currentThread().setName("Zhuangdy | Waiting for graceful stop at port " + port)
//      while (true) {
//        val socket = serverSocket.accept()
//        (new Handler(socket, ssc)).run()
//      }
//    }
//  }
//
//  class Handler(socket: Socket, ssc: StreamingContext) extends Runnable {
//    def run() {
//      val reader = new InputStreamReader(socket.getInputStream)
//      val br = new BufferedReader(reader)
//      if (br.readLine() == "kill") {
//        ssc.stop(true, true)
//      }
//      br.close();
//    }
//  }
//
//  def run(port:Int, ssc: StreamingContext): Unit ={
//    (new NetworkService(port, ssc)).run
//  }
//}