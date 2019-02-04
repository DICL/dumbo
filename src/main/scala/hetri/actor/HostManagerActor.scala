package hetri.actor

import akka.actor.Actor
import hetri.actor.HostManagerActor.ProblemRequestMessage
import hetri.actor.LocalManagerActor.{FinishMessage, NoMoreProblemMessage, ProblemResponseMessage}
import hetri.allocator.TaskAllocator
import me.tongfei.progressbar.ProgressBar

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

class HostManagerActor(numColors: Int, alloc: TaskAllocator) extends Actor{

  var nomoreproblem: Boolean = false

  val numProblems = numColors * numColors * numColors

  val assigned: mutable.Map[Int, ArrayBuffer[(Int, Int, Int)]] = new mutable.HashMap[Int, mutable.ArrayBuffer[(Int, Int, Int)]]()


  val pb = new ProgressBar("Counting Triangles", numProblems)

  override def receive: Receive = {
    case msg: ProblemRequestMessage =>

      val wid = msg.id

      alloc.allocate(wid) match {
        case Some(pid) =>
          if(!assigned.contains(wid)) assigned(wid) = new mutable.ArrayBuffer[(Int, Int, Int)]()
          assigned(wid) += alloc.task(pid, numColors)
          sender ! ProblemResponseMessage(pid)
        case None => sender ! NoMoreProblemMessage
      }

    case msg: FinishMessage =>
      pb.step()
      if(pb.getMax == pb.getCurrent) pb.close()
  }

//  override def postStop(): Unit = {
//
//    assigned.foreach { case (wid, tasks) =>
//      println(wid + ":")
//      tasks.foreach(println)
//    }
//
//  }
}

object HostManagerActor{
  case class ProblemRequestMessage(id: Int)
}


