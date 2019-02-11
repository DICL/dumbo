/*
 * HeTri: Multi-level Node Coloring for Efficient Triangle Enumeration on Heterogeneous Clusters
 * Authors: Ha-Myung Park and U Kang
 *
 * -------------------------------------------------------------------------
 * File: HostMangerActor.java
 * - job manager.
 */


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

  /**
    * It receives a message and take an action
    * @return none
    */
  override def receive: Receive = {

    // a worker has requested a task
    case msg: ProblemRequestMessage =>

      val wid = msg.id

      alloc.allocate(wid) match {
        case Some(pid) =>
          if(!assigned.contains(wid)) assigned(wid) = new mutable.ArrayBuffer[(Int, Int, Int)]()
          assigned(wid) += alloc.task(pid, numColors)
          sender ! ProblemResponseMessage(pid)
        case None => sender ! NoMoreProblemMessage
      }

    // a task is solved (from a task manager)
    case msg: FinishMessage =>
      pb.step()
      if(pb.getMax == pb.getCurrent) pb.close()
  }

}

object HostManagerActor{
  case class ProblemRequestMessage(id: Int)
}


