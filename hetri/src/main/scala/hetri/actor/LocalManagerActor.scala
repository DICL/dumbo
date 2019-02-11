/*
 * HeTri: Multi-level Node Coloring for Efficient Triangle Enumeration on Heterogeneous Clusters
 * Authors: Ha-Myung Park and U Kang
 *
 * -------------------------------------------------------------------------
 * File: LocalManagerActor.java
 * - Task manager.
 */

package hetri.actor

import akka.actor.{Actor, ActorRef, ActorSelection, Props}
import hetri.actor.HostManagerActor.ProblemRequestMessage
import hetri.actor.LocalManagerActor.{FinishMessage, NoMoreProblemMessage, ProblemResponseMessage}
import org.apache.hadoop.io.Text
import org.apache.hadoop.mapreduce.Mapper

import scala.collection.mutable


class LocalManagerActor(id: Int, numColors: Int, concurrencyLevel: Int, graphManager: ActorRef, hostManager: ActorSelection,
                        hadoopMapperContext: Mapper[Object, Text, Object, Object]#Context) extends Actor{

  var pids_solving: mutable.Set[Int] = mutable.Set[Int]()
  var nomoreproblem: Boolean = false

  for (i <- 0 until concurrencyLevel) {
    hostManager ! ProblemRequestMessage(id)
  }

  /**
    * It receives a message and take an action
    * @return none
    */
  override def receive: Receive = {

    // no task remains to assign (from the job manager).
    case NoMoreProblemMessage =>
      nomoreproblem = true
      if(pids_solving.isEmpty){
        context.stop(self)
        context.system.terminate()
      }


    // a task is assigned (from the job manager).
    case msg: ProblemResponseMessage =>

      context.system.actorOf(Props(classOf[TriCntActor], msg.pid, numColors, self, graphManager), "tri-" + msg.pid)
      pids_solving.add(msg.pid)

    // a task is solved (from a task solver).
    case msg: FinishMessage =>
      hadoopMapperContext.getCounter("GTE", "triangles").increment(msg.numTriangles)
      pids_solving.remove(msg.pid)

      if(!nomoreproblem) hostManager ! ProblemRequestMessage(id)
      else if (pids_solving.isEmpty){
        context.system.stop(self)
        context.system.terminate()
      }
      hostManager ! msg
  }
}

object LocalManagerActor{
  case object NoMoreProblemMessage
  case class ProblemResponseMessage(pid: Int)
  case class FinishMessage(pid: Int, numTriangles: Long)
}
