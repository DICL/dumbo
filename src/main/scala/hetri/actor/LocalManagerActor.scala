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

  override def receive: Receive = {
    // from host manager
    case NoMoreProblemMessage =>
      nomoreproblem = true
      if(pids_solving.isEmpty){
        context.stop(self)
        context.system.terminate()
      }
    case msg: ProblemResponseMessage =>

      context.system.actorOf(Props(classOf[TriCntActor], msg.pid, numColors, self, graphManager), "tri-" + msg.pid)
      pids_solving.add(msg.pid)

    // from triangle solver
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
