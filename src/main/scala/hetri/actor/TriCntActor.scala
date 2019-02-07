/*
 * HeTri: Multi-level Node Coloring for Efficient Triangle Enumeration on Heterogeneous Clusters
 * Authors: Ha-Myung Park and U Kang
 *
 * -------------------------------------------------------------------------
 * File: TriCntActor.java
 * - Task solver.
 */

package hetri.actor

import java.lang.management.ManagementFactory

import akka.actor.{Actor, ActorRef}
import hetri.actor.GraphManagingActor.{GraphRequestMessage, GraphReturnMessage}
import hetri.actor.LocalManagerActor.FinishMessage
import hetri.actor.TriCntActor.GraphPreparedMessage
import hetri.allocator.Problem
import hetri.graph.{CSR, CSRV, Graph}
import hetri.triangle.{TriangleCounter, TriangleCounterCSR}

import scala.collection.mutable


class TriCntActor(pid: Int, numcolors: Int, localManager: ActorRef, graphManager: ActorRef) extends Actor{


  val p = new Problem(pid, numcolors)

  val graphs: mutable.Map[Int, Graph] = mutable.Map[Int, Graph]()

  val eids = Array(p.colors(0).toInt << 8 | p.colors(1),
  p.colors(0).toInt << 8 | p.colors(2),
  p.colors(1).toInt << 8 | p.colors(2))

  eids.distinct.foreach(eid => graphManager ! GraphRequestMessage(eid))

  /**
    * It receives a message and take an action
    * @return none
    */
  override def receive: Receive = {

    // a graph is prepared
    case msg: GraphPreparedMessage =>
      graphs(msg.eid) = msg.g

      if (eids.forall(graphs.contains)) TriCntActor.time {

        val count = graphs(eids(0)) match {
          case csr: CSR =>
            TriangleCounterCSR.countTriangles(csr, graphs(eids(1)).asInstanceOf[CSR], graphs(eids(2)).asInstanceOf[CSR])
          case _ =>
            TriangleCounter.countTriangles(graphs(eids(0)).asInstanceOf[CSRV],
              graphs(eids(1)).asInstanceOf[CSRV], graphs(eids(2)).asInstanceOf[CSRV], false)
        }

        localManager ! FinishMessage(pid, count)
        eids.distinct.foreach(eid => graphManager ! GraphReturnMessage(eid))

        context.stop(self)

      }
  }

}

object TriCntActor{
  case class GraphPreparedMessage(eid: Int, g: Graph)

  var executionTime: Long = 0

  def time[V](block: => V): Unit ={

    val tmxb = ManagementFactory.getThreadMXBean
    tmxb.getCurrentThreadCpuTime()

    val t: Long = System.currentTimeMillis()
    val result = block
    increaseTime(System.currentTimeMillis() - t)
  }

  def increaseTime(time: Long): Unit = synchronized {
    executionTime += time
  }
}