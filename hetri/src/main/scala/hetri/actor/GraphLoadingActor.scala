/*
 * HeTri: Multi-level Node Coloring for Efficient Triangle Enumeration on Heterogeneous Clusters
 * Authors: Ha-Myung Park and U Kang
 *
 * -------------------------------------------------------------------------
 * File: GraphLoadingActor.java
 * - graph loader.
 */

package hetri.actor

import akka.actor.{Actor, ActorRef}
import com.esotericsoftware.kryo.io.Input
import hetri.actor.TriCntActor.GraphPreparedMessage
import hetri.graph.Graph
import org.apache.hadoop.fs.{FileSystem, Path}


class GraphLoadingActor(var graph: Graph, eid: Int, part: Path, fs: FileSystem, graphManager: ActorRef, gClass: Class[_ <: Graph]) extends Actor{

  GraphLoadingActor.time {
    if(graph == null){
      graph = gClass.newInstance()
    }
    val base = hdfsBasePath(eid)
    val iedge = new Input(fs.open(base.suffix(".edge")))
    val inode = new Input(fs.open(base.suffix(".node")))
    graph.read(iedge, inode)

    graphManager ! GraphPreparedMessage(eid, graph)

    context.stop(self)
  }

  /**
    * get the path of the graph eid
    * @param eid graph id
    * @return the path to the graph
    */
  private def hdfsBasePath(eid: Int) = {
    val ceString = ((eid >> 8) & 0xFF) + "-" + (eid & 0xFF)
    part.suffix("/graph-" + ceString)
  }

  /**
    * It receives a message and take an action
    * @return none
    */
  override def receive: Receive = {
    case msg: Any => /* This actor receives no message*/
  }
}

object GraphLoadingActor{
  var executionTime: Long = 0

  def time[V](block: => V): Unit ={
    val t: Long = System.currentTimeMillis()
    val result = block
    increaseTime(System.currentTimeMillis() - t)
  }

  def increaseTime(time: Long): Unit = synchronized {
    executionTime += time
  }

}

