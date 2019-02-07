/*
 * HeTri: Multi-level Node Coloring for Efficient Triangle Enumeration on Heterogeneous Clusters
 * Authors: Ha-Myung Park and U Kang
 *
 * -------------------------------------------------------------------------
 * File: GraphManagingActor.java
 * - Graph Manager.
 */

package hetri.actor

import akka.actor.{Actor, ActorRef, Props}
import hetri.actor.GraphManagingActor.{GraphRequestMessage, GraphReturnMessage}
import hetri.actor.TriCntActor.GraphPreparedMessage
import hetri.graph.Graph
import hetri.tool.MemoryInfo
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.log4j.Logger

import scala.collection.mutable


class GraphManagingActor(part: Path, fs: FileSystem, gClass: Class[_ <: Graph]) extends Actor{

  private val logger = Logger.getLogger(getClass)

  case class GraphWrapper(var graph: Graph, var usable: Boolean = true){

    var count: Int = 0

    def increaseCounter(): Unit = count += 1
    def decreaseCounter(): Unit = count -= 1
    def getCounter: Int = count

  }

  val graphs: mutable.Map[Int, GraphWrapper] = mutable.Map[Int, GraphWrapper]()

  val waiting: mutable.Map[Int, List[ActorRef]] = mutable.Map[Int, List[ActorRef]]()

  val priority: mutable.ArrayBuffer[Int] = mutable.ArrayBuffer[Int]()

  var onloading_conter = 0

  /**
    * make the priority of a graph to be the highest
    * @param eid graph id
    * @return false if the graph is not in memory, true otherwise.
    */
  def tick(eid: Int): Boolean ={
    val i = priority.indexOf(eid)

    if (i == -1) false
    else {
      priority.remove(i)
      priority.append(eid)
      true
    }
  }

  /**
    * It receives a message and take an action
    * @return none
    */
  override def receive: Receive = {

    // a graph is requested.
    case msg: GraphRequestMessage =>
      logger.info("msg: " + msg)

      if(graphs.contains(msg.eid)){

        if(graphs(msg.eid).usable){
          // The graph is in memory
          val gw = graphs(msg.eid)

          tick(msg.eid)
          gw.increaseCounter()
          sender ! GraphPreparedMessage(msg.eid, graphs(msg.eid).graph)
        }
        else{
          // The graph is on loading.
          // Enroll the sender (TriCntActor) to the waiting list.
          // When loading complete, GraphManager will notify all waiting senders of it.
          val w_eid = waiting.getOrElseUpdate(msg.eid, List[ActorRef]())
          waiting.update(msg.eid, sender :: w_eid)
        }

      }
      else {
        // The graph is not in memory. Have GraphLoadingActor load it from hdfs.
        // When loading complete, GraphManager will notify all waiting senders of it.
        // Mark in `graphs` that the graph is on loading.

        val numGraphsInMem = priority.length
        val memUsed = MemoryInfo.getUsedMem
        val memForOneGraph = if (numGraphsInMem > 0) memUsed / numGraphsInMem else 0
        val memFree = MemoryInfo.getFreeMem



        logger.info(memUsed + ", " + memForOneGraph + ", " + memFree + ", " + numGraphsInMem + ", " + onloading_conter)

//        if (graphs.size < 15) {
                  if (memForOneGraph * 2 < memFree) {
          //        if (numGraphsInMem < 20){
          onloading_conter += 1
          graphs.put(msg.eid, GraphWrapper(null, usable = false))
          waiting.put(msg.eid, List[ActorRef](sender))

          context.system.actorOf(Props(new GraphLoadingActor(null, msg.eid, part, fs, self, gClass)))
        }
        else {

//          logger.info(priority.map(eid => graphs(eid).getCounter).mkString(" ") + "?")
          val idx = priority.indexWhere(eid => graphs(eid).getCounter == 0)

          if(idx < 0){
            logger.warn(s"No graph to discard! ")

            Thread.sleep(1000)

//            self ! GraphRequestMessage(msg.eid) (context.sender())
            self forward GraphRequestMessage(msg.eid)
          }
          else{
            val eid = priority.remove(idx)
            val gw = graphs(eid)
            graphs.remove(eid)

            onloading_conter += 1
            graphs.put(msg.eid, GraphWrapper(null, usable = false))
            waiting.put(msg.eid, List[ActorRef](sender))

            context.system.actorOf(Props(new GraphLoadingActor(gw.graph, msg.eid, part, fs, self, gClass)))
          }


        }
      }

    // a graph is released by a task solver
    case msg: GraphReturnMessage =>
      logger.info("msg: " + msg)
      val gw = graphs(msg.eid)
      gw.decreaseCounter()

    // a graph is prepared by a graph loader
    case msg: GraphPreparedMessage =>
      logger.info("msg: " + msg)
      // A graph is prepared.
      // Check the graph is available
      // Notify all waiting senders.
      // Remove the waiting list.
      priority.append(msg.eid)
      onloading_conter -= 1

      val gw = graphs(msg.eid)
      gw.usable = true
      gw.graph = msg.g
      waiting(msg.eid).foreach{ awaiter =>

        if(tick(msg.eid)){
          gw.increaseCounter()
          awaiter ! msg
        }
        else{
          logger.error(s"${msg.eid} is not in the priority queue.")
        }

      }
      waiting.remove(msg.eid)

//      logger.info("numGraphsInMem: " + priority.length)
  }


}


object GraphManagingActor{
  case class GraphRequestMessage(eid: Int)
  case class GraphReturnMessage(eid: Int)
}
