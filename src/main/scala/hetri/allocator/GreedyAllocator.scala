/*
 * HeTri: Multi-level Node Coloring for Efficient Triangle Enumeration on Heterogeneous Clusters
 * Authors: Ha-Myung Park and U Kang
 *
 * -------------------------------------------------------------------------
 * File: GreedyAllocator.java
 * - A greedy task allocator.
 */

package hetri.allocator

import scala.collection.mutable

class GreedyAllocator(numColors: Int) extends TaskAllocator {

  val numTasks: Int = numColors * numColors * numColors

  val tasks: mutable.BitSet = new mutable.BitSet()
  (0 until numTasks).foreach(tasks.add)

  // edge color set of workers
  val workers: mutable.HashMap[Int, mutable.HashSet[(Int, Int)]] = new mutable.HashMap[Int, mutable.HashSet[(Int, Int)]]()

  /**
    * find a task to allocate to worker wid
    * @param wid the worker id
    * @return a task to allocate
    */
  override def allocate(wid: Int): Option[Int] = {

    val assigned = workers.getOrElseUpdate(wid, new mutable.HashSet[(Int, Int)])


    if(tasks.nonEmpty){
      val min = tasks.iterator.map{ tid =>
        val (i, j, k) = task(tid, numColors)
        Set((i, j), (i, k), (j, k)).count(x => !assigned.contains(x))
      }.min

      val candidate = tasks.iterator.filter { tid =>
        val (i, j, k) = task(tid, numColors)
        Set((i, j), (i, k), (j, k)).count(x => !assigned.contains(x)) == min
      }

      val res = candidate.maxBy{tid =>
        val (i, j, k) = task(tid, numColors)
        Set(i, j, k).size
      }

      tasks.remove(res)
      Some(res)
    }
    else None

  }

}
