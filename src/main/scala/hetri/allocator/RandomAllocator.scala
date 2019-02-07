/*
 * HeTri: Multi-level Node Coloring for Efficient Triangle Enumeration on Heterogeneous Clusters
 * Authors: Ha-Myung Park and U Kang
 *
 * -------------------------------------------------------------------------
 * File: RandomAllocator.java
 * - Random task allocator.
 */


package hetri.allocator

import scala.util.Random

class RandomAllocator(numColors: Int) extends TaskAllocator {

  val numTasks: Int = numColors * numColors * numColors
  val tasks: Array[Int] = shuffle((0 until numTasks).toArray)

  var cur: Int = 0

  /**
    * find a task to allocate to worker wid
    * @param wid the worker id
    * @return a task to allocate
    */
  override def allocate(wid: Int): Option[Int] = {
    val res = if(cur < numTasks) Some(tasks(cur))
    else None

    cur += 1

    res
  }

  /**
    * shuffle an array
    * @param arr an array
    * @return return the shuffled array
    */
  def shuffle(arr: Array[Int]): Array[Int] = {
    val rand = new Random()
    val cnt = arr.length

    for (i <- cnt until 1 by -1)
      swap(arr, i - 1, rand.nextInt(i))

    arr
  }

  /**
    * swap two array entries
    * @param arr the array
    * @param i the first entry id
    * @param j the second entry id
    */
  def swap(arr: Array[Int], i: Int, j: Int): Unit = {
    val tmp = arr(i)
    arr(i) = arr(j)
    arr(j) = tmp
  }


}
