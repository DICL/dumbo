package hetri.allocator

import scala.util.Random

class RandomAllocator(numColors: Int) extends TaskAllocator {

  val numTasks: Int = numColors * numColors * numColors
  val tasks: Array[Int] = shuffle((0 until numTasks).toArray)

  var cur: Int = 0

  override def allocate(wid: Int): Option[Int] = {
    val res = if(cur < numTasks) Some(tasks(cur))
    else None

    cur += 1

    res
  }

  def shuffle(arr: Array[Int]): Array[Int] = {
    val rand = new Random()
    val cnt = arr.length

    for (i <- cnt until 1 by -1)
      swap(arr, i - 1, rand.nextInt(i))

    arr
  }

  def swap(arr: Array[Int], i: Int, j: Int): Unit = {
    val tmp = arr(i)
    arr(i) = arr(j)
    arr(j) = tmp
  }


}
