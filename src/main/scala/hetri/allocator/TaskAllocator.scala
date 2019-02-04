package hetri.allocator

trait TaskAllocator {
  def allocate(wid: Int): Option[Int]
  def task(pid: Int, numColors: Int): (Int, Int, Int) = {
    var tmp = pid
    val i = tmp % numColors
    tmp /= numColors
    val j = tmp % numColors
    tmp /= numColors
    val k = tmp
    (i, j, k)
  }
}
