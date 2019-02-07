/*
 * HeTri: Multi-level Node Coloring for Efficient Triangle Enumeration on Heterogeneous Clusters
 * Authors: Ha-Myung Park and U Kang
 *
 * -------------------------------------------------------------------------
 * File: Problem.java
 * - Sub-problem (Task) of HeTri.
 */

package hetri.allocator

class Problem(val id: Int, val numColors: Int) {

  val colors: Array[Byte] = Array.ofDim[Byte](3)

  {
    var tmp = id
    for (i <- colors.indices) {
      colors(i) = (tmp % numColors).toByte
      tmp /= numColors
    }
  }

  val eids: Array[Int] = Array(colors(0) * numColors + colors(1),
                               colors(0) * numColors + colors(2),
                               colors(1) * numColors + colors(2))

  override def toString: String = s"[$id,(${colors.mkString(",")})]"
}