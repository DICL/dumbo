/*
 * HeTri: Multi-level Node Coloring for Efficient Triangle Enumeration on Heterogeneous Clusters
 * Authors: Ha-Myung Park and U Kang
 *
 * -------------------------------------------------------------------------
 * File: MultiLevelColoring.java
 * - A task allocator based on multi-level node coloring.
 */

package hetri.allocator

import hetri.allocator.MultiLevelColoring.TaskGroup

import scala.collection.mutable

class MultiLevelColoring(numColors: Int) extends TaskAllocator {

  val L = if(numColors == 1) 0 else Integer.bitCount((Integer.highestOneBit(numColors-1) << 1) - 1)

  val numTasksOf: mutable.Map[TaskGroup, Int] = new mutable.HashMap[TaskGroup, Int]().withDefaultValue(0)

  val remainingTasks: mutable.BitSet = new mutable.BitSet()

  //(wid, level, i, j)
  val edgecolorset: mutable.HashSet[(Int, Int, Int, Int)] = new mutable.HashSet[(Int, Int, Int, Int)]()

  val preoccupied: mutable.HashSet[TaskGroup] = new mutable.HashSet[TaskGroup]()

  for(i <- 0 until numColors; j <- 0 until numColors; k <- 0 until numColors){

    val pid = ((k * numColors) + j) * numColors + i
    remainingTasks.add(pid)

    for (l <- 0 to L) {
      val tg = getTaskGroup(l, i, j, k)
      numTasksOf(tg) = numTasksOf.getOrElse(tg, 0) + 1
    }

  }

  /**
    * get the pid of a task
    * @param i the first color of the task
    * @param j the second color of the task
    * @param k the third color of the task
    * @return the pid
    */
  def pid(i: Int, j: Int, k: Int): Int ={
    ((k * numColors) + j) * numColors + i
  }

  /**
    * get the remaining tasks in task group tg
    * @param tg the task group
    * @return a stream of tasks
    */
  def getRemainingTasksOf(tg: TaskGroup): Seq[Int] ={

    if(tg.level == L){
      if(tg.j == -1){
        if(tg.i < numColors) {
          Stream(pid(tg.i, tg.i, tg.i)).filter(remainingTasks.contains)
        }
        else{
          Stream.empty[Int]
        }
      }
      else if(tg.k == -1){
        if(tg.i < numColors && tg.j < numColors) {
          Stream(pid(tg.i, tg.i, tg.j), pid(tg.j, tg.i, tg.i),
            pid(tg.i, tg.j, tg.i), pid(tg.j, tg.i, tg.j),
            pid(tg.i, tg.j, tg.j), pid(tg.j, tg.j, tg.i)).filter(remainingTasks.contains)
        }
        else{
          Stream.empty[Int]
        }
      }
      else{
        if(tg.i < numColors && tg.j < numColors && tg.k < numColors) {
          Stream(pid(tg.i, tg.j, tg.k), pid(tg.i, tg.k, tg.j),
            pid(tg.j, tg.i, tg.k), pid(tg.j, tg.k, tg.i),
            pid(tg.k, tg.i, tg.j), pid(tg.k, tg.j, tg.i)).filter(remainingTasks.contains)
        }
        else{
          Stream.empty[Int]
        }
      }

    }
    else{
      tg.getChildren.toStream.flatMap(getRemainingTasksOf)
    }

  }


  val lastTaskAssignedTo: mutable.HashMap[Int, Int] = new mutable.HashMap()

  /**
    * get task group of a task at a level
    * @param level the level
    * @param i the first color of a task
    * @param j the second color of a task
    * @param k the third color of a task
    * @return a task group
    */
  def getTaskGroup(level: Int, i: Int, j: Int, k: Int): TaskGroup ={
    val mask = (1 << level) - 1
    val x = Array(i & mask, j & mask, k & mask).distinct.sorted

    x.length match {
      case 1 => TaskGroup(level, x(0))
      case 2 => TaskGroup(level, x(0), x(1))
      case 3 => TaskGroup(level, x(0), x(1), x(2))
    }

  }

  /**
    * find the nearest task from the given task
    * @param pid task id
    * @param wid worker id
    * @return a task
    */
  def getNearestTaskFrom(pid: Int, wid: Int): Option[Int] ={
    val (i, j, k) = task(pid, numColors)
    Array(i, j, k).distinct.sorted

    // get non empty ancestor
    val ancestorLevel = (L to 0 by -1).find(l => numTasksOf(getTaskGroup(l, i, j, k)) > 0)

    ancestorLevel match {
      case Some(l) =>
        val leaf = getMaxLeafTaskGroupFrom(getTaskGroup(l, i, j, k), wid)
        val nearest_pid = getRemainingTasksOf(leaf).head
        Some(nearest_pid)
      case None => None// every task is assigned.
    }

  }


  /**
    * find the leaf node having max priority
    * @param tg task group
    * @param wid worker id
    * @return
    */
  def getMaxLeafTaskGroupFrom(tg: TaskGroup, wid: Int): TaskGroup ={
    if(tg.level == L){
      return tg
    }
    else{
      val children = tg.getChildren
      return byPreemptionAndColorset(wid, children)
    }
  }

  /**
    * select the child that has max priority
    * @param wid worker id
    * @param children a list of task groups
    * @return a task group
    */
  private def byPreemptionAndColorset(wid: Int, children: Seq[TaskGroup]): TaskGroup = {

    val po = children.filterNot(preoccupied).filter{x => numTasksOf(x) > 0}

    if(po.nonEmpty) byEdgecolorset(wid, po)
    else byEdgecolorset(wid, children)

  }

  /**
    * select the child that has max priority by edge color
    * @param wid worker id
    * @param children a list of task groups
    * @return a task group
    */
  private def byEdgecolorset(wid: Int, children: Seq[TaskGroup]): TaskGroup = {
    val max = numTasksOf(children.maxBy(numTasksOf))

    val min = children.filter(x => numTasksOf(x) != 0).minBy { child =>

      var cnt = 0

      if(child.k != -1){
        if(!edgecolorset.contains((wid, child.level, child.i, child.j))) cnt += 1
        if(!edgecolorset.contains((wid, child.level, child.i, child.k))) cnt += 1
        if(!edgecolorset.contains((wid, child.level, child.j, child.k))) cnt += 1
      }
      else if(child.j != -1){
        if(!edgecolorset.contains((wid, child.level, child.i, child.j))) cnt += 1
        if(!edgecolorset.contains((wid, child.level, child.i, child.i))) cnt += 1
        if(!edgecolorset.contains((wid, child.level, child.j, child.j))) cnt += 1
      }
      else{
        if(!edgecolorset.contains((wid, child.level, child.i, child.i))) cnt += 1
      }

      cnt
    }

    val min_val = {
      var cnt = 0

      if(min.k != -1){
        if(edgecolorset.contains((wid, min.level, min.i, min.j))) cnt += 1
        if(edgecolorset.contains((wid, min.level, min.i, min.k))) cnt += 1
        if(edgecolorset.contains((wid, min.level, min.j, min.k))) cnt += 1
      }
      else if(min.j != -1){
        if(edgecolorset.contains((wid, min.level, min.i, min.j))) cnt += 1
        if(edgecolorset.contains((wid, min.level, min.i, min.i))) cnt += 1
        if(edgecolorset.contains((wid, min.level, min.j, min.j))) cnt += 1
      }
      else{
        if(edgecolorset.contains((wid, min.level, min.i, min.i))) cnt += 1
      }

      cnt
    }

    val selected = children.filter { child =>

      var cnt = 0

      if(child.k != -1){
        if(edgecolorset.contains((wid, child.level, child.i, child.j))) cnt += 1
        if(edgecolorset.contains((wid, child.level, child.i, child.k))) cnt += 1
        if(edgecolorset.contains((wid, child.level, child.j, child.k))) cnt += 1
      }
      else if(child.j != -1){
        if(edgecolorset.contains((wid, child.level, child.i, child.j))) cnt += 1
        if(edgecolorset.contains((wid, child.level, child.i, child.i))) cnt += 1
        if(edgecolorset.contains((wid, child.level, child.j, child.j))) cnt += 1
      }
      else{
        if(edgecolorset.contains((wid, child.level, child.i, child.i))) cnt += 1
      }

      cnt == min_val
    }.maxBy(numTasksOf)

    getMaxLeafTaskGroupFrom(selected, wid)
  }


  /**
    * find a task to allocate to worker wid
    * @param wid the worker id
    * @return a task to allocate
    */
  override def allocate(wid: Int): Option[Int] ={

    val pid_selected = lastTaskAssignedTo.get(wid) match {
      case Some(pid) =>
        getNearestTaskFrom(pid, wid)
      case None =>
        val top = TaskGroup(0,0)

        if(numTasksOf(top) > 0){
          val leaf = getMaxLeafTaskGroupFrom(TaskGroup(0,0), wid)
          val nearest_pid = getRemainingTasksOf(leaf).head
          Some(nearest_pid)
        }
        else None
    }



    if(pid_selected.nonEmpty){

      val pid = pid_selected.get

      val (i, j, k) = task(pid, numColors)

      // delete pid_selected
      (0 to L).foreach { l =>
        val tg = getTaskGroup(l, i, j, k)
        numTasksOf(tg) -= 1
      }
      remainingTasks.remove(pid)


      //maintain preocupied
      (0 to L).foreach { l =>
        val tg = getTaskGroup(l, i, j, k)
        preoccupied.add(tg)
      }


      // maintain colorset
      (0 to L).foreach { l =>
        val tg = getTaskGroup(l, i, j, k)
        if(tg.k != -1){
          edgecolorset.add((wid, l, tg.i, tg.j))
          edgecolorset.add((wid, l, tg.i, tg.k))
          edgecolorset.add((wid, l, tg.j, tg.k))
        }
        else if(tg.j != -1){
          edgecolorset.add((wid, l, tg.i, tg.j))
          edgecolorset.add((wid, l, tg.i, tg.i))
          edgecolorset.add((wid, l, tg.j, tg.j))
        }
        else{
          edgecolorset.add((wid, l, tg.i, tg.i))
        }
      }

      //memo on wid
      lastTaskAssignedTo(wid) = pid
    }


    pid_selected

  }

}

object MultiLevelColoring{
  case class TaskGroup(level: Int, i: Int, j: Int = -1, k: Int = -1){

    /**
      * get parent
      * @return parent task group
      */
    def getParent: TaskGroup = {

      val pl = level - 1
      val mask = (1 << pl) - 1

      if(level == 0) null
      else {
        val x = (if (j == -1) {
          Array(i & mask)
        }
        else if (k == -1) {
          Array(i & mask, j & mask)
        }
        else {
          Array(i & mask, j & mask, k & mask)
        }).distinct.sorted

        x.length match {
          case 1 => TaskGroup(pl, x(0))
          case 2 => TaskGroup(pl, x(0), x(1))
          case 3 => TaskGroup(pl, x(0), x(1), x(2))
        }
      }
    }

    /**
      * get children
      * @return children task groups
      */
    def getChildren: Seq[TaskGroup] ={

      val childLevel = level + 1

      if(j == -1){ // single color
        val ii = i + (1 << level)
        Seq(TaskGroup(childLevel, i), TaskGroup(childLevel, i, ii), TaskGroup(childLevel, ii))
      }
      else if (k == -1){ // two colors
        val ii = i + (1 << level)
        val jj = j + (1 << level)
        Seq(TaskGroup(childLevel, i, j, ii), TaskGroup(childLevel, i, j),
          TaskGroup(childLevel, i, j, jj), TaskGroup(childLevel, j, ii),
          TaskGroup(childLevel, j, ii, jj), TaskGroup(childLevel, ii, jj),
          TaskGroup(childLevel, i, ii, jj), TaskGroup(childLevel, i, jj))
      }
      else { // three colors
        val ii = i + (1 << level)
        val jj = j + (1 << level)
        val kk = k + (1 << level)

        Seq(TaskGroup(childLevel, i, j, k), TaskGroup(childLevel, i, j, kk),
          TaskGroup(childLevel, i, jj, kk), TaskGroup(childLevel, i, k, jj),
          TaskGroup(childLevel, k, ii, jj), TaskGroup(childLevel, j, k, ii),
          TaskGroup(childLevel, j, ii, kk), TaskGroup(childLevel, ii, jj, kk)
        )

      }
    }

    /**
      * task group to string
      * @return a string
      */
    override def toString: String = {
      if(j == -1) s"($level, $i)"
      else if(k == -1) s"($level, $i, $j)"
      else s"($level, $i, $j, $k)"
    }


  }
}