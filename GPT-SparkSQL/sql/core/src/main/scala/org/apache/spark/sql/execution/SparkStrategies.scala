/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.sql.execution

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Strategy
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.expressions.aggregate.First
import org.apache.spark.sql.catalyst.planning._
import org.apache.spark.sql.catalyst.plans._
import org.apache.spark.sql.catalyst.plans.logical._
import org.apache.spark.sql.catalyst.plans.physical._
import org.apache.spark.sql.execution
import org.apache.spark.sql.execution.columnar.{InMemoryRelation, InMemoryTableScanExec}
import org.apache.spark.sql.execution.command._
import org.apache.spark.sql.execution.datasources.{HadoopFsRelation, LogicalRelation}
import org.apache.spark.sql.execution.exchange.ShuffleExchange
import org.apache.spark.sql.execution.joins.{BuildLeft, BuildRight}
import org.apache.spark.sql.execution.streaming._
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.streaming.StreamingQuery
import org.apache.spark.util.logging

/**
 * Converts a logical plan into zero or more SparkPlans.  This API is exposed for experimenting
 * with the query planner and is not designed to be stable across spark releases.  Developers
 * writing libraries should instead consider using the stable APIs provided in
 * [[org.apache.spark.sql.sources]]
 */
abstract class SparkStrategy extends GenericStrategy[SparkPlan] {

  override protected def planLater(plan: LogicalPlan): SparkPlan = PlanLater(plan)
}

case class PlanLater(plan: LogicalPlan) extends LeafExecNode {

  override def output: Seq[Attribute] = plan.output

  protected override def doExecute(): RDD[InternalRow] = {
    throw new UnsupportedOperationException()
  }
}

abstract class SparkStrategies extends QueryPlanner[SparkPlan] {

  self: SparkPlanner =>

  /**
   * Plans special cases of limit operators.
   */
  object SpecialLimits extends Strategy {
    override def apply(plan: LogicalPlan): Seq[SparkPlan] = plan match {
      case logical.ReturnAnswer(rootPlan) => rootPlan match {
        case logical.Limit(IntegerLiteral(limit), logical.Sort(order, true, child)) =>
          execution.TakeOrderedAndProjectExec(limit, order, child.output, planLater(child)) :: Nil
        case logical.Limit(
            IntegerLiteral(limit),
            logical.Project(projectList, logical.Sort(order, true, child))) =>
          execution.TakeOrderedAndProjectExec(
            limit, order, projectList, planLater(child)) :: Nil
        case logical.Limit(IntegerLiteral(limit), child) =>
          execution.CollectLimitExec(limit, planLater(child)) :: Nil
        case other => planLater(other) :: Nil
      }
      case logical.Limit(IntegerLiteral(limit), logical.Sort(order, true, child)) =>
        execution.TakeOrderedAndProjectExec(limit, order, child.output, planLater(child)) :: Nil
      case logical.Limit(
          IntegerLiteral(limit), logical.Project(projectList, logical.Sort(order, true, child))) =>
        execution.TakeOrderedAndProjectExec(
          limit, order, projectList, planLater(child)) :: Nil
      case _ => Nil
    }
  }

  /**
   * Select the proper physical plan for join based on joining keys and size of logical plan.
   *
   * At first, uses the [[ExtractEquiJoinKeys]] pattern to find joins where at least some of the
   * predicates can be evaluated by matching join keys. If found,  Join implementations are chosen
   * with the following precedence:
   *
   * - Broadcast: if one side of the join has an estimated physical size that is smaller than the
   *     user-configurable [[SQLConf.AUTO_BROADCASTJOIN_THRESHOLD]] threshold
   *     or if that side has an explicit broadcast hint (e.g. the user applied the
   *     [[org.apache.spark.sql.functions.broadcast()]] function to a DataFrame), then that side
   *     of the join will be broadcasted and the other side will be streamed, with no shuffling
   *     performed. If both sides of the join are eligible to be broadcasted then the
   * - Shuffle hash join: if the average size of a single partition is small enough to build a hash
   *     table.
   * - Sort merge: if the matching join keys are sortable.
   *
   * If there is no joining keys, Join implementations are chosen with the following precedence:
   * - BroadcastNestedLoopJoin: if one side of the join could be broadcasted
   * - CartesianProduct: for Inner join
   * - BroadcastNestedLoopJoin
   */
  object JoinSelection extends Strategy with PredicateHelper {

    /**
     * Matches a plan whose output should be small enough to be used in broadcast join.
     */
    private def canBroadcast(plan: LogicalPlan): Boolean = {
      plan.stats(conf).hints.isBroadcastable.getOrElse(false) ||
        (plan.stats(conf).sizeInBytes >= 0 &&
          plan.stats(conf).sizeInBytes <= conf.autoBroadcastJoinThreshold)
    }

    /**
     * Matches a plan whose single partition should be small enough to build a hash table.
     *
     * Note: this assume that the number of partition is fixed, requires additional work if it's
     * dynamic.
     */
    private def canBuildLocalHashMap(plan: LogicalPlan): Boolean = {
      plan.stats(conf).sizeInBytes < conf.autoBroadcastJoinThreshold * conf.numShufflePartitions
    }

    /**
     * Returns whether plan a is much smaller (3X) than plan b.
     *
     * The cost to build hash map is higher than sorting, we should only build hash map on a table
     * that is much smaller than other one. Since we does not have the statistic for number of rows,
     * use the size of bytes here as estimation.
     */
    private def muchSmaller(a: LogicalPlan, b: LogicalPlan): Boolean = {
      a.stats(conf).sizeInBytes * 3 <= b.stats(conf).sizeInBytes
    }

    private def canBuildRight(joinType: JoinType): Boolean = joinType match {
      case _: InnerLike | LeftOuter | LeftSemi | LeftAnti => true
      case j: ExistenceJoin => true
      case _ => false
    }

    private def canBuildLeft(joinType: JoinType): Boolean = joinType match {
      case _: InnerLike | RightOuter => true
      case _ => false
    }

    def findScanOP_v2(p: LogicalPlan, joinCol: String, logicalRel: scala.collection.mutable.ArrayBuffer[LogicalRelation]) : Unit = p match {
      // p : is logical plan or any kind of query operator

      case _ if !p.isInstanceOf[LogicalRelation] => {
        val plan = p
        //logDebug(s"curPlanOP: " + plan.nodeName + ", with joinCol: " + joinCol)
        val childSize = plan.children.size
        childSize match {
          case 1 => {
            //logDebug(s"curPlan Child-0 OP: " + plan.children(0).nodeName)
            findScanOP_v2(plan.children(0), joinCol, logicalRel)
          }
          case 2 => {
            //logDebug(s"curPlan Child-0 OP: " + plan.children(0).nodeName)
            //logDebug(s"curPlan Child-1 OP: " + plan.children(1).nodeName)
            findScanOP_v2(plan.children(0), joinCol, logicalRel)
            findScanOP_v2(plan.children(1), joinCol, logicalRel)
          }

          case 0 => {
            Unit
          }

          case _ => {
            Unit
          }
        }
      }

      case _ if p.isInstanceOf[LogicalRelation] => {
        val candidateRel = p.asInstanceOf[LogicalRelation]
        candidateRel.schema.foreach { c =>
          if(c.name == joinCol) {
            //logDebug(s"FOUND CORRECT REL FOR JoinCol: " + joinCol)
            logicalRel += candidateRel
          }
        }
      }
    }

    def findScanOP(p: LogicalPlan): LogicalPlan = {

      var curChildOP = p

      while (!curChildOP.isInstanceOf[LogicalRelation] && !curChildOP.children.isEmpty) {
        curChildOP = curChildOP.children(0)
      }
      curChildOP
    }

    def apply(plan: LogicalPlan): Seq[SparkPlan] = plan match {

      // --- BroadcastHashJoin --------------------------------------------------------------------

      case ExtractEquiJoinKeys(joinType, leftKeys, rightKeys, condition, left, right)
        if canBuildRight(joinType) && canBroadcast(right) => {

        plan.setGPTJoinType(joinType.toString)
        var anyLeftJoinCol = leftKeys(0).references.toSeq(0).name
        var anyRightJoinCol = rightKeys(0).references.toSeq(0).name
        val leftStorage = scala.collection.mutable.ArrayBuffer[LogicalRelation]()
        val rightStorage = scala.collection.mutable.ArrayBuffer[LogicalRelation]()
        findScanOP_v2(left, anyLeftJoinCol, leftStorage)
        findScanOP_v2(right, anyRightJoinCol, rightStorage)

        if(!leftStorage.isEmpty && !rightStorage.isEmpty) {

          val leftRel = leftStorage(0)
          val rightRel = rightStorage(0)

        //logDebug(s"Found LogicalRelation for leftRel: " + leftRel.nodeName)
        //logDebug(s"Found LogicalRelation for rightRel: " + rightRel.nodeName)

          var leftHadoopRel: HadoopFsRelation = null
          var rightHadoopRel: HadoopFsRelation = null

          if (leftRel.isInstanceOf[LogicalRelation]) {
            val logicalRel = leftRel.asInstanceOf[LogicalRelation]
            logicalRel.relation match {
              case fsRel: HadoopFsRelation =>
                leftHadoopRel = fsRel
            }
          }
          if (rightRel.isInstanceOf[LogicalRelation]) {
            val logicalRel = rightRel.asInstanceOf[LogicalRelation]
            logicalRel.relation match {
              case fsRel: HadoopFsRelation =>
                rightHadoopRel = fsRel
            }
          }

          if (leftHadoopRel != null) {
            if (leftHadoopRel.bucketSpec.isDefined) {
              logDebug(s"[Left] Base Scan Table is GPT Partitioned Table: " + leftHadoopRel.bucketSpec.get.toString)
              //leftHadoopRel.schema.foreach(c => logDebug(s"\tcol: " + c.name))
              //leftHadoopRel.dataSchema.foreach(c => logDebug(s"\tdataCol: " + c.name))
              //leftHadoopRel.partitionSchema.foreach(c => logDebug(s"\tpartitionCol: " + c.name))
            } else {
              logDebug(s"[Left] Base Scan Table is Non-GPT Partitioned Table: " + left.nodeName)
            }
          }

          if (rightHadoopRel != null) {
            if (rightHadoopRel.bucketSpec.isDefined) {
              logDebug(s"[Right] Base Scan Table is GPT Partitioned Table: " + rightHadoopRel.bucketSpec.get.toString)
              //rightHadoopRel.schema.foreach(c => logDebug(s"\tcol: " + c.name))
              //rightHadoopRel.dataSchema.foreach(c => logDebug(s"\tdataCol: " + c.name))
              //rightHadoopRel.partitionSchema.foreach(c => logDebug(s"\tpartitionCol: " + c.name))
            } else {
              logDebug(s"[Right] Base Scan Table is Non-GPT Partitioned Table: " + right.nodeName)
            }
          }

          logDebug(s"BroadcastHashJoin (right): " + plan.numberedTreeString)
          leftKeys.foreach(e => e.references.foreach { a =>
            plan.addJoinType(a.name, "BroadcastHashJoinExec")
            left.addJoinType(a.name, "BroadcastHashJoinExec")
            right.addJoinType(a.name, "BroadcastHashJoinExec")
            if (leftHadoopRel != null && rightHadoopRel != null) {
              var rightPlaceToAdd = false
              var leftPlaceToAdd = false

              leftHadoopRel.schema.foreach { c =>
                //logDebug(s"Check right place to add [leftHadoopRel]: " + c.name + " == " + a.name + "?")
                if (c.name == a.name)
                  leftPlaceToAdd = true
              }

              rightHadoopRel.schema.foreach { c =>
                //logDebug(s"Check right place to add [rightHadoopRel]: " + c.name + " == " + a.name + "?")
                if (c.name == a.name)
                  rightPlaceToAdd = true
              }

              if (leftPlaceToAdd && leftHadoopRel != null)
                leftHadoopRel.addJoinType(a.name, "BroadcastHashJoinExec")

              if (rightPlaceToAdd && rightHadoopRel != null)
                rightHadoopRel.addJoinType(a.name, "BroadcastHashJoinExec")
            }
            left.addJoinCols(a.name)
          })
          rightKeys.foreach(e => e.references.foreach { a =>
            plan.addJoinType(a.name, "BroadcastHashJoinExec")
            left.addJoinType(a.name, "BroadcastHashJoinExec")
            right.addJoinType(a.name, "BroadcastHashJoinExec")
            if (leftHadoopRel != null) {
              var rightPlaceToAdd = false
              var leftPlaceToAdd = false

              leftHadoopRel.schema.foreach { c =>
                //logDebug(s"Check right place to add [leftHadoopRel]: " + c.name + " == " + a.name + "?")
                if (c.name == a.name)
                  leftPlaceToAdd = true
              }

              rightHadoopRel.schema.foreach { c =>
                //logDebug(s"Check right place to add [rightHadoopRel]: " + c.name + " == " + a.name + "?")
                if (c.name == a.name)
                  rightPlaceToAdd = true
              }

              if (leftPlaceToAdd && leftHadoopRel != null)
                leftHadoopRel.addJoinType(a.name, "BroadcastHashJoinExec")

              if (rightPlaceToAdd && rightHadoopRel != null)
                rightHadoopRel.addJoinType(a.name, "BroadcastHashJoinExec")
            }
            right.addJoinCols(a.name)
          })
        }   else {
          // the usual case: nested query
          leftKeys.foreach(e => e.references.foreach { a =>
            plan.addJoinType(a.name, "BroadcastHashJoinExec")
            left.addJoinType(a.name, "BroadcastHashJoinExec")
            right.addJoinType(a.name, "BroadcastHashJoinExec")
            left.addJoinCols(a.name)
          })

          rightKeys.foreach(e => e.references.foreach { a =>
            plan.addJoinType(a.name, "BroadcastHashJoinExec")
            left.addJoinType(a.name, "BroadcastHashJoinExec")
            right.addJoinType(a.name, "BroadcastHashJoinExec")
            right.addJoinCols(a.name)
          })
        }
        logDebug(s"BroadcastHashJoinExec is generated --> right side tbl is broadcasted [" + condition + "]")
        Seq(joins.BroadcastHashJoinExec(
          leftKeys, rightKeys, joinType, BuildRight, condition, planLater(left), planLater(right)))
      }


      case ExtractEquiJoinKeys(joinType, leftKeys, rightKeys, condition, left, right)
        if canBuildLeft(joinType) && canBroadcast(left) => {
        findScanOP(plan);
        plan.setGPTJoinType(joinType.toString)
        //logDebug(s"BroadcastHashJoin (left): " + plan.numberedTreeString)
        var anyLeftJoinCol = leftKeys(0).references.toSeq(0).name
        var anyRightJoinCol = rightKeys(0).references.toSeq(0).name

        val leftStorage = scala.collection.mutable.ArrayBuffer[LogicalRelation]()
        val rightStorage = scala.collection.mutable.ArrayBuffer[LogicalRelation]()
        findScanOP_v2(left, anyLeftJoinCol, leftStorage)
        findScanOP_v2(right, anyRightJoinCol, rightStorage)

        if(!leftStorage.isEmpty && !rightStorage.isEmpty) {

          val leftRel = leftStorage(0)
          val rightRel = rightStorage(0)
          //logDebug(s"Found LogicalRelation for leftRel: " + leftRel.nodeName)
          //logDebug(s"Found LogicalRelation for rightRel: " + rightRel.nodeName)
          var leftHadoopRel: HadoopFsRelation = null
          var rightHadoopRel: HadoopFsRelation = null

          if (leftRel.isInstanceOf[LogicalRelation]) {
            val logicalRel = leftRel.asInstanceOf[LogicalRelation]
            logicalRel.relation match {
              case fsRel: HadoopFsRelation =>
                leftHadoopRel = fsRel
            }
          }
          if (rightRel.isInstanceOf[LogicalRelation]) {
            val logicalRel = rightRel.asInstanceOf[LogicalRelation]
            logicalRel.relation match {
              case fsRel: HadoopFsRelation =>
                rightHadoopRel = fsRel
            }
          }

          if (leftHadoopRel != null) {
            if (leftHadoopRel.bucketSpec.isDefined) {
              logDebug(s"[Left] Base Scan Table is GPT Partitioned Table: " + leftHadoopRel.bucketSpec.get.toString)
              //leftHadoopRel.schema.foreach(c => logDebug(s"\tcol: " + c.name))
              //leftHadoopRel.dataSchema.foreach(c => logDebug(s"\tdataCol: " + c.name))
              //leftHadoopRel.partitionSchema.foreach(c => logDebug(s"\tpartitionCol: " + c.name))
            } else {
              logDebug(s"[Left] Base Scan Table is Non-GPT Partitioned Table: " + left.nodeName)
            }
          }

          if (rightHadoopRel != null) {
            if (rightHadoopRel.bucketSpec.isDefined) {
              logDebug(s"[Right] Base Scan Table is GPT Partitioned Table: " + rightHadoopRel.bucketSpec.get.toString)
              //rightHadoopRel.schema.foreach(c => logDebug(s"\tcol: " + c.name))
              //rightHadoopRel.dataSchema.foreach(c => logDebug(s"\tdataCol: " + c.name))
              //rightHadoopRel.partitionSchema.foreach(c => logDebug(s"\tpartitionCol: " + c.name))
            } else {
              logDebug(s"[Right] Base Scan Table is Non-GPT Partitioned Table: " + right.nodeName)
            }
          }


          leftKeys.foreach(e => e.references.foreach { a =>
            plan.addJoinType(a.name, "BroadcastHashJoinExec")
            left.addJoinType(a.name, "BroadcastHashJoinExec")
            right.addJoinType(a.name, "BroadcastHashJoinExec")
            if (leftHadoopRel != null && rightHadoopRel != null) {
              var rightPlaceToAdd = false
              var leftPlaceToAdd = false

              leftHadoopRel.schema.foreach { c =>
                //logDebug(s"Check right place to add [leftHadoopRel]: " + c.name + " == " + a.name + "?")
                if (c.name == a.name)
                  leftPlaceToAdd = true
              }

              rightHadoopRel.schema.foreach { c =>
                //logDebug(s"Check right place to add [rightHadoopRel]: " + c.name + " == " + a.name + "?")
                if (c.name == a.name)
                  rightPlaceToAdd = true
              }

              if (leftPlaceToAdd && leftHadoopRel != null)
                leftHadoopRel.addJoinType(a.name, "BroadcastHashJoinExec")

              if (rightPlaceToAdd && rightHadoopRel != null)
                rightHadoopRel.addJoinType(a.name, "BroadcastHashJoinExec")
            }
            left.addJoinCols(a.name)
          })
          rightKeys.foreach(e => e.references.foreach { a =>
            plan.addJoinType(a.name, "BroadcastHashJoinExec")
            left.addJoinType(a.name, "BroadcastHashJoinExec")
            right.addJoinType(a.name, "BroadcastHashJoinExec")
            if (leftHadoopRel != null) {
              var rightPlaceToAdd = false
              var leftPlaceToAdd = false

              leftHadoopRel.schema.foreach { c =>
                //logDebug(s"Check right place to add [leftHadoopRel]: " + c.name + " == " + a.name + "?")
                if (c.name == a.name)
                  leftPlaceToAdd = true
              }

              rightHadoopRel.schema.foreach { c =>
                //logDebug(s"Check right place to add [rightHadoopRel]: " + c.name + " == " + a.name + "?")
                if (c.name == a.name)
                  rightPlaceToAdd = true
              }

              if (leftPlaceToAdd && leftHadoopRel != null)
                leftHadoopRel.addJoinType(a.name, "BroadcastHashJoinExec")

              if (rightPlaceToAdd && rightHadoopRel != null)
                rightHadoopRel.addJoinType(a.name, "BroadcastHashJoinExec")
            }
            right.addJoinCols(a.name)
          })
        }   else {
          // the usual case: nested query
          leftKeys.foreach(e => e.references.foreach { a =>
            plan.addJoinType(a.name, "BroadcastHashJoinExec")
            left.addJoinType(a.name, "BroadcastHashJoinExec")
            right.addJoinType(a.name, "BroadcastHashJoinExec")
            left.addJoinCols(a.name)
          })

          rightKeys.foreach(e => e.references.foreach { a =>
            plan.addJoinType(a.name, "BroadcastHashJoinExec")
            left.addJoinType(a.name, "BroadcastHashJoinExec")
            right.addJoinType(a.name, "BroadcastHashJoinExec")
            right.addJoinCols(a.name)
          })
        }
        logDebug(s"BroadcastHashJoinExec is generated --> left side tbl is broadcasted [" + condition + "]")
        Seq(joins.BroadcastHashJoinExec(
          leftKeys, rightKeys, joinType, BuildLeft, condition, planLater(left), planLater(right)))
      }

      // --- ShuffledHashJoin ---------------------------------------------------------------------

      case ExtractEquiJoinKeys(joinType, leftKeys, rightKeys, condition, left, right)
         if !conf.preferSortMergeJoin && canBuildRight(joinType) && canBuildLocalHashMap(right)
           && muchSmaller(right, left) ||
           !RowOrdering.isOrderable(leftKeys) => {
        findScanOP(plan);
        plan.setGPTJoinType(joinType.toString)
        //logDebug(s"ShuffledHashJoin (right): " + plan.numberedTreeString)
        var anyLeftJoinCol = leftKeys(0).references.toSeq(0).name
        var anyRightJoinCol = rightKeys(0).references.toSeq(0).name

        val leftStorage = scala.collection.mutable.ArrayBuffer[LogicalRelation]()
        val rightStorage = scala.collection.mutable.ArrayBuffer[LogicalRelation]()
        findScanOP_v2(left, anyLeftJoinCol, leftStorage)
        findScanOP_v2(right, anyRightJoinCol, rightStorage)
        if(!leftStorage.isEmpty && !rightStorage.isEmpty) {

          val leftRel = leftStorage(0)
          val rightRel = rightStorage(0)
          //logDebug(s"Found LogicalRelation for leftRel: " + leftRel.nodeName)
          //logDebug(s"Found LogicalRelation for rightRel: " + rightRel.nodeName)
          var leftHadoopRel: HadoopFsRelation = null
          var rightHadoopRel: HadoopFsRelation = null

          if (leftRel.isInstanceOf[LogicalRelation]) {
            val logicalRel = leftRel.asInstanceOf[LogicalRelation]
            logicalRel.relation match {
              case fsRel: HadoopFsRelation =>
                leftHadoopRel = fsRel
            }
          }
          if (rightRel.isInstanceOf[LogicalRelation]) {
            val logicalRel = rightRel.asInstanceOf[LogicalRelation]
            logicalRel.relation match {
              case fsRel: HadoopFsRelation =>
                rightHadoopRel = fsRel
            }
          }

          if (leftHadoopRel != null) {
            if (leftHadoopRel.bucketSpec.isDefined) {
              logDebug(s"[Left] Base Scan Table is GPT Partitioned Table: " + leftHadoopRel.bucketSpec.get.toString)
            } else {
              logDebug(s"[Left] Base Scan Table is Non-GPT Partitioned Table: " + left.nodeName)
            }
          }

          if (rightHadoopRel != null) {
            if (rightHadoopRel.bucketSpec.isDefined) {
              logDebug(s"[Right] Base Scan Table is GPT Partitioned Table: " + rightHadoopRel.bucketSpec.get.toString)
            } else {
              logDebug(s"[Right] Base Scan Table is Non-GPT Partitioned Table: " + right.nodeName)
            }
          }

          leftKeys.foreach(e => e.references.foreach { a =>
            left.addJoinCols(a.name)
            plan.addJoinType(a.name, "ShuffledHashJoinExec")
            left.addJoinType(a.name, "ShuffledHashJoinExec")
            right.addJoinType(a.name, "ShuffledHashJoinExec")
            if (leftHadoopRel != null && rightHadoopRel != null) {
              var rightPlaceToAdd = false
              var leftPlaceToAdd = false

              leftHadoopRel.schema.foreach { c =>
                //logDebug(s"Check right place to add [leftHadoopRel]: " + c.name + " == " + a.name + "?")
                if (c.name == a.name)
                  leftPlaceToAdd = true
              }

              rightHadoopRel.schema.foreach { c =>
                //logDebug(s"Check right place to add [rightHadoopRel]: " + c.name + " == " + a.name + "?")
                if (c.name == a.name)
                  rightPlaceToAdd = true
              }

              if (leftPlaceToAdd && leftHadoopRel != null)
                leftHadoopRel.addJoinType(a.name, "ShuffledHashJoinExec")

              if (rightPlaceToAdd && rightHadoopRel != null)
                rightHadoopRel.addJoinType(a.name, "ShuffledHashJoinExec")
            }
          })
          rightKeys.foreach(e => e.references.foreach { a =>
            right.addJoinCols(a.name)
            plan.addJoinType(a.name, "ShuffledHashJoinExec")
            left.addJoinType(a.name, "ShuffledHashJoinExec")
            right.addJoinType(a.name, "ShuffledHashJoinExec")
            if (leftHadoopRel != null && rightHadoopRel != null) {
              var rightPlaceToAdd = false
              var leftPlaceToAdd = false

              leftHadoopRel.schema.foreach { c =>
                //logDebug(s"Check right place to add [leftHadoopRel]: " + c.name + " == " + a.name + "?")
                if (c.name == a.name)
                  leftPlaceToAdd = true
              }

              rightHadoopRel.schema.foreach { c =>
                //logDebug(s"Check right place to add [rightHadoopRel]: " + c.name + " == " + a.name + "?")
                if (c.name == a.name)
                  rightPlaceToAdd = true
              }

              if (leftPlaceToAdd && leftHadoopRel != null)
                leftHadoopRel.addJoinType(a.name, "ShuffledHashJoinExec")

              if (rightPlaceToAdd && rightHadoopRel != null)
                rightHadoopRel.addJoinType(a.name, "ShuffledHashJoinExec")
            }
          })
        }   else {
          // the usual case: nested query
          leftKeys.foreach(e => e.references.foreach { a =>
            plan.addJoinType(a.name, "ShuffledHashJoinExec")
            left.addJoinType(a.name, "ShuffledHashJoinExec")
            right.addJoinType(a.name, "ShuffledHashJoinExec")
            left.addJoinCols(a.name)
          })

          rightKeys.foreach(e => e.references.foreach { a =>
            plan.addJoinType(a.name, "ShuffledHashJoinExec")
            left.addJoinType(a.name, "ShuffledHashJoinExec")
            right.addJoinType(a.name, "ShuffledHashJoinExec")
            right.addJoinCols(a.name)
          })
        }
        logDebug(s"ShuffledHashJoinExec is generated --> build right tbl [" + condition + "]")
        Seq(joins.ShuffledHashJoinExec(
          leftKeys, rightKeys, joinType, BuildRight, condition, planLater(left), planLater(right)))
      }

      case ExtractEquiJoinKeys(joinType, leftKeys, rightKeys, condition, left, right)
         if !conf.preferSortMergeJoin && canBuildLeft(joinType) && canBuildLocalHashMap(left)
           && muchSmaller(left, right) ||
           !RowOrdering.isOrderable(leftKeys) => {
        findScanOP(plan);
        plan.setGPTJoinType(joinType.toString)
        //logDebug(s"ShuffledHashJoin (left): " + plan.numberedTreeString)
        var anyLeftJoinCol = leftKeys(0).references.toSeq(0).name
        var anyRightJoinCol = rightKeys(0).references.toSeq(0).name

        val leftStorage = scala.collection.mutable.ArrayBuffer[LogicalRelation]()
        val rightStorage = scala.collection.mutable.ArrayBuffer[LogicalRelation]()
        findScanOP_v2(left, anyLeftJoinCol, leftStorage)
        findScanOP_v2(right, anyRightJoinCol, rightStorage)
        if(!leftStorage.isEmpty && !rightStorage.isEmpty) {

          val leftRel = leftStorage(0)
          val rightRel = rightStorage(0)
          //logDebug(s"Found LogicalRelation for leftRel: " + leftRel.nodeName)
          //logDebug(s"Found LogicalRelation for rightRel: " + rightRel.nodeName)
          var leftHadoopRel: HadoopFsRelation = null
          var rightHadoopRel: HadoopFsRelation = null

          if (leftRel.isInstanceOf[LogicalRelation]) {
            val logicalRel = leftRel.asInstanceOf[LogicalRelation]
            logicalRel.relation match {
              case fsRel: HadoopFsRelation =>
                leftHadoopRel = fsRel
            }
          }
          if (rightRel.isInstanceOf[LogicalRelation]) {
            val logicalRel = rightRel.asInstanceOf[LogicalRelation]
            logicalRel.relation match {
              case fsRel: HadoopFsRelation =>
                rightHadoopRel = fsRel
            }
          }

          if (leftHadoopRel != null) {
            if (leftHadoopRel.bucketSpec.isDefined) {
              logDebug(s"[Left] Base Scan Table is GPT Partitioned Table: " + leftHadoopRel.bucketSpec.get.toString)
            } else {
              logDebug(s"[Left] Base Scan Table is Non-GPT Partitioned Table: " + left.nodeName)
            }
          }

          if (rightHadoopRel != null) {
            if (rightHadoopRel.bucketSpec.isDefined) {
              logDebug(s"[Right] Base Scan Table is GPT Partitioned Table: " + rightHadoopRel.bucketSpec.get.toString)
            } else {
              logDebug(s"[Right] Base Scan Table is Non-GPT Partitioned Table: " + right.nodeName)
            }
          }
          leftKeys.foreach(e => e.references.foreach { a =>
            plan.addJoinType(a.name, "ShuffledHashJoinExec")
            left.addJoinType(a.name, "ShuffledHashJoinExec")
            right.addJoinType(a.name, "ShuffledHashJoinExec")
            if (leftHadoopRel != null && rightHadoopRel != null) {
              var rightPlaceToAdd = false
              var leftPlaceToAdd = false

              leftHadoopRel.schema.foreach { c =>
                //logDebug(s"Check right place to add [leftHadoopRel]: " + c.name + " == " + a.name + "?")
                if (c.name == a.name)
                  leftPlaceToAdd = true
              }

              rightHadoopRel.schema.foreach { c =>
                //logDebug(s"Check right place to add [rightHadoopRel]: " + c.name + " == " + a.name + "?")
                if (c.name == a.name)
                  rightPlaceToAdd = true
              }

              if (leftPlaceToAdd && leftHadoopRel != null)
                leftHadoopRel.addJoinType(a.name, "ShuffledHashJoinExec")

              if (rightPlaceToAdd && rightHadoopRel != null)
                rightHadoopRel.addJoinType(a.name, "ShuffledHashJoinExec")
            }
            left.addJoinCols(a.name)
          })
          rightKeys.foreach(e => e.references.foreach { a =>
            plan.addJoinType(a.name, "ShuffledHashJoinExec")
            left.addJoinType(a.name, "ShuffledHashJoinExec")
            right.addJoinType(a.name, "ShuffledHashJoinExec")
            if (leftHadoopRel != null && rightHadoopRel != null) {
              var rightPlaceToAdd = false
              var leftPlaceToAdd = false

              leftHadoopRel.schema.foreach { c =>
                //logDebug(s"Check right place to add [leftHadoopRel]: " + c.name + " == " + a.name + "?")
                if (c.name == a.name)
                  leftPlaceToAdd = true
              }

              rightHadoopRel.schema.foreach { c =>
                //logDebug(s"Check right place to add [rightHadoopRel]: " + c.name + " == " + a.name + "?")
                if (c.name == a.name)
                  rightPlaceToAdd = true
              }

              if (leftPlaceToAdd && leftHadoopRel != null)
                leftHadoopRel.addJoinType(a.name, "ShuffledHashJoinExec")

              if (rightPlaceToAdd && rightHadoopRel != null)
                rightHadoopRel.addJoinType(a.name, "ShuffledHashJoinExec")
            }
            right.addJoinCols(a.name)
          })
        }  else {
          // the usual case: nested query
          leftKeys.foreach(e => e.references.foreach { a =>
            plan.addJoinType(a.name, "ShuffledHashJoinExec")
            left.addJoinType(a.name, "ShuffledHashJoinExec")
            right.addJoinType(a.name, "ShuffledHashJoinExec")
            left.addJoinCols(a.name)
          })

          rightKeys.foreach(e => e.references.foreach { a =>
            plan.addJoinType(a.name, "ShuffledHashJoinExec")
            left.addJoinType(a.name, "ShuffledHashJoinExec")
            right.addJoinType(a.name, "ShuffledHashJoinExec")
            right.addJoinCols(a.name)
          })
        }
        logDebug(s"ShuffledHashJoinExec is generated --> build left tbl [" + condition + "]")
        Seq(joins.ShuffledHashJoinExec(
          leftKeys, rightKeys, joinType, BuildLeft, condition, planLater(left), planLater(right)))
      }

      // --- SortMergeJoin ------------------------------------------------------------

      case ExtractEquiJoinKeys(joinType, leftKeys, rightKeys, condition, left, right)
        if RowOrdering.isOrderable(leftKeys) => {
        findScanOP(plan);
        plan.setGPTJoinType(joinType.toString)
        //logDebug(s"SortMergeJoin: " + plan.numberedTreeString)
        var anyLeftJoinCol = leftKeys(0).references.toSeq(0).name
        var anyRightJoinCol = rightKeys(0).references.toSeq(0).name

        val leftStorage = scala.collection.mutable.ArrayBuffer[LogicalRelation]()
        val rightStorage = scala.collection.mutable.ArrayBuffer[LogicalRelation]()
        findScanOP_v2(left, anyLeftJoinCol, leftStorage)
        findScanOP_v2(right, anyRightJoinCol, rightStorage)

        if(!leftStorage.isEmpty && !rightStorage.isEmpty) {

          val leftRel = leftStorage(0)
          val rightRel = rightStorage(0)
          //logDebug(s"Found LogicalRelation for leftRel: " + leftRel.nodeName)
          //logDebug(s"Found LogicalRelation for rightRel: " + rightRel.nodeName)
          var leftHadoopRel: HadoopFsRelation = null
          var rightHadoopRel: HadoopFsRelation = null

          if (leftRel.isInstanceOf[LogicalRelation]) {
            val logicalRel = leftRel.asInstanceOf[LogicalRelation]
            logicalRel.relation match {
              case fsRel: HadoopFsRelation =>
                leftHadoopRel = fsRel
            }
          }
          if (rightRel.isInstanceOf[LogicalRelation]) {
            val logicalRel = rightRel.asInstanceOf[LogicalRelation]
            logicalRel.relation match {
              case fsRel: HadoopFsRelation =>
                rightHadoopRel = fsRel
            }
          }

          if (leftHadoopRel != null) {
            if (leftHadoopRel.bucketSpec.isDefined) {
              logDebug(s"[Left] Base Scan Table is GPT Partitioned Table: " + leftHadoopRel.bucketSpec.get.toString)
            } else {
              logDebug(s"[Left] Base Scan Table is Non-GPT Partitioned Table: " + left.nodeName)
            }
          }

          if (rightHadoopRel != null) {
            if (rightHadoopRel.bucketSpec.isDefined) {
              logDebug(s"[Right] Base Scan Table is GPT Partitioned Table: " + rightHadoopRel.bucketSpec.get.toString)
            } else {
              logDebug(s"[Right] Base Scan Table is Non-GPT Partitioned Table: " + right.nodeName)
            }
          }
          leftKeys.foreach(e => e.references.foreach { a =>
            plan.addJoinType(a.name, "SortMergeJoinExec")
            left.addJoinType(a.name, "SortMergeJoinExec")
            right.addJoinType(a.name, "SortMergeJoinExec")
            if (leftHadoopRel != null && rightHadoopRel != null) {
              var rightPlaceToAdd = false
              var leftPlaceToAdd = false

              leftHadoopRel.schema.foreach { c =>
                //logDebug(s"Check right place to add [leftHadoopRel]: " + c.name + " == " + a.name + "?")
                if (c.name == a.name)
                  leftPlaceToAdd = true
              }

              rightHadoopRel.schema.foreach { c =>
                //logDebug(s"Check right place to add [rightHadoopRel]: " + c.name + " == " + a.name + "?")
                if (c.name == a.name)
                  rightPlaceToAdd = true
              }

              if (leftPlaceToAdd && leftHadoopRel != null)
                leftHadoopRel.addJoinType(a.name, "SortMergeJoinExec")

              if (rightPlaceToAdd && rightHadoopRel != null)
                rightHadoopRel.addJoinType(a.name, "SortMergeJoinExec")
            }
            left.addJoinCols(a.name)
          })
          rightKeys.foreach(e => e.references.foreach { a =>
            plan.addJoinType(a.name, "SortMergeJoinExec")
            left.addJoinType(a.name, "SortMergeJoinExec")
            right.addJoinType(a.name, "SortMergeJoinExec")
            if (leftHadoopRel != null && rightHadoopRel != null) {
              var rightPlaceToAdd = false
              var leftPlaceToAdd = false

              leftHadoopRel.schema.foreach { c =>
                //logDebug(s"Check right place to add [leftHadoopRel]: " + c.name + " == " + a.name + "?")
                if (c.name == a.name)
                  leftPlaceToAdd = true
              }

              rightHadoopRel.schema.foreach { c =>
                //logDebug(s"Check right place to add [rightHadoopRel]: " + c.name + " == " + a.name + "?")
                if (c.name == a.name)
                  rightPlaceToAdd = true
              }

              if (leftPlaceToAdd && leftHadoopRel != null)
                leftHadoopRel.addJoinType(a.name, "SortMergeJoinExec")

              if (rightPlaceToAdd && rightHadoopRel != null)
                rightHadoopRel.addJoinType(a.name, "SortMergeJoinExec")
            }
            right.addJoinCols(a.name)
          })
        } else {
          // the usual case: nested query
          leftKeys.foreach(e => e.references.foreach { a =>
            plan.addJoinType(a.name, "SortMergeJoinExec")
            left.addJoinType(a.name, "SortMergeJoinExec")
            right.addJoinType(a.name, "SortMergeJoinExec")
            left.addJoinCols(a.name)
          })

          rightKeys.foreach(e => e.references.foreach { a =>
            plan.addJoinType(a.name, "SortMergeJoinExec")
            left.addJoinType(a.name, "SortMergeJoinExec")
            right.addJoinType(a.name, "SortMergeJoinExec")
            right.addJoinCols(a.name)
          })
        }
        logDebug(s"SortMergeJoinExec is generated [" + condition + "]")
        joins.SortMergeJoinExec(
          leftKeys, rightKeys, joinType, condition, planLater(left), planLater(right)) :: Nil
      }

      // --- Without joining keys ------------------------------------------------------------

      // Pick BroadcastNestedLoopJoin if one side could be broadcasted
      case j @ logical.Join(left, right, joinType, condition)
          if canBuildRight(joinType) && canBroadcast(right) => {
        plan.setGPTJoinType(joinType.toString)
        logDebug(s"BroadcastNestedLoopJoinExec (broadcast right): " + plan.numberedTreeString)
        val leftRel = findScanOP(left)
        val rightRel = findScanOP(right)
        var leftHadoopRel : HadoopFsRelation = null
        var rightHadoopRel : HadoopFsRelation = null

        if(leftRel.isInstanceOf[LogicalRelation]) {
          val logicalRel = leftRel.asInstanceOf[LogicalRelation]
          logicalRel.relation match {
            case fsRel: HadoopFsRelation =>
              leftHadoopRel = fsRel
          }
        }
        if(rightRel.isInstanceOf[LogicalRelation]) {
          val logicalRel = rightRel.asInstanceOf[LogicalRelation]
          logicalRel.relation match {
            case fsRel: HadoopFsRelation =>
              rightHadoopRel = fsRel
          }
        }

        if(leftHadoopRel != null) {
          if(leftHadoopRel.bucketSpec.isDefined) {
            logDebug(s"[Left] Base Scan Table is GPT Partitioned Table: " + leftHadoopRel.bucketSpec.get.toString)
          } else {
            logDebug(s"[Left] Base Scan Table is Non-GPT Partitioned Table: " + left.nodeName)
          }
        }

        if(rightHadoopRel != null) {
          if(rightHadoopRel.bucketSpec.isDefined) {
            logDebug(s"[Right] Base Scan Table is GPT Partitioned Table: " + rightHadoopRel.bucketSpec.get.toString)
          } else {
            logDebug(s"[Right] Base Scan Table is Non-GPT Partitioned Table: " + right.nodeName)
          }
        }
        joins.BroadcastNestedLoopJoinExec(
          planLater(left), planLater(right), BuildRight, joinType, condition) :: Nil
      }

      case j @ logical.Join(left, right, joinType, condition)
          if canBuildLeft(joinType) && canBroadcast(left) => {
        plan.setGPTJoinType(joinType.toString)
        logDebug(s"BroadcastNestedLoopJoinExec (broadcast left): " + plan.numberedTreeString)
        val leftRel = findScanOP(left)
        val rightRel = findScanOP(right)
        var leftHadoopRel : HadoopFsRelation = null
        var rightHadoopRel : HadoopFsRelation = null

        if(leftRel.isInstanceOf[LogicalRelation]) {
          val logicalRel = leftRel.asInstanceOf[LogicalRelation]
          logicalRel.relation match {
            case fsRel: HadoopFsRelation =>
              leftHadoopRel = fsRel
          }
        }
        if(rightRel.isInstanceOf[LogicalRelation]) {
          val logicalRel = rightRel.asInstanceOf[LogicalRelation]
          logicalRel.relation match {
            case fsRel: HadoopFsRelation =>
              rightHadoopRel = fsRel
          }
        }

        if(leftHadoopRel != null) {
          if(leftHadoopRel.bucketSpec.isDefined) {
            logDebug(s"[Left] Base Scan Table is GPT Partitioned Table: " + leftHadoopRel.bucketSpec.get.toString)
          } else {
            logDebug(s"[Left] Base Scan Table is Non-GPT Partitioned Table: " + left.nodeName)
          }
        }

        if(rightHadoopRel != null) {
          if(rightHadoopRel.bucketSpec.isDefined) {
            logDebug(s"[Right] Base Scan Table is GPT Partitioned Table: " + rightHadoopRel.bucketSpec.get.toString)
          } else {
            logDebug(s"[Right] Base Scan Table is Non-GPT Partitioned Table: " + right.nodeName)
          }
        }
        joins.BroadcastNestedLoopJoinExec(
          planLater(left), planLater(right), BuildLeft, joinType, condition) :: Nil
      }

      // Pick CartesianProduct for InnerJoin
      case logical.Join(left, right, _: InnerLike, condition) => {
        plan.setGPTJoinType("InnerLike")
        logDebug(s"CartesianProduct: " + plan.numberedTreeString)
        val leftRel = findScanOP(left)
        val rightRel = findScanOP(right)
        var leftHadoopRel : HadoopFsRelation = null
        var rightHadoopRel : HadoopFsRelation = null

        if(leftRel.isInstanceOf[LogicalRelation]) {
          val logicalRel = leftRel.asInstanceOf[LogicalRelation]
          logicalRel.relation match {
            case fsRel: HadoopFsRelation =>
              leftHadoopRel = fsRel
          }
        }
        if(rightRel.isInstanceOf[LogicalRelation]) {
          val logicalRel = rightRel.asInstanceOf[LogicalRelation]
          logicalRel.relation match {
            case fsRel: HadoopFsRelation =>
              rightHadoopRel = fsRel
          }
        }

        if(leftHadoopRel != null) {
          if(leftHadoopRel.bucketSpec.isDefined) {
            logDebug(s"[Left] Base Scan Table is GPT Partitioned Table: " + leftHadoopRel.bucketSpec.get.toString)
          } else {
            logDebug(s"[Left] Base Scan Table is Non-GPT Partitioned Table: " + left.nodeName)
          }
        }

        if(rightHadoopRel != null) {
          if(rightHadoopRel.bucketSpec.isDefined) {
            logDebug(s"[Right] Base Scan Table is GPT Partitioned Table: " + rightHadoopRel.bucketSpec.get.toString)
          } else {
            logDebug(s"[Right] Base Scan Table is Non-GPT Partitioned Table: " + right.nodeName)
          }
        }
        joins.CartesianProductExec(planLater(left), planLater(right), condition) :: Nil
      }

      case logical.Join(left, right, joinType, condition) => {
        plan.setGPTJoinType(joinType.toString)
        var buildSideStr = ""
        val buildSide =
          if (right.stats(conf).sizeInBytes <= left.stats(conf).sizeInBytes) {
            buildSideStr = "BuildRight"
            BuildRight
          } else {
            buildSideStr = "BuildLeft"
            BuildLeft
          }
        logDebug(s"BroadcastNestedLoopJoinExec (buildSideStr): " + plan.numberedTreeString)
        val leftRel = findScanOP(left)
        val rightRel = findScanOP(right)
        var leftHadoopRel : HadoopFsRelation = null
        var rightHadoopRel : HadoopFsRelation = null

        if(leftRel.isInstanceOf[LogicalRelation]) {
          val logicalRel = leftRel.asInstanceOf[LogicalRelation]
          logicalRel.relation match {
            case fsRel: HadoopFsRelation =>
              leftHadoopRel = fsRel
          }
        }
        if(rightRel.isInstanceOf[LogicalRelation]) {
          val logicalRel = rightRel.asInstanceOf[LogicalRelation]
          logicalRel.relation match {
            case fsRel: HadoopFsRelation =>
              rightHadoopRel = fsRel
          }
        }

        if(leftHadoopRel != null) {
          if(leftHadoopRel.bucketSpec.isDefined) {
            logDebug(s"[Left] Base Scan Table is GPT Partitioned Table: " + leftHadoopRel.bucketSpec.get.toString)
          } else {
            logDebug(s"[Left] Base Scan Table is Non-GPT Partitioned Table: " + left.nodeName)
          }
        }

        if(rightHadoopRel != null) {
          if(rightHadoopRel.bucketSpec.isDefined) {
            logDebug(s"[Right] Base Scan Table is GPT Partitioned Table: " + rightHadoopRel.bucketSpec.get.toString)
          } else {
            logDebug(s"[Right] Base Scan Table is Non-GPT Partitioned Table: " + right.nodeName)
          }
        }
        // This join could be very slow or OOM
        joins.BroadcastNestedLoopJoinExec(
          planLater(left), planLater(right), buildSide, joinType, condition) :: Nil
      }

      // --- Cases where this strategy does not apply ---------------------------------------------

      case _ => Nil
    }
  }

  /**
   * Used to plan aggregation queries that are computed incrementally as part of a
   * [[StreamingQuery]]. Currently this rule is injected into the planner
   * on-demand, only when planning in a [[org.apache.spark.sql.execution.streaming.StreamExecution]]
   */
  object StatefulAggregationStrategy extends Strategy {
    override def apply(plan: LogicalPlan): Seq[SparkPlan] = plan match {
      case EventTimeWatermark(columnName, delay, child) =>
        EventTimeWatermarkExec(columnName, delay, planLater(child)) :: Nil

      case PhysicalAggregation(
        namedGroupingExpressions, aggregateExpressions, rewrittenResultExpressions, child) =>

        aggregate.AggUtils.planStreamingAggregation(
          namedGroupingExpressions,
          aggregateExpressions,
          rewrittenResultExpressions,
          planLater(child))

      case _ => Nil
    }
  }

  /**
   * Used to plan the streaming deduplicate operator.
   */
  object StreamingDeduplicationStrategy extends Strategy {
    override def apply(plan: LogicalPlan): Seq[SparkPlan] = plan match {
      case Deduplicate(keys, child, true) =>
        StreamingDeduplicateExec(keys, planLater(child)) :: Nil

      case _ => Nil
    }
  }

  /**
   * Used to plan the aggregate operator for expressions based on the AggregateFunction2 interface.
   */
  object Aggregation extends Strategy {
    def apply(plan: LogicalPlan): Seq[SparkPlan] = plan match {
      case PhysicalAggregation(
          groupingExpressions, aggregateExpressions, resultExpressions, child) =>

        val (functionsWithDistinct, functionsWithoutDistinct) =
          aggregateExpressions.partition(_.isDistinct)
        if (functionsWithDistinct.map(_.aggregateFunction.children).distinct.length > 1) {
          // This is a sanity check. We should not reach here when we have multiple distinct
          // column sets. Our MultipleDistinctRewriter should take care this case.
          sys.error("You hit a query analyzer bug. Please report your query to " +
              "Spark user mailing list.")
        }

        val aggregateOperator =
          if (functionsWithDistinct.isEmpty) {
            aggregate.AggUtils.planAggregateWithoutDistinct(
              groupingExpressions,
              aggregateExpressions,
              resultExpressions,
              planLater(child))
          } else {
            aggregate.AggUtils.planAggregateWithOneDistinct(
              groupingExpressions,
              functionsWithDistinct,
              functionsWithoutDistinct,
              resultExpressions,
              planLater(child))
          }

        aggregateOperator

      case _ => Nil
    }
  }

  protected lazy val singleRowRdd = sparkContext.parallelize(Seq(InternalRow()), 1)

  object InMemoryScans extends Strategy {
    def apply(plan: LogicalPlan): Seq[SparkPlan] = plan match {
      case PhysicalOperation(projectList, filters, mem: InMemoryRelation) => {
        logDebug("InMemoryScans in PhysicalOperation: " + mem.tableName)
        pruneFilterProject(
          projectList,
          filters,
          identity[Seq[Expression]], // All filters still need to be evaluated.
          InMemoryTableScanExec(_, filters, mem)) :: Nil
      }
      case _ => Nil
    }
  }

  /**
   * This strategy is just for explaining `Dataset/DataFrame` created by `spark.readStream`.
   * It won't affect the execution, because `StreamingRelation` will be replaced with
   * `StreamingExecutionRelation` in `StreamingQueryManager` and `StreamingExecutionRelation` will
   * be replaced with the real relation using the `Source` in `StreamExecution`.
   */
  object StreamingRelationStrategy extends Strategy {
    def apply(plan: LogicalPlan): Seq[SparkPlan] = plan match {
      case s: StreamingRelation =>
        StreamingRelationExec(s.sourceName, s.output) :: Nil
      case s: StreamingExecutionRelation =>
        StreamingRelationExec(s.toString, s.output) :: Nil
      case _ => Nil
    }
  }

  /**
   * Strategy to convert [[FlatMapGroupsWithState]] logical operator to physical operator
   * in streaming plans. Conversion for batch plans is handled by [[BasicOperators]].
   */
  object FlatMapGroupsWithStateStrategy extends Strategy {
    override def apply(plan: LogicalPlan): Seq[SparkPlan] = plan match {
      case FlatMapGroupsWithState(
        func, keyDeser, valueDeser, groupAttr, dataAttr, outputAttr, stateEnc, outputMode, _,
        timeout, child) =>
        val execPlan = FlatMapGroupsWithStateExec(
          func, keyDeser, valueDeser, groupAttr, dataAttr, outputAttr, None, stateEnc, outputMode,
          timeout, batchTimestampMs = None, eventTimeWatermark = None, planLater(child))
        execPlan :: Nil
      case _ =>
        Nil
    }
  }

  // Can we automate these 'pass through' operations?
  object BasicOperators extends Strategy {
    def apply(plan: LogicalPlan): Seq[SparkPlan] = plan match {
      case r: RunnableCommand => ExecutedCommandExec(r) :: Nil

      case MemoryPlan(sink, output) =>
        val encoder = RowEncoder(sink.schema)
        LocalTableScanExec(output, sink.allData.map(r => encoder.toRow(r).copy())) :: Nil

      case logical.Distinct(child) =>
        throw new IllegalStateException(
          "logical distinct operator should have been replaced by aggregate in the optimizer")
      case logical.Intersect(left, right) =>
        throw new IllegalStateException(
          "logical intersect operator should have been replaced by semi-join in the optimizer")
      case logical.Except(left, right) =>
        throw new IllegalStateException(
          "logical except operator should have been replaced by anti-join in the optimizer")

      case logical.DeserializeToObject(deserializer, objAttr, child) =>
        execution.DeserializeToObjectExec(deserializer, objAttr, planLater(child)) :: Nil
      case logical.SerializeFromObject(serializer, child) =>
        execution.SerializeFromObjectExec(serializer, planLater(child)) :: Nil
      case logical.MapPartitions(f, objAttr, child) =>
        execution.MapPartitionsExec(f, objAttr, planLater(child)) :: Nil
      case logical.MapPartitionsInR(f, p, b, is, os, objAttr, child) =>
        execution.MapPartitionsExec(
          execution.r.MapPartitionsRWrapper(f, p, b, is, os), objAttr, planLater(child)) :: Nil
      case logical.FlatMapGroupsInR(f, p, b, is, os, key, value, grouping, data, objAttr, child) =>
        execution.FlatMapGroupsInRExec(f, p, b, is, os, key, value, grouping,
          data, objAttr, planLater(child)) :: Nil
      case logical.MapElements(f, _, _, objAttr, child) =>
        execution.MapElementsExec(f, objAttr, planLater(child)) :: Nil
      case logical.AppendColumns(f, _, _, in, out, child) =>
        execution.AppendColumnsExec(f, in, out, planLater(child)) :: Nil
      case logical.AppendColumnsWithObject(f, childSer, newSer, child) =>
        execution.AppendColumnsWithObjectExec(f, childSer, newSer, planLater(child)) :: Nil
      case logical.MapGroups(f, key, value, grouping, data, objAttr, child) =>
        execution.MapGroupsExec(f, key, value, grouping, data, objAttr, planLater(child)) :: Nil
      case logical.FlatMapGroupsWithState(
          f, key, value, grouping, data, output, _, _, _, timeout, child) =>
        execution.MapGroupsExec(
          f, key, value, grouping, data, output, timeout, planLater(child)) :: Nil
      case logical.CoGroup(f, key, lObj, rObj, lGroup, rGroup, lAttr, rAttr, oAttr, left, right) =>
        execution.CoGroupExec(
          f, key, lObj, rObj, lGroup, rGroup, lAttr, rAttr, oAttr,
          planLater(left), planLater(right)) :: Nil

      case logical.Repartition(numPartitions, shuffle, child) =>
        if (shuffle) {
          ShuffleExchange(RoundRobinPartitioning(numPartitions), planLater(child)) :: Nil
        } else {
          execution.CoalesceExec(numPartitions, planLater(child)) :: Nil
        }
      case logical.Sort(sortExprs, global, child) =>
        execution.SortExec(sortExprs, global, planLater(child)) :: Nil
      case logical.Project(projectList, child) =>
        execution.ProjectExec(projectList, planLater(child)) :: Nil
      case logical.Filter(condition, child) =>
        execution.FilterExec(condition, planLater(child)) :: Nil
      case f: logical.TypedFilter =>
        execution.FilterExec(f.typedCondition(f.deserializer), planLater(f.child)) :: Nil
      case e @ logical.Expand(_, _, child) =>
        execution.ExpandExec(e.projections, e.output, planLater(child)) :: Nil
      case logical.Window(windowExprs, partitionSpec, orderSpec, child) =>
        execution.window.WindowExec(windowExprs, partitionSpec, orderSpec, planLater(child)) :: Nil
      case logical.Sample(lb, ub, withReplacement, seed, child) =>
        execution.SampleExec(lb, ub, withReplacement, seed, planLater(child)) :: Nil
      case logical.LocalRelation(output, data) =>
        LocalTableScanExec(output, data) :: Nil
      case logical.LocalLimit(IntegerLiteral(limit), child) =>
        execution.LocalLimitExec(limit, planLater(child)) :: Nil
      case logical.GlobalLimit(IntegerLiteral(limit), child) =>
        execution.GlobalLimitExec(limit, planLater(child)) :: Nil
      case logical.Union(unionChildren) =>
        execution.UnionExec(unionChildren.map(planLater)) :: Nil
      case g @ logical.Generate(generator, join, outer, _, _, child) =>
        execution.GenerateExec(
          generator, join = join, outer = outer, g.qualifiedGeneratorOutput,
          planLater(child)) :: Nil
      case logical.OneRowRelation =>
        execution.RDDScanExec(Nil, singleRowRdd, "OneRowRelation") :: Nil
      case r: logical.Range =>
        execution.RangeExec(r) :: Nil
      case logical.RepartitionByExpression(expressions, child, numPartitions) =>
        exchange.ShuffleExchange(HashPartitioning(
          expressions, numPartitions), planLater(child)) :: Nil
      case ExternalRDD(outputObjAttr, rdd) => ExternalRDDScanExec(outputObjAttr, rdd) :: Nil
      case r: LogicalRDD =>
        RDDScanExec(r.output, r.rdd, "ExistingRDD", r.outputPartitioning, r.outputOrdering) :: Nil
      case h: ResolvedHint => planLater(h.child) :: Nil
      case _ => Nil
    }
  }
}
