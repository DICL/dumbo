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

package org.apache.spark.sql.execution.datasources

import java.util.regex.Pattern

import scala.util.control.NonFatal

import org.apache.spark.sql.{AnalysisException, SaveMode, SparkSession}
import org.apache.spark.sql.catalyst.TableIdentifier
import org.apache.spark.sql.catalyst.analysis._
import org.apache.spark.sql.catalyst.catalog.{BucketSpec, CatalogRelation, CatalogTable, SessionCatalog}
import org.apache.spark.sql.catalyst.expressions.{Alias, Attribute, Cast, RowOrdering}
import org.apache.spark.sql.catalyst.plans.logical
import org.apache.spark.sql.catalyst.plans.logical._
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.execution.command.DDLUtils
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.sources.{BaseRelation, InsertableRelation}
import org.apache.spark.sql.types.{AtomicType, StructType}

/**
 * Try to replaces [[UnresolvedRelation]]s with [[ResolveDataSource]].
 */
class ResolveDataSource(sparkSession: SparkSession) extends Rule[LogicalPlan] {
  def apply(plan: LogicalPlan): LogicalPlan = plan resolveOperators {
    case u: UnresolvedRelation if u.tableIdentifier.database.isDefined =>
      try {
        val dataSource = DataSource(
          sparkSession,
          paths = u.tableIdentifier.table :: Nil,
          className = u.tableIdentifier.database.get)

        val notSupportDirectQuery = try {
          !classOf[FileFormat].isAssignableFrom(dataSource.providingClass)
        } catch {
          case NonFatal(e) => false
        }
        if (notSupportDirectQuery) {
          throw new AnalysisException("Unsupported data source type for direct query on files: " +
            s"${u.tableIdentifier.database.get}")
        }
        val plan = LogicalRelation(dataSource.resolveRelation())
        u.alias.map(a => SubqueryAlias(u.alias.get, plan, None)).getOrElse(plan)
      } catch {
        case e: ClassNotFoundException => u
        case e: Exception =>
          // the provider is valid, but failed to create a logical plan
          u.failAnalysis(e.getMessage)
      }
  }
}

/**
 * Analyze [[CreateTable]] and do some normalization and checking.
 * For CREATE TABLE AS SELECT, the SELECT query is also analyzed.
 */
case class AnalyzeCreateTable(sparkSession: SparkSession) extends Rule[LogicalPlan] {

  def apply(plan: LogicalPlan): LogicalPlan = plan transform {
    // When we CREATE TABLE without specifying the table schema, we should fail the query if
    // bucketing information is specified, as we can't infer bucketing from data files currently.
    // Since the runtime inferred partition columns could be different from what user specified,
    // we fail the query if the partitioning information is specified.
    case c @ CreateTable(tableDesc, _, None) if tableDesc.schema.isEmpty =>
      if (tableDesc.bucketSpec.isDefined) {
        failAnalysis("Cannot specify bucketing information if the table schema is not specified " +
          "when creating and will be inferred at runtime")
      }
      if (tableDesc.partitionColumnNames.nonEmpty) {
        failAnalysis("It is not allowed to specify partition columns when the table schema is " +
          "not defined. When the table schema is not provided, schema and partition columns " +
          "will be inferred.")
      }
      c

    // Here we normalize partition, bucket and sort column names, w.r.t. the case sensitivity
    // config, and do various checks:
    //   * column names in table definition can't be duplicated.
    //   * partition, bucket and sort column names must exist in table definition.
    //   * partition, bucket and sort column names can't be duplicated.
    //   * can't use all table columns as partition columns.
    //   * partition columns' type must be AtomicType.
    //   * sort columns' type must be orderable.
    case c @ CreateTable(tableDesc, mode, query) =>
      val analyzedQuery = query.map { q =>
        // Analyze the query in CTAS and then we can do the normalization and checking.
        val qe = sparkSession.sessionState.executePlan(q)
        qe.assertAnalyzed()
        qe.analyzed
      }
      val schema = if (analyzedQuery.isDefined) {
        analyzedQuery.get.schema
      } else {
        tableDesc.schema
      }
      val columnNames = if (sparkSession.sessionState.conf.caseSensitiveAnalysis) {
        schema.map(_.name)
      } else {
        schema.map(_.name.toLowerCase)
      }
      checkDuplication(columnNames, "table definition of " + tableDesc.identifier)

      val partitionColsChecked = checkPartitionColumns(schema, tableDesc)
      val bucketColsChecked = checkBucketColumns(schema, partitionColsChecked)
      c.copy(tableDesc = bucketColsChecked, query = analyzedQuery)
  }

  private def checkPartitionColumns(schema: StructType, tableDesc: CatalogTable): CatalogTable = {
    val normalizedPartitionCols = tableDesc.partitionColumnNames.map { colName =>
      normalizeColumnName(tableDesc.identifier, schema, colName, "partition")
    }
    checkDuplication(normalizedPartitionCols, "partition")

    if (schema.nonEmpty && normalizedPartitionCols.length == schema.length) {
      if (tableDesc.provider.get == DDLUtils.HIVE_PROVIDER) {
        // When we hit this branch, it means users didn't specify schema for the table to be
        // created, as we always include partition columns in table schema for hive serde tables.
        // The real schema will be inferred at hive metastore by hive serde, plus the given
        // partition columns, so we should not fail the analysis here.
      } else {
        failAnalysis("Cannot use all columns for partition columns")
      }

    }

    schema.filter(f => normalizedPartitionCols.contains(f.name)).map(_.dataType).foreach {
      case _: AtomicType => // OK
      case other => failAnalysis(s"Cannot use ${other.simpleString} for partition column")
    }

    tableDesc.copy(partitionColumnNames = normalizedPartitionCols)
  }

  private def checkBucketColumns(schema: StructType, tableDesc: CatalogTable): CatalogTable = {
    tableDesc.bucketSpec match {
      case Some(BucketSpec(numBuckets, bucketColumnNames, sortColumnNames)) =>
        val normalizedBucketCols = bucketColumnNames.map { colName =>
          normalizeColumnName(tableDesc.identifier, schema, colName, "bucket")
        }
        checkDuplication(normalizedBucketCols, "bucket")

        val normalizedSortCols = sortColumnNames.map { colName =>
          normalizeColumnName(tableDesc.identifier, schema, colName, "sort")
        }
        checkDuplication(normalizedSortCols, "sort")

        schema.filter(f => normalizedSortCols.contains(f.name)).map(_.dataType).foreach {
          case dt if RowOrdering.isOrderable(dt) => // OK
          case other => failAnalysis(s"Cannot use ${other.simpleString} for sorting column")
        }

        tableDesc.copy(
          bucketSpec = Some(BucketSpec(numBuckets, normalizedBucketCols, normalizedSortCols))
        )

      case None => tableDesc
    }
  }

  private def checkDuplication(colNames: Seq[String], colType: String): Unit = {
    if (colNames.distinct.length != colNames.length) {
      val duplicateColumns = colNames.groupBy(identity).collect {
        case (x, ys) if ys.length > 1 => x
      }
      failAnalysis(s"Found duplicate column(s) in $colType: ${duplicateColumns.mkString(", ")}")
    }
  }

  private def normalizeColumnName(
      tableIdent: TableIdentifier,
      schema: StructType,
      colName: String,
      colType: String): String = {
    val tableCols = schema.map(_.name)
    val resolver = sparkSession.sessionState.conf.resolver
    tableCols.find(resolver(_, colName)).getOrElse {
      failAnalysis(s"$colType column $colName is not defined in table $tableIdent, " +
        s"defined table columns are: ${tableCols.mkString(", ")}")
    }
  }

  private def failAnalysis(msg: String) = throw new AnalysisException(msg)
}

/**
 * Preprocess the [[InsertIntoTable]] plan. Throws exception if the number of columns mismatch, or
 * specified partition columns are different from the existing partition columns in the target
 * table. It also does data type casting and field renaming, to make sure that the columns to be
 * inserted have the correct data type and fields have the correct names.
 */
case class PreprocessTableInsertion(conf: SQLConf) extends Rule[LogicalPlan] {
  private def preprocess(
      insert: InsertIntoTable,
      tblName: String,
      partColNames: Seq[String]): InsertIntoTable = {

    val normalizedPartSpec = PartitioningUtils.normalizePartitionSpec(
      insert.partition, partColNames, tblName, conf.resolver)

    val expectedColumns = {
      val staticPartCols = normalizedPartSpec.filter(_._2.isDefined).keySet
      insert.table.output.filterNot(a => staticPartCols.contains(a.name))
    }

    if (expectedColumns.length != insert.child.schema.length) {
      throw new AnalysisException(
        s"Cannot insert into table $tblName because the number of columns are different: " +
          s"need ${expectedColumns.length} columns, " +
          s"but query has ${insert.child.schema.length} columns.")
    }

    if (normalizedPartSpec.nonEmpty) {
      if (normalizedPartSpec.size != partColNames.length) {
        throw new AnalysisException(
          s"""
             |Requested partitioning does not match the table $tblName:
             |Requested partitions: ${normalizedPartSpec.keys.mkString(",")}
             |Table partitions: ${partColNames.mkString(",")}
           """.stripMargin)
      }

      castAndRenameChildOutput(insert.copy(partition = normalizedPartSpec), expectedColumns)
    } else {
      // All partition columns are dynamic because the InsertIntoTable command does
      // not explicitly specify partitioning columns.
      castAndRenameChildOutput(insert, expectedColumns)
        .copy(partition = partColNames.map(_ -> None).toMap)
    }
  }

  private def castAndRenameChildOutput(
      insert: InsertIntoTable,
      expectedOutput: Seq[Attribute]): InsertIntoTable = {
    val newChildOutput = expectedOutput.zip(insert.child.output).map {
      case (expected, actual) =>
        if (expected.dataType.sameType(actual.dataType) &&
            expected.name == actual.name &&
            expected.metadata == actual.metadata) {
          actual
        } else {
          // Renaming is needed for handling the following cases like
          // 1) Column names/types do not match, e.g., INSERT INTO TABLE tab1 SELECT 1, 2
          // 2) Target tables have column metadata
          Alias(Cast(actual, expected.dataType), expected.name)(
            explicitMetadata = Option(expected.metadata))
        }
    }

    if (newChildOutput == insert.child.output) {
      insert
    } else {
      insert.copy(child = Project(newChildOutput, insert.child))
    }
  }

  def apply(plan: LogicalPlan): LogicalPlan = plan transform {
    case i @ InsertIntoTable(table, partition, child, _, _) if table.resolved && child.resolved =>
      table match {
        case relation: CatalogRelation =>
          val metadata = relation.catalogTable
          preprocess(i, metadata.identifier.quotedString, metadata.partitionColumnNames)
        case LogicalRelation(h: HadoopFsRelation, _, catalogTable) =>
          val tblName = catalogTable.map(_.identifier.quotedString).getOrElse("unknown")
          preprocess(i, tblName, h.partitionSchema.map(_.name))
        case LogicalRelation(_: InsertableRelation, _, catalogTable) =>
          val tblName = catalogTable.map(_.identifier.quotedString).getOrElse("unknown")
          preprocess(i, tblName, Nil)
        case other => i
      }
  }
}

/**
 * A rule to check whether the functions are supported only when Hive support is enabled
 */
object HiveOnlyCheck extends (LogicalPlan => Unit) {
  def apply(plan: LogicalPlan): Unit = {
    plan.foreach {
      case CreateTable(tableDesc, _, Some(_))
          if tableDesc.provider.get == DDLUtils.HIVE_PROVIDER =>
        throw new AnalysisException("Hive support is required to use CREATE Hive TABLE AS SELECT")

      case _ => // OK
    }
  }
}

/**
 * A rule to do various checks before inserting into or writing to a data source table.
 */
case class PreWriteCheck(conf: SQLConf, catalog: SessionCatalog)
  extends (LogicalPlan => Unit) {

  def failAnalysis(msg: String): Unit = { throw new AnalysisException(msg) }

  def apply(plan: LogicalPlan): Unit = {
    plan.foreach {
      case c @ CreateTable(tableDesc, mode, query) if c.resolved =>
        if (query.isDefined &&
          mode == SaveMode.Overwrite &&
          catalog.tableExists(tableDesc.identifier)) {
          // Need to remove SubQuery operator.
          EliminateSubqueryAliases(catalog.lookupRelation(tableDesc.identifier)) match {
            // Only do the check if the table is a data source table
            // (the relation is a BaseRelation).
            case LogicalRelation(dest: BaseRelation, _, _) =>
              // Get all input data source relations of the query.
              val srcRelations = query.get.collect {
                case LogicalRelation(src: BaseRelation, _, _) => src
              }
              if (srcRelations.contains(dest)) {
                failAnalysis(
                  s"Cannot overwrite table ${tableDesc.identifier} that is also being read from")
              }
            case _ => // OK
          }
        }

      case logical.InsertIntoTable(
          l @ LogicalRelation(t: InsertableRelation, _, _), partition, query, _, _) =>
        // Right now, we do not support insert into a data source table with partition specs.
        if (partition.nonEmpty) {
          failAnalysis(s"Insert into a partition is not allowed because $l is not partitioned.")
        } else {
          // Get all input data source relations of the query.
          val srcRelations = query.collect {
            case LogicalRelation(src: BaseRelation, _, _) => src
          }
          if (srcRelations.contains(t)) {
            failAnalysis(
              "Cannot insert overwrite into table that is also being read from.")
          } else {
            // OK
          }
        }

      case logical.InsertIntoTable(
        LogicalRelation(r: HadoopFsRelation, _, _), part, query, _, _) =>
        // We need to make sure the partition columns specified by users do match partition
        // columns of the relation.
        val existingPartitionColumns = r.partitionSchema.fieldNames.toSet
        val specifiedPartitionColumns = part.keySet
        if (existingPartitionColumns != specifiedPartitionColumns) {
          failAnalysis("Specified partition columns " +
            s"(${specifiedPartitionColumns.mkString(", ")}) " +
            "do not match the partition columns of the table. Please use " +
            s"(${existingPartitionColumns.mkString(", ")}) as the partition columns.")
        } else {
          // OK
        }

        PartitioningUtils.validatePartitionColumn(
          r.schema, part.keySet.toSeq, conf.caseSensitiveAnalysis)

        // Get all input data source relations of the query.
        val srcRelations = query.collect {
          case LogicalRelation(src: BaseRelation, _, _) => src
        }
        if (srcRelations.contains(r)) {
          failAnalysis(
            "Cannot insert overwrite into table that is also being read from.")
        } else {
          // OK
        }

      case logical.InsertIntoTable(l: LogicalRelation, _, _, _, _) =>
        // The relation in l is not an InsertableRelation.
        failAnalysis(s"$l does not allow insertion.")

      case _ => // OK
    }
  }
}
