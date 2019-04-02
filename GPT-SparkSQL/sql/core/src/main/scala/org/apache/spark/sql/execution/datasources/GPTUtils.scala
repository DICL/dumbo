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

import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession

object GPTUtils {

  // GPT-{split}-{Table name}-{Bit vectors}-{Partition ID}
  def getPartitionID(filePath: String): Int = {

    val tokens = filePath.split("-")

    // e,g., 00000 , 00001 , ..., 00109
    val partitionID = tokens(tokens.length-1)

    val tmp = partitionID.toString
    var idx = 0
    var realIdx = 0
    var foundNZ = false
    tmp.foreach{ c =>
      if (c != '0' && !foundNZ) {
        foundNZ = true
        realIdx = idx
      } else {
        idx += 1
      }
    }
    if (realIdx == 0) {
      tmp.toInt
    } else {
      tmp.substring(realIdx).toInt
    }
  }
}
