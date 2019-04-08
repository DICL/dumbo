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

package org.apache.spark.ml.tuning

import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.classification.LogisticRegressionSuite.generateLogisticInput
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, Evaluator, RegressionEvaluator}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.HasInputCol
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.util.{DefaultReadWriteTest, MLTestingUtils}
import org.apache.spark.mllib.util.{LinearDataGenerator, MLlibTestSparkContext}
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.StructType

class TrainValidationSplitSuite
  extends SparkFunSuite with MLlibTestSparkContext with DefaultReadWriteTest {

  import testImplicits._

  test("train validation with logistic regression") {
    val dataset = sc.parallelize(generateLogisticInput(1.0, 1.0, 100, 42), 2).toDF()

    val lr = new LogisticRegression
    val lrParamMaps = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.001, 1000.0))
      .addGrid(lr.maxIter, Array(0, 10))
      .build()
    val eval = new BinaryClassificationEvaluator
    val cv = new TrainValidationSplit()
      .setEstimator(lr)
      .setEstimatorParamMaps(lrParamMaps)
      .setEvaluator(eval)
      .setTrainRatio(0.5)
      .setSeed(42L)
    val cvModel = cv.fit(dataset)
    val parent = cvModel.bestModel.parent.asInstanceOf[LogisticRegression]
    assert(cv.getTrainRatio === 0.5)
    assert(parent.getRegParam === 0.001)
    assert(parent.getMaxIter === 10)
    assert(cvModel.validationMetrics.length === lrParamMaps.length)
  }

  test("train validation with linear regression") {
    val dataset = sc.parallelize(
      LinearDataGenerator.generateLinearInput(
        6.3, Array(4.7, 7.2), Array(0.9, -1.3), Array(0.7, 1.2), 100, 42, 0.1), 2)
      .map(_.asML).toDF()

    val trainer = new LinearRegression().setSolver("l-bfgs")
    val lrParamMaps = new ParamGridBuilder()
      .addGrid(trainer.regParam, Array(1000.0, 0.001))
      .addGrid(trainer.maxIter, Array(0, 10))
      .build()
    val eval = new RegressionEvaluator()
    val cv = new TrainValidationSplit()
      .setEstimator(trainer)
      .setEstimatorParamMaps(lrParamMaps)
      .setEvaluator(eval)
      .setTrainRatio(0.5)
      .setSeed(42L)
    val cvModel = cv.fit(dataset)

    // copied model must have the same paren.
    MLTestingUtils.checkCopy(cvModel)

    val parent = cvModel.bestModel.parent.asInstanceOf[LinearRegression]
    assert(parent.getRegParam === 0.001)
    assert(parent.getMaxIter === 10)
    assert(cvModel.validationMetrics.length === lrParamMaps.length)

      eval.setMetricName("r2")
    val cvModel2 = cv.fit(dataset)
    val parent2 = cvModel2.bestModel.parent.asInstanceOf[LinearRegression]
    assert(parent2.getRegParam === 0.001)
    assert(parent2.getMaxIter === 10)
    assert(cvModel2.validationMetrics.length === lrParamMaps.length)
  }

  test("transformSchema should check estimatorParamMaps") {
    import TrainValidationSplitSuite._

    val est = new MyEstimator("est")
    val eval = new MyEvaluator
    val paramMaps = new ParamGridBuilder()
      .addGrid(est.inputCol, Array("input1", "input2"))
      .build()

    val cv = new TrainValidationSplit()
      .setEstimator(est)
      .setEstimatorParamMaps(paramMaps)
      .setEvaluator(eval)
      .setTrainRatio(0.5)
    cv.transformSchema(new StructType()) // This should pass.

    val invalidParamMaps = paramMaps :+ ParamMap(est.inputCol -> "")
    cv.setEstimatorParamMaps(invalidParamMaps)
    intercept[IllegalArgumentException] {
      cv.transformSchema(new StructType())
    }
  }

  test("read/write: TrainValidationSplit") {
    val lr = new LogisticRegression().setMaxIter(3)
    val evaluator = new BinaryClassificationEvaluator()
    val paramMaps = new ParamGridBuilder()
        .addGrid(lr.regParam, Array(0.1, 0.2))
        .build()
    val tvs = new TrainValidationSplit()
      .setEstimator(lr)
      .setEvaluator(evaluator)
      .setTrainRatio(0.5)
      .setEstimatorParamMaps(paramMaps)
      .setSeed(42L)

    val tvs2 = testDefaultReadWrite(tvs, testParams = false)

    assert(tvs.getTrainRatio === tvs2.getTrainRatio)
    assert(tvs.getSeed === tvs2.getSeed)
  }

  test("read/write: TrainValidationSplitModel") {
    val lr = new LogisticRegression()
      .setThreshold(0.6)
    val lrModel = new LogisticRegressionModel(lr.uid, Vectors.dense(1.0, 2.0), 1.2)
      .setThreshold(0.6)
    val evaluator = new BinaryClassificationEvaluator()
    val paramMaps = new ParamGridBuilder()
        .addGrid(lr.regParam, Array(0.1, 0.2))
        .build()
    val tvs = new TrainValidationSplitModel("cvUid", lrModel, Array(0.3, 0.6))
    tvs.set(tvs.estimator, lr)
      .set(tvs.evaluator, evaluator)
      .set(tvs.trainRatio, 0.5)
      .set(tvs.estimatorParamMaps, paramMaps)
      .set(tvs.seed, 42L)

    val tvs2 = testDefaultReadWrite(tvs, testParams = false)

    assert(tvs.getTrainRatio === tvs2.getTrainRatio)
    assert(tvs.validationMetrics === tvs2.validationMetrics)
    assert(tvs.getSeed === tvs2.getSeed)
  }
}

object TrainValidationSplitSuite {

  abstract class MyModel extends Model[MyModel]

  class MyEstimator(override val uid: String) extends Estimator[MyModel] with HasInputCol {

    override def fit(dataset: Dataset[_]): MyModel = {
      throw new UnsupportedOperationException
    }

    override def transformSchema(schema: StructType): StructType = {
      require($(inputCol).nonEmpty)
      schema
    }

    override def copy(extra: ParamMap): MyEstimator = defaultCopy(extra)
  }

  class MyEvaluator extends Evaluator {

    override def evaluate(dataset: Dataset[_]): Double = {
      throw new UnsupportedOperationException
    }

    override def isLargerBetter: Boolean = true

    override val uid: String = "eval"

    override def copy(extra: ParamMap): MyEvaluator = defaultCopy(extra)
  }
}
