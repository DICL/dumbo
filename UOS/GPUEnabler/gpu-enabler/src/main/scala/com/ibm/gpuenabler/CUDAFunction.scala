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

package com.ibm.gpuenabler

import java.io._
import java.net.URL

import jcuda.{CudaException, Pointer}
import jcuda.driver.JCudaDriver._
import jcuda.driver.{CUdeviceptr, CUfunction, CUstream}
import jcuda.runtime.{JCuda, cudaStream_t}
import org.apache.commons.io.IOUtils
import org.apache.spark._
import org.apache.spark.api.java.function.{Function => JFunction, Function2 => JFunction2}
import org.apache.spark.storage.{BlockId, RDDBlockId}

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.language.{existentials, implicitConversions}
import scala.reflect.ClassTag

/**
  * An abstract class to represent a ''User Defined function'' from a Native GPU program.
  */
abstract class ExternalFunction extends Serializable {
 def compute[U : ClassTag, T : ClassTag](inp: HybridIterator[T],
                                         columnSchemas: Seq[ColumnPartitionSchema],
                                         outputSize: Option[Int] = None,
                                         outputArraySizes: Seq[Int] = null,
                                         inputFreeVariables: Seq[Any] = null,
                                         blockId : Option[BlockId] = None) : Iterator[U]

  def inputColumnsOrder(): Seq[String]

  def outputColumnsOrder(): Seq[String]
}

/**
  *   * A class to represent a ''User Defined function'' from a Native GPU program.
  *   * Wrapper Java function for CUDAFunction scala function
  *
  * Specify the `funcName`, `_inputColumnsOrder`, `_outputColumnsOrder`,
  * and `resourceURL` when creating a new `CUDAFunction`,
  * then pass this object as an input argument to `mapExtFunc` or
  *  `reduceExtFunc` as follows,
  *
  * {{{
  *
  * JavaCUDAFunction mapFunction = new JavaCUDAFunction(
  *              "multiplyBy2",
  *              Arrays.asList("this"),
  *              Arrays.asList("this"),
  *              ptxURL);
  *
  *   JavaRDD<Integer> inputData = sc.parallelize(range).cache();
  *   ClassTag<Integer> tag = scala.reflect.ClassTag$.MODULE$.apply(Integer.TYPE);
  *   JavaCUDARDD<Integer> ci = new JavaCUDARDD(inputData.rdd(), tag);
  *   
  *   JavaCUDARDD<Integer> output = ci.mapExtFunc((new Function<Integer, Integer>() {
  *          public Integer call(Integer x) {
  *              return (2 * x);
  *          }
  *    }), mapFunction, tag)
  *
  * }}}
  *
  * @constructor The "compute" method is initialized so that when invoked it will
  *             load and launch the GPU kernel with the required set of parameters
  *             based on the input & output column order.
  * @param funcName Name of the Native code's function
  * @param _inputColumnsOrder List of input columns name mapping to corresponding
  *                           class members of the input RDD.
  * @param _outputColumnsOrder List of output columns name mapping to corresponding
  *                            class members of the result RDD.
  * @param resourceURL  Points to the resource URL where the GPU kernel is present
  * @param constArgs  Sequence of constant argument that need to passed in to a
  *                   GPU Kernel
  * @param stagesCount  Provide a function which is used to determine the number
  *                     of stages required to run this GPU kernel in spark based on the
  *                     number of partition items to process. Default function return "1".
  * @param dimensions Provide a function which is used to determine the GPU compute
  *                   dimensions for each stage. Default function will determined the
  *                   dimensions based on the number of partition items but for a single
  *                   stage.
  */
class JavaCUDAFunction(val funcName: String,
                       val _inputColumnsOrder: java.util.List[String] = null,
                       val _outputColumnsOrder: java.util.List[String] = null,
                       val resourceURL: URL,
                       val constArgs: Seq[AnyVal] = Seq(),
                       val stagesCount: Option[JFunction[Long, Integer]] = null,
                       val dimensions: Option[JFunction2[Long, Integer, Tuple2[Integer,Integer]]] = null) 
    extends Serializable {

  implicit def toScalaTuples(x: Tuple2[Integer,Integer]) : Tuple2[Int,Int] = (x._1, x._2)

  implicit def toScalaFunction(fun: JFunction[Long, Integer]):
    Option[Long => Int] = if (fun != null)
      Some(x => fun.call(x))
    else None

  implicit def toScalaFunction(fun: JFunction2[Long, Integer, Tuple2[Integer, Integer]]):
    Option[(Long, Int) => Tuple2[Int, Int]] =  if (fun != null)
      Some((x, y) => fun.call(x, y))
    else None

  val stagesCountFn: Option[Long => Int]  = stagesCount match {
    case Some(fun: JFunction[Long, Integer]) => fun
    case _ => None
  }

  val dimensionsFn: Option[(Long, Int) => Tuple2[Int, Int]] = dimensions match {
    case Some(fun: JFunction2[Long, Integer, Tuple2[Integer, Integer]] ) => fun
    case _ => None
  }

  val cf = new CUDAFunction(funcName, _inputColumnsOrder.asScala, _outputColumnsOrder.asScala,
    resourceURL, constArgs, stagesCountFn, dimensionsFn)
  
  /* 
   * 3 variants - call invocations
   */
  def this(funcName: String, _inputColumnsOrder: java.util.List[String],
          _outputColumnsOrder: java.util.List[String],
          resourceURL: URL) =
    this(funcName, _inputColumnsOrder, _outputColumnsOrder,
      resourceURL, Seq(), None, None)

  def this(funcName: String, _inputColumnsOrder: java.util.List[String],
           _outputColumnsOrder: java.util.List[String],
           resourceURL: URL, constArgs: Seq[AnyVal]) =
    this(funcName, _inputColumnsOrder, _outputColumnsOrder,
    resourceURL, constArgs, None, None)

  def this(funcName: String, _inputColumnsOrder: java.util.List[String],
           _outputColumnsOrder: java.util.List[String],
           resourceURL: URL, constArgs: Seq[AnyVal],
            stagesCount: Option[JFunction[Long, Integer]]) =
    this(funcName, _inputColumnsOrder, _outputColumnsOrder,
    resourceURL, Seq(), stagesCount, None)
}


/**
  * A class to represent a ''User Defined function'' from a Native GPU program.
  *
  * Specify the `funcName`, `_inputColumnsOrder`, `_outputColumnsOrder`,
  * and `resourceURL` when creating a new `CUDAFunction`,
  * then pass this object as an input argument to `mapExtFunc` or
  *  `reduceExtFunc` as follows,
  *
  * {{{
  *     val mapFunction = new CUDAFunction(
  *           "multiplyBy2",
  *           Array("this"),
  *           Array("this"),
  *           ptxURL)
  *
  *   val output = sc.parallelize(1 to n, 1)
  *       .mapExtFunc((x: Int) => 2 * x, mapFunction)
  *
  * }}}
  *
  * @constructor The "compute" method is initialized so that when invoked it will
  *             load and launch the GPU kernel with the required set of parameters
  *             based on the input & output column order.
  * @param funcName Name of the Native code's function
  * @param _inputColumnsOrder List of input columns name mapping to corresponding
  *                           class members of the input RDD.
  * @param _outputColumnsOrder List of output columns name mapping to corresponding
  *                            class members of the result RDD.
  * @param resource  Points to the resource URL where the GPU kernel is present
  * @param constArgs  Sequence of constant argument that need to passed in to a
  *                   GPU Kernel
  * @param stagesCount  Provide a function which is used to determine the number
  *                     of stages required to run this GPU kernel in spark based on the
  *                     number of partition items to process. Default function return "1".
  * @param dimensions Provide a function which is used to determine the GPU compute
  *                   dimensions for each stage. Default function will determined the
  *                   dimensions based on the number of partition items but for a single
  *                   stage.
  */
class CUDAFunction(
                    val funcName: String,
                    val _inputColumnsOrder: Seq[String] = null,
                    val _outputColumnsOrder: Seq[String] = null,
                    val resource: Any,
                    val constArgs: Seq[AnyVal] = Seq(),
                    val stagesCount: Option[Long => Int] = None,
                    val dimensions: Option[(Long, Int) => (Int, Int)] = None
                   )
  extends ExternalFunction {
  implicit def toScalaFunction(fun: JFunction[Long, Int]): Long => Int = x => fun.call(x)

  implicit def toScalaFunction(fun: JFunction2[Long, Int, Tuple2[Int,Int]]): (Long, Int) => 
    Tuple2[Int,Int] = (x, y) => fun.call(x, y)

  def inputColumnsOrder: Seq[String] = _inputColumnsOrder
  def outputColumnsOrder: Seq[String] = _outputColumnsOrder

  var _blockId: Option[BlockId] = Some(RDDBlockId(0, 0))


  //touch GPUSparkEnv for endpoint init
  GPUSparkEnv.get
  val ptxmodule = resource match {
    case resourceURL: URL =>
      (resourceURL.toString, {
        val inputStream = resourceURL.openStream()
        val moduleBinaryData = IOUtils.toByteArray(inputStream)
        inputStream.close()
        new String(moduleBinaryData.map(_.toChar))
      })
    case (name: String, ptx: String) => (name, ptx)
    case _ => throw new UnsupportedOperationException("this type is not supported for module")
  }

  // asynchronous Launch of kernel
  private[gpuenabler] def launchKernel(function: CUfunction, numElements: Int,
                           kernelParameters: Pointer,
                           dimensions: Option[(Long, Int) => (Int, Int)] = None,
                           stageNumber: Int = 1,
                           cuStream: CUstream) = {

    val (gpuGridSize, gpuBlockSize) = dimensions match {
      case Some(computeDim) => computeDim(numElements, stageNumber)
      case None => GPUSparkEnv.get.cudaManager.computeDimensions(numElements)
    }

    cuLaunchKernel(function,
      gpuGridSize, 1, 1,  // how many blocks
      gpuBlockSize, 1, 1, // threads per block (eg. 1024)
      0, cuStream, // Shared memory size and stream
      kernelParameters, null // Kernel- and extra parameters
    )
  }

  def createkernelParameterDesc2(a: Any, cuStream: CUstream):
        Tuple5[_, _, CUdeviceptr, Pointer, _] = {
    val (arr, hptr, devPtr, gptr, sz) = a match {
      case h if h.isInstanceOf[Int] => {
        val arr = Array.fill(1)(a.asInstanceOf[Int])
        val devPtr = GPUSparkEnv.get.cudaManager.allocateGPUMemory(INT_COLUMN.bytes)
        cuMemcpyHtoDAsync(devPtr, Pointer.to(arr), INT_COLUMN.bytes, cuStream)
        (arr, Pointer.to(arr), devPtr, Pointer.to(devPtr), INT_COLUMN.bytes)
      }
      case h if h.isInstanceOf[Byte] => {
        val arr = Array.fill(1)(a.asInstanceOf[Byte])
        val devPtr = GPUSparkEnv.get.cudaManager.allocateGPUMemory(BYTE_COLUMN.bytes)
        cuMemcpyHtoDAsync(devPtr, Pointer.to(arr), BYTE_COLUMN.bytes, cuStream)
        (arr, Pointer.to(arr), devPtr, Pointer.to(devPtr), BYTE_COLUMN.bytes)
      }
      case h if h.isInstanceOf[Short] => {
        val arr = Array.fill(1)(a.asInstanceOf[Short])
        val devPtr = GPUSparkEnv.get.cudaManager.allocateGPUMemory(SHORT_COLUMN.bytes)
        cuMemcpyHtoDAsync(devPtr, Pointer.to(arr), SHORT_COLUMN.bytes, cuStream)
        (arr, Pointer.to(arr), devPtr, Pointer.to(devPtr), SHORT_COLUMN.bytes)
      }
      case h if h.isInstanceOf[Long] => {
        val arr = Array.fill(1)(a.asInstanceOf[Long])
        val devPtr = GPUSparkEnv.get.cudaManager.allocateGPUMemory(LONG_COLUMN.bytes)
        cuMemcpyHtoDAsync(devPtr, Pointer.to(arr), LONG_COLUMN.bytes, cuStream)
        (arr, Pointer.to(arr), devPtr, Pointer.to(devPtr), LONG_COLUMN.bytes)
      }
      case h if h.isInstanceOf[Double] => {
        val arr = Array.fill(1)(a.asInstanceOf[Double])
        val devPtr = GPUSparkEnv.get.cudaManager.allocateGPUMemory(DOUBLE_COLUMN.bytes)
        cuMemcpyHtoDAsync(devPtr, Pointer.to(arr), DOUBLE_COLUMN.bytes, cuStream)
        (arr, Pointer.to(arr), devPtr, Pointer.to(devPtr), DOUBLE_COLUMN.bytes)
      }
      case h if h.isInstanceOf[Float] => {
        val arr = Array.fill(1)(a.asInstanceOf[Float])
        val devPtr = GPUSparkEnv.get.cudaManager.allocateGPUMemory(FLOAT_COLUMN.bytes)
        cuMemcpyHtoDAsync(devPtr, Pointer.to(arr), FLOAT_COLUMN.bytes, cuStream)
        (arr, Pointer.to(arr), devPtr, Pointer.to(devPtr), FLOAT_COLUMN.bytes)
      }
      case h if h.isInstanceOf[Array[Double]] => {
        val arr = h.asInstanceOf[Array[Double]]
        val sz = h.asInstanceOf[Array[Double]].length * DOUBLE_COLUMN.bytes
        val devPtr = GPUSparkEnv.get.cudaManager.allocateGPUMemory(sz)
        cuMemcpyHtoDAsync(devPtr, Pointer.to(arr), sz, cuStream)
        (arr, Pointer.to(arr), devPtr, Pointer.to(devPtr), sz)
      }
      case h if h.isInstanceOf[Array[Int]] => {
        val arr = h.asInstanceOf[Array[Int]]
        val sz = h.asInstanceOf[Array[Int]].length * INT_COLUMN.bytes
        val devPtr = GPUSparkEnv.get.cudaManager.allocateGPUMemory(sz)
        cuMemcpyHtoDAsync(devPtr, Pointer.to(arr), sz, cuStream)
        (arr, Pointer.to(arr), devPtr, Pointer.to(devPtr), sz)
      }
      case h if h.isInstanceOf[Array[Long]] => {
        val arr = h.asInstanceOf[Array[Long]]
        val sz = h.asInstanceOf[Array[Long]].length * LONG_COLUMN.bytes
        val devPtr = GPUSparkEnv.get.cudaManager.allocateGPUMemory(sz)
        cuMemcpyHtoDAsync(devPtr, Pointer.to(arr), sz, cuStream)
        (arr, Pointer.to(arr), devPtr, Pointer.to(devPtr), sz)
      }
      case h if h.isInstanceOf[Array[Float]] => {
        val arr = h.asInstanceOf[Array[Float]]
        val sz = h.asInstanceOf[Array[Float]].length * FLOAT_COLUMN.bytes
        val devPtr = GPUSparkEnv.get.cudaManager.allocateGPUMemory(sz)
        cuMemcpyHtoDAsync(devPtr, Pointer.to(arr), sz, cuStream)
        (arr, Pointer.to(arr), devPtr, Pointer.to(devPtr), sz)
      }
    }
    (arr, hptr, devPtr, gptr, sz)
  }

  /**
    *  This function is invoked from RDD `compute` function and it load & launches
    *  the GPU kernel and performs the computation on GPU and returns the results
    *  in an iterator.
    *
    * @param inputHyIter Provide the HybridIterator instance
    * @param columnSchemas Provide the input and output column schema
    * @param outputSize Specify the number of expected result. Default is equal to
    *                   the number of element in that partition/
    * @param outputArraySizes If the expected result is an array folded in a linear
    *                         form, specific a sequence of the array length for every
    *                         output columns
    * @param inputFreeVariables Specify a list of free variable that need to be
    *                           passed in to the GPU kernel function, if any
    * @param blockId  Specify the block ID associated with this operation
    * @tparam U Output Iterator's type
    * @tparam T Input Iterator's type
    * @return Returns an iterator of type U
    */
  def compute[U: ClassTag, T: ClassTag](inputHyIter: HybridIterator[T],
                                        columnSchemas: Seq[ColumnPartitionSchema],
                                        outputSize: Option[Int] = None,
                                        outputArraySizes: Seq[Int] = null,
                                        inputFreeVariables: Seq[Any] = null,
                                        blockId: Option[BlockId] = None): Iterator[U] = {

		//hyeonjin added
		val time_start = System.nanoTime()
		println(s"hyeonjin: CUDAFunction compute() starts [${time_start}]")

    val threadId = Thread.currentThread().getId // for debugging multi tasks
    println(s"thread $threadId compute cuda function start time ${System.nanoTime()}") // debug
    //    val start0 = System.nanoTime() // profiling
    val module = GPUSparkEnv.get.cudaManager.cachedLoadModule(Right(ptxmodule))
    val function = new CUfunction
    cuModuleGetFunction(function, module, funcName)


    val stream = new cudaStream_t
    JCuda.cudaStreamCreateWithFlags(stream, JCuda.cudaStreamNonBlocking)
    val cuStream = new CUstream(stream)

    _blockId = blockId

    val inputColumnSchema = columnSchemas(0)
    val outputColumnSchema = columnSchemas(1)


    // Ensure the GPU is loaded with the same data in memory
    inputHyIter.copyCpuToGpu

    var listDevPtr: List[CUdeviceptr] = null
    //    println(s"input copy h2d time: ${(System.nanoTime() - start0)/1000000}")//

    //    val start = System.nanoTime() // profiling
    // hardcoded first argument
    val (arr, hptr, devPtr: CUdeviceptr, gptr, sz) = createkernelParameterDesc2(inputHyIter.numElements, cuStream)

    listDevPtr = List(devPtr)
    // size + input Args based on inputColumnOrder + constArgs
    var kp: List[Pointer] = List(gptr) ++ inputHyIter.listKernParmDesc.map(_.gpuPtr)

    val outputHyIter = new HybridIterator[U](null, null, outputColumnSchema,
      outputColumnsOrder, blockId,
      outputSize.getOrElse(inputHyIter.numElements), outputArraySizes = outputArraySizes)

    kp = kp ++ outputHyIter.listKernParmDesc.map(_.gpuPtr)
    val outputInitEnd = System.nanoTime() // profiling
    //    println(s"output init time: ${(outputInitEnd - inputColEnd) / 1000000}") // profiling

    // add additional user input parameters
    if (inputFreeVariables != null) {
      val inputFreeVarPtrs = inputFreeVariables.map { inputFreeVariable =>
        createkernelParameterDesc2(inputFreeVariable, cuStream)
      }
      kp = kp ++ inputFreeVarPtrs.map(_._4) // gptr
      listDevPtr = listDevPtr ++ inputFreeVarPtrs.map(_._3) // CUdeviceptr
    }
    cuCtxSynchronize() // profiling
    val inputFreeVarMemcpyEnd = System.nanoTime() // profiling
    println(s"thread $threadId input free variable memcpy time: ${(inputFreeVarMemcpyEnd - outputInitEnd) / 1000000}") // profiling

    // add outputArraySizes to the list of arguments
    if (outputArraySizes != null) {
      val outputArraySizes_kpd = createkernelParameterDesc2(outputArraySizes.toArray, cuStream)
      kp = kp ++ Seq(outputArraySizes_kpd._4) // gpuPtr
      listDevPtr = listDevPtr ++ List(outputArraySizes_kpd._3) // CUdeviceptr
    }

    // add user provided constant variables
    if (constArgs != null) {
      //      val inputConstPtrs = constArgs.map { constVariable =>
      //        createkernelParameterDesc2(constVariable, cuStream)
      //      }
      val inputConstPtrs = constArgs.map {
        case v: Byte => Pointer.to(Array(v))
        case v: Char => Pointer.to(Array(v))
        case v: Short => Pointer.to(Array(v))
        case v: Int => Pointer.to(Array(v))
        case v: Long => Pointer.to(Array(v))
        case v: Float => Pointer.to(Array(v))
        case v: Double => Pointer.to(Array(v))
        case _ => throw new SparkException("Unsupported type passed to kernel as a constant argument")
      }
      //      kp = kp ++ inputConstPtrs.map(_._4) // gpuPtr
      //      listDevPtr = listDevPtr ++ inputConstPtrs.map(_._3) // CUdeviceptr
      kp = kp ++ inputConstPtrs // gpuPtr

    }
    cuCtxSynchronize() // profiling

    println(s"thread $threadId launch kerenl time: ${System.nanoTime()}") // profiling

    stagesCount match {
      // normal launch, no stages, suitable for map

      case None =>
        val kernelParameters = Pointer.to(kp: _*)
        // Start the GPU execution with the populated kernel parameters
        println("None")
        launchKernel(function, inputHyIter.numElements, kernelParameters, dimensions, 1, cuStream)
        cuCtxSynchronize() // profiling

      // launch kernel multiple times (multiple stages), suitable for reduce
      case Some(totalStagesFun) =>
        val totalStages = totalStagesFun(inputHyIter.numElements)
        if (totalStages <= 0) {
          throw new SparkException("Number of stages in a kernel launch must be positive")
        }

        // preserve the kernel parameter list so as to use it for every stage.
        val preserve_kp = kp
        println("some")
        (0 to totalStages - 1).foreach { stageNumber =>
          val stageParams =
            List(Pointer.to(Array[Int](stageNumber)), Pointer.to(Array[Int](totalStages)))

          // restore the preserved kernel parameters
          kp = preserve_kp

          kp = kp ++ Seq(Pointer.to(Array(stageNumber))) // debug
          //          val stageNumber_kpd = createkernelParameterDesc2(stageNumber, cuStream)
          //          kp = kp ++ Seq(stageNumber_kpd._4) // gpuPtr
          //          listDevPtr = listDevPtr ++ List(stageNumber_kpd._3) // CUdeviceptr

          kp = kp ++ Seq(Pointer.to(Array(totalStages))) // debug
          //          val totalStages_kpd = createkernelParameterDesc2(totalStages, cuStream)
          //          kp = kp ++ Seq(totalStages_kpd._4) // gpuPtr
          //          listDevPtr = listDevPtr ++ List(totalStages_kpd._3) // CUdeviceptr

          // val kernelParameters = Pointer.to(params: _*)
          val kernelParameters = Pointer.to(kp: _*)

          // Start the GPU execution with the populated kernel parameters
          launchKernel(function, inputHyIter.numElements, kernelParameters, dimensions, stageNumber, cuStream)
          cuCtxSynchronize() // profiling
          println(s"kernel end time ${System.nanoTime()} ")
        }
    }
    cuCtxSynchronize() // profiling
    println(s"thread $threadId kernel end time ${System.nanoTime()} ")
    //    println(s"kernel time: ${(kernelEnd - constNoutputArraySizeMemcpyEnd) / 1000000}") // profiling

    // Free up locally allocated GPU memory
    listDevPtr.foreach(devPtr => {
      cuMemFree(devPtr)
    })
    listDevPtr = List()

    outputHyIter.freeGPUMemory
    //    println("input hyIter start") // debug
    inputHyIter.freeGPUMemory

    cuCtxSynchronize() // profiling
    println(s"thread $threadId stream destory start time ${System.nanoTime()} ")
    JCuda.cudaStreamDestroy(stream)

    println(s"thread $threadId compute end time ${System.nanoTime()}") // debug

    val a = outputHyIter.asInstanceOf[Iterator[U]]
    //    val a = outputHyIter.getResultIterator // TODO change the code for new line to get iterator
    //    println(s"outputHyiter as Iterator by getResultIterator time ${System.nanoTime()}") // debug

		//hyeonjin added
		val time_end = System.nanoTime()
		println(s"hyeonjin: CUDAFunction compute() ends [${time_end}]")
		println(s"hyeonjin: gpu time=${(time_end - time_start)/1000000}ms.")


		
		a


  }
}

class CUDAFunction2(
                     override val funcName: String,
                     override val _inputColumnsOrder: Seq[String] = null,
                     override val _outputColumnsOrder: Seq[String] = null,
                     val subKernOutputType: Seq[String] = Seq(), // output type of sequential kernels, it goes to next kernels input
                     val subKernOutputSize: Seq[(Int, Int)] = Seq(), // (numElement, Dimension)
                     override val resource: Any,
                     override val constArgs: Seq[AnyVal] = Seq(),
                     override val stagesCount: Option[Long => Int] = None,
                     override val dimensions: Option[(Long, Int) => (Int, Int)] = None
                   ) extends CUDAFunction(funcName, _inputColumnsOrder, _outputColumnsOrder,
  resource, constArgs, stagesCount, dimensions) {

  override def compute[U: ClassTag, T: ClassTag](inputHyIter: HybridIterator[T],
                                                 columnSchemas: Seq[ColumnPartitionSchema],
                                                 outputSize: Option[Int] = None,
                                                 outputArraySizes: Seq[Int] = null,
                                                 inputFreeVariables: Seq[Any] = null,
                                                 blockId: Option[BlockId] = None): Iterator[U] = {

		//hyeonjin added
		val time_start = System.nanoTime()
		println(s"hyeonjin: CUDAFunction2 compute() starts [${time_start}]")

    val module = GPUSparkEnv.get.cudaManager.cachedLoadModule(Right(ptxmodule))
    val funcNameArr = funcName.split(", ")
    val functionArr = funcNameArr.map{ name =>
      val func = new CUfunction
      cuModuleGetFunction(func, module, name)
      func
    }

    _blockId = blockId

    val inputColumnSchema = columnSchemas(0)
    val outputColumnSchema = columnSchemas(1)


    val outputHyIter = new HybridIterator[U](null, null, outputColumnSchema,
      outputColumnsOrder, blockId,
      outputSize.getOrElse(0), outputArraySizes = outputArraySizes,
      isPipelined = true, subPartitionSizes = inputHyIter.getNumStreams)

    val totalNumKernel = funcNameArr.length
    var kernelCount = 0 // count

    val threadId = Thread.currentThread().getId
    println(s"thread $threadId funcDetail start")

    val funcDetail = (functionArr, subKernOutputType, subKernOutputSize).zipped.map {
      case (func, oType, (numElem, dim)) => (func, oType, numElem, dim)
    }
    var nextInputDevPtr: mutable.HashMap[String, CUdeviceptr] = new mutable.HashMap[String, CUdeviceptr]() // next input ptr for next kernel
    var nextInputNumElem: mutable.HashMap[String, Int] = new mutable.HashMap[String, Int]() // next input ptr for next kernel
    var nextInputDim: Int = 0
    var listCuStream: List[(Int, cudaStream_t)] = List()
    var globalStreamNumber: Int = 0

    var freeVarPtr: List[Pointer] = Nil

    println(s"thread $threadId funcDetail each start")
    funcDetail.foreach { case (func, outputType, outputSize, outputDim) =>
      def createOutputDevPtr(numElement: Int, stream: CUstream): CUdeviceptr = {
        val outputCpuPtr: Pointer = new Pointer()
        val outputDevPtr: CUdeviceptr = new CUdeviceptr()
        try {
          outputType match {
            case "Int" =>
              val size = numElement * INT_COLUMN.bytes
              JCuda.cudaHostAlloc(outputCpuPtr, size, JCuda.cudaHostAllocPortable)
              cuMemAlloc(outputDevPtr, size)
              cuMemcpyHtoDAsync(outputDevPtr, outputCpuPtr, size, stream)
//              println("cuMemsetD32 " + cuMemsetD32Async(outputDevPtr, 0, size / 4, stream))
            case "Short" =>
              val size = numElement * SHORT_COLUMN.bytes
              JCuda.cudaHostAlloc(outputCpuPtr, size, JCuda.cudaHostAllocPortable)
              cuMemAlloc(outputDevPtr, size)
              cuMemcpyHtoDAsync(outputDevPtr, outputCpuPtr, size, stream)
            case "Float" =>
              val size = numElement * FLOAT_COLUMN.bytes
              JCuda.cudaHostAlloc(outputCpuPtr, size, JCuda.cudaHostAllocPortable)
              cuMemAlloc(outputDevPtr, size)
              cuMemcpyHtoDAsync(outputDevPtr, outputCpuPtr, size, stream)
            case "Double" =>
              val size = numElement * DOUBLE_COLUMN.bytes
              JCuda.cudaHostAlloc(outputCpuPtr, size, JCuda.cudaHostAllocPortable)
              cuMemAlloc(outputDevPtr, size)
              cuMemcpyHtoDAsync(outputDevPtr, outputCpuPtr, size, stream)
          }
        } catch {
          case e: CudaException => println(e.getMessage)
        }
        outputDevPtr
      }


      var streamN = 0
      var totalElem = 0
      var seqListDevPtr: Seq[List[CUdeviceptr]] = Seq()

      val inputIterStreamNumber = inputHyIter.setElemeCount
        .filter{case (name, count) => name.startsWith(inputHyIter.blockId.get.toString)}
        .keySet.size

      if (inputIterStreamNumber != 0) {
        println(s"it selected ${inputIterStreamNumber}")
        globalStreamNumber = inputIterStreamNumber
      }

      while (
        if (kernelCount == 0) {
          if(inputIterStreamNumber == 0) {
            println("number 1 while")
            inputHyIter.hasNext
          } else {
            println("number 2 while")
            streamN < inputIterStreamNumber
          }
        } else {
          println("number 3 while")
          streamN < globalStreamNumber
        }
      ) {

        println(s"thread $threadId H2D start time: ${System.nanoTime()}") // profiling
        val stream = if (kernelCount == 0) {
          val temp = new cudaStream_t
          JCuda.cudaStreamCreateWithFlags(temp, JCuda.cudaStreamNonBlocking)
          listCuStream = listCuStream ++ List((streamN, temp))
          temp
        } else listCuStream(streamN)._2

        val cuStream = new CUstream(stream)


        // Ensure the GPU is loaded with the same data in memory
        if (kernelCount == 0) inputHyIter.copyCpuToGpuChunk(cuStream, streamN)

        val inputKernParmDesc = if (kernelCount == 0) {
//          if (inputHyIter.columnsOrder.length == 1)
//            inputHyIter.lazyInputListKernParmDesc(cuStream, streamN).next()
//          else
            inputHyIter.lazyInputListKernParmDescTest(cuStream, streamN).next()
        } else Seq()
//        println(s"thread $threadId input kernel parm desc finished")
        // TODO I think it has to change more efficiently and generally
        val numElement = if (kernelCount == 0) {
          //          inputKernParmDesc(0).sz / inputColumnSchema.orderedColumns(inputColumnsOrder)(0).columnType.bytes
          println(s"${inputHyIter.blockId.get}_stream_${streamN}")
          inputHyIter.setElemeCount(inputHyIter.blockId.get + "_stream_" + streamN)
        } else {
          nextInputNumElem(s"kernel_${kernelCount - 1}_" + streamN)
        }
        println(s"set element count ${inputHyIter.setElemeCount.keySet.size}")

//        println(s"thread $threadId numElement is $numElement")

        // hardcoded first argument
        val (arr, hptr, devPtr: CUdeviceptr, gptr: Pointer, sz) = createkernelParameterDesc2(numElement, cuStream)

//        println(s"list dev ptr start")
        var listDevPtr: List[CUdeviceptr] = Nil
        listDevPtr = List(devPtr)
        // size + input Args based on inputColumnOrder + constArgs
//        println(s"kp kernel parm desc finished")
        var kp: List[Pointer] = Nil
        try {
          if (kernelCount == 0) {
                        //            beforeInputDevPtr = beforeInputDevPtr ++ inputKernParmDesc.map(_.devPtr)
            listDevPtr = listDevPtr ++ inputKernParmDesc.map(_.devPtr)
            kp = kp ++ List(gptr) ++ inputKernParmDesc.map(_.gpuPtr)
          } else {
            listDevPtr = listDevPtr ++ inputKernParmDesc.map(_.devPtr)
            //            beforeInputDevPtr = beforeInputDevPtr ++ List(nextInputDevPtr(streamN))
            kp = kp ++ List(gptr) ++ List(Pointer.to(nextInputDevPtr(s"kernel_${kernelCount - 1}_" + streamN)))
          }
        } catch {
          case e: Exception => println(e.getMessage)
        }

//        println(s"output dev ptr start")
        //          kp = kp ++ outputHyIter.lazyOutputListKernParmDesc(inputKernParmDesc, cuStream, streamN).map(_.gpuPtr)
        val outputNumElement = if (outputSize == 0) numElement else outputSize
        val outputDimension = if(outputArraySizes == null) outputDim else outputArraySizes.head
        val outputDevPtr = createOutputDevPtr(outputNumElement * outputDimension, cuStream)

        kp = kp ++ List(Pointer.to(outputDevPtr))

        if(freeVarPtr == Nil) {
          // add additional user input parameters
          if (inputFreeVariables != null) {
            val inputFreeVarPtrs = inputFreeVariables.map { inputFreeVariable =>
              createkernelParameterDesc2(inputFreeVariable, cuStream)
            }
//            kp = kp ++ inputFreeVarPtrs.map(_._4) // gptr
            freeVarPtr = freeVarPtr ++ inputFreeVarPtrs.map(_._4) // gptr
            listDevPtr = listDevPtr ++ inputFreeVarPtrs.map(_._3) // CUdeviceptr
          }

          // add outputArraySizes to the list of arguments

          if (outputArraySizes != null) {
            val outputArraySizes_kpd = createkernelParameterDesc2(outputArraySizes.toArray, cuStream)
//            kp = kp ++ Seq(outputArraySizes_kpd._4) // gpuPtr
            freeVarPtr = freeVarPtr ++ Seq(outputArraySizes_kpd._4) // gpuPtr
            listDevPtr = listDevPtr ++ List(outputArraySizes_kpd._3) // CUdeviceptr
          }

          // add user provided constant variables
          if (constArgs != null) {
            //      val inputConstPtrs = constArgs.map { constVariable =>
            //        createkernelParameterDesc2(constVariable, cuStream)
            //      }
            val inputConstPtrs = constArgs.map {
              case v: Byte => Pointer.to(Array(v.asInstanceOf[Byte]))
              case v: Char => Pointer.to(Array(v.asInstanceOf[Char]))
              case v: Short => Pointer.to(Array(v.asInstanceOf[Short]))
              case v: Int => Pointer.to(Array(v.asInstanceOf[Int]))
              case v: Long => Pointer.to(Array(v.asInstanceOf[Long]))
              case v: Float => Pointer.to(Array(v.asInstanceOf[Float]))
              case v: Double => Pointer.to(Array(v.asInstanceOf[Double]))
              case _ => throw new SparkException("Unsupported type passed to kernel as a constant argument")
            }
            //      kp = kp ++ inputConstPtrs.map(_._4) // gpuPtr
            //      listDevPtr = listDevPtr ++ inputConstPtrs.map(_._3) // CUdeviceptr
//            kp = kp ++ inputConstPtrs // gpuPtr
            freeVarPtr = freeVarPtr ++ inputConstPtrs

          }
        }

        kp = kp ++ freeVarPtr


        kp = kp ++ Seq(Pointer.to(Array(streamN)))

        //        cuCtxSynchronize() // profiling
        //        val constNoutputArraySizeMemcpyEnd = System.nanoTime() // profiling
        println(s"thread $threadId H2D end and kernel start time: ${System.nanoTime()}") // profiling


        stagesCount match {
          // normal launch, no stages, suitable for map

          case None =>
            val kernelParameters = Pointer.to(kp: _*)
            // Start the GPU execution with the populated kernel parameters
            launchKernel(func, numElement, kernelParameters, dimensions, 1, cuStream)

          // launch kernel multiple times (multiple stages), suitable for reduce
          case Some(totalStagesFun) =>
            val totalStages = totalStagesFun(numElement)
            if (totalStages <= 0) {
              throw new SparkException("Number of stages in a kernel launch must be positive")
            }

            // preserve the kernel parameter list so as to use it for every stage.
            val preserve_kp = kp
            (0 to totalStages - 1).foreach { stageNumber =>
              val stageParams =
                List(Pointer.to(Array[Int](stageNumber)), Pointer.to(Array[Int](totalStages)))

              // restore the preserved kernel parameters
              kp = preserve_kp

              kp = kp ++ Seq(Pointer.to(Array(stageNumber))) // debug


              kp = kp ++ Seq(Pointer.to(Array(totalStages))) // debug


              val kernelParameters = Pointer.to(kp: _*)

              // Start the GPU execution with the populated kernel parameters
              launchKernel(func, numElement, kernelParameters, dimensions, stageNumber, cuStream)
            }
        }

        println(s"thread $threadId kernel end time: ${System.nanoTime()}") // profiling

        if (listDevPtr == Nil) println(s"listDevPtr is Nil")

        seqListDevPtr = seqListDevPtr ++ Seq(listDevPtr) // TODO remove after debug

        if (seqListDevPtr.isEmpty) println(s"seqListDevPtr is Empty")
        /*
      // Free up locally allocated GPU memory
      listDevPtr.foreach(devPtr => {
        cuMemFree(devPtr)
      })
      listDevPtr = List()
      */

        // TODO remove after debug


        //        outputHyIter.freeChunkGPUMemory(cuStream, streamN)
        //        inputHyIter.freeChunkGPUMemory(cuStream, streamN)



        nextInputDevPtr.put(s"kernel_${kernelCount}_" + streamN, outputDevPtr)
        nextInputNumElem.put(s"kernel_${kernelCount}_" + streamN, outputNumElement)
        nextInputDim = outputDimension

        streamN += 1
        totalElem += outputNumElement
        //        JCuda.cudaStreamDestroy(stream)
      }

      //          JCuda.cudaDeviceSynchronize()


      //          GPUSparkEnv.get.cudaManager.freeGPUMemory(outputDevPtr)

      if(kernelCount == 0) globalStreamNumber = streamN

      kernelCount += 1

/*
      println(s"free gpu memory start")
      try {
        seqListDevPtr.foreach(list =>
          list.foreach(cuMemFree)) // CUDA_ERROR_ILLEGAL_ADDRESS
      } catch {
        case e: CudaException => println(e.getMessage); e.printStackTrace()
      }

      try {
        listCuStream.foreach { case (streamN, stream) =>
          //            GPUSparkEnv.get.cudaManager.freeGPUMemory(beforeInputDevPtr(streamN))
          //          inputHyIter.freeChunkGPUMemory(streamN)
          //            JCuda.cudaStreamDestroy(stream)
        }
      } catch {
        case e: CudaException => println(e.getMessage); e.printStackTrace()
      }
      */

      println(s"thread $threadId kernel ${kernelCount - 1} end time: ${System.nanoTime()}") // profiling

      outputHyIter.totalChunkCount = streamN
      outputHyIter.totalElementCount = totalElem
      outputHyIter.subPartitionSize = totalElem / streamN

      //      }
      /*else {

        println(s"\n\nkernel ${kernelCount} start\n\n") // profiling
        val stream = new cudaStream_t
        JCuda.cudaStreamCreateWithFlags(stream, JCuda.cudaStreamNonBlocking)
        val cuStream = new CUstream(stream)
        listCuStream = listCuStream ++ List((streamN, stream))

        val numElement = nextInputNumElem

        // hardcoded first argument
        val (arr, hptr, devPtr: CUdeviceptr, gptr, sz) = createkernelParameterDesc2(numElement, cuStream)

        var listDevPtr: List[CUdeviceptr] = null
        listDevPtr = List(devPtr)
        // size + input Args based on inputColumnOrder + constArgs
        var kp: List[Pointer] = List(gptr) ++ List(Pointer.to(nextInputDevPtr))

        //    val inputColEnd = System.nanoTime() // profiling
        //    println(s"input columnize time: ${(inputColEnd - start) / 1000000}") // profiling

        //          kp = kp ++ outputHyIter.lazyOutputListKernParmDesc(inputKernParmDesc, cuStream, streamN).map(_.gpuPtr)
        kp = kp ++ Seq(Pointer.to(outputDevPtr))

        // add additional user input parameters
        if (inputFreeVariables != null) {
          val inputFreeVarPtrs = inputFreeVariables.map { inputFreeVariable =>
            createkernelParameterDesc2(inputFreeVariable, cuStream)
          }
          kp = kp ++ inputFreeVarPtrs.map(_._4) // gptr
          listDevPtr = listDevPtr ++ inputFreeVarPtrs.map(_._3) // CUdeviceptr
        }

        //        cuCtxSynchronize() // profiling
        //        val inputFreeVarMemcpyEnd = System.nanoTime() // profiling
        //        println(s"input free variable memcpy time: ${(inputFreeVarMemcpyEnd - outputInitEnd) / 1000000}") // profiling

        // add outputArraySizes to the list of arguments

        if (outputArraySizes != null) {
          val outputArraySizes_kpd = createkernelParameterDesc2(outputArraySizes.toArray, cuStream)
          kp = kp ++ Seq(outputArraySizes_kpd._4) // gpuPtr
          listDevPtr = listDevPtr ++ List(outputArraySizes_kpd._3) // CUdeviceptr
        }

        // add user provided constant variables
        if (constArgs != null) {
          //      val inputConstPtrs = constArgs.map { constVariable =>
          //        createkernelParameterDesc2(constVariable, cuStream)
          //      }
          val inputConstPtrs = constArgs.map {
            case v: Byte => Pointer.to(Array(v))
            case v: Char => Pointer.to(Array(v))
            case v: Short => Pointer.to(Array(v))
            case v: Int => Pointer.to(Array(v))
            case v: Long => Pointer.to(Array(v))
            case v: Float => Pointer.to(Array(v))
            case v: Double => Pointer.to(Array(v))
            case _ => throw new SparkException("Unsupported type passed to kernel as a constant argument")
          }
          //      kp = kp ++ inputConstPtrs.map(_._4) // gpuPtr
          //      listDevPtr = listDevPtr ++ inputConstPtrs.map(_._3) // CUdeviceptr
          kp = kp ++ inputConstPtrs // gpuPtr

        }

        //        cuCtxSynchronize() // profiling
        //        val constNoutputArraySizeMemcpyEnd = System.nanoTime() // profiling
        println(s"H2D end and kernel start time: ${System.nanoTime()}") // profiling
        try {
          stagesCount match {
            // normal launch, no stages, suitable for map

            case None =>
              val kernelParameters = Pointer.to(kp: _*)
              // Start the GPU execution with the populated kernel parameters
              launchKernel(func, numElement, kernelParameters, dimensions, 1, cuStream)

            // launch kernel multiple times (multiple stages), suitable for reduce
            case Some(totalStagesFun) =>
              val totalStages = totalStagesFun(numElement)
              if (totalStages <= 0) {
                throw new SparkException("Number of stages in a kernel launch must be positive")
              }

              // preserve the kernel parameter list so as to use it for every stage.
              val preserve_kp = kp
              (0 to totalStages - 1).foreach { stageNumber =>
                val stageParams =
                  List(Pointer.to(Array[Int](stageNumber)), Pointer.to(Array[Int](totalStages)))

                // restore the preserved kernel parameters
                kp = preserve_kp

                kp = kp ++ Seq(Pointer.to(Array(stageNumber))) // debug
                //          val stageNumber_kpd = createkernelParameterDesc2(stageNumber, cuStream)
                //          kp = kp ++ Seq(stageNumber_kpd._4) // gpuPtr
                //          listDevPtr = listDevPtr ++ List(stageNumber_kpd._3) // CUdeviceptr

                kp = kp ++ Seq(Pointer.to(Array(totalStages))) // debug
                //          val totalStages_kpd = createkernelParameterDesc2(totalStages, cuStream)
                //          kp = kp ++ Seq(totalStages_kpd._4) // gpuPtr
                //          listDevPtr = listDevPtr ++ List(totalStages_kpd._3) // CUdeviceptr

                // val kernelParameters = Pointer.to(params: _*)
                val kernelParameters = Pointer.to(kp: _*)

                // Start the GPU execution with the populated kernel parameters
                launchKernel(func, numElement, kernelParameters, dimensions, stageNumber, cuStream)
              }
          }
        } catch {
          case e: CudaException => println(e.getMessage)
        }
        //    cuCtxSynchronize() // profiling
        //    val kernelEnd = System.nanoTime() // profiling
        //    println(s"kernel time: ${(kernelEnd - constNoutputArraySizeMemcpyEnd) / 1000000}") // profiling

        //        cuCtxSynchronize() // profiling
        //        val constNoutputArraySizeMemcpyEnd = System.nanoTime() // profiling
        println(s"kernel end time: ${System.nanoTime()}") // profiling

        seqListDevPtr = seqListDevPtr ++ Seq(listDevPtr) // TODO remove after debug
        GPUSparkEnv.get.cudaManager.freeGPUMemory(nextInputDevPtr)
        nextInputDevPtr = outputDevPtr
        nextInputNumElem = outputSize * outputDim

        println(s"kernel ${kernelCount} end time: ${System.nanoTime()}") // profiling


        kernelCount += 1
      }
    */


			//hyeonjin added
			val time_end = System.nanoTime()
			println(s"hyeonjin: CUDAFunction2 compute() ends [${time_end}]")
			println(s"hyeonjin: gpu time=${(time_end - time_start)/1000000}ms.")
    }

    val funcDetailLast = funcDetail.last



    listCuStream.foreach{ case (streamN, stream) =>
      outputHyIter.setElemeCount.update(
        blockId.get + "_stream_" + streamN,
        nextInputNumElem(s"kernel_${kernelCount-1}_" + streamN)
      )
      outputHyIter.copyGpuToCpuChunkTest2(new CUstream(stream), streamN,
        nextInputNumElem(s"kernel_${kernelCount-1}_" + streamN) * nextInputDim,
        nextInputDevPtr(s"kernel_${kernelCount-1}_" + streamN))
      /*
      funcDetailLast._2 match {
        case "Int"    =>
          val tmp = new Array[Int](nextInputNumElem(streamN))
          cuMemcpyDtoHAsync(Pointer.to(tmp), nextInputDevPtr(streamN),
            nextInputNumElem(streamN) * INT_COLUMN.bytes, new CUstream(stream))
          tmp
        case "Long"   =>
          val tmp = new Array[Long](nextInputNumElem(streamN))
          cuMemcpyDtoHAsync(Pointer.to(tmp), nextInputDevPtr(streamN),
            nextInputNumElem(streamN) * LONG_COLUMN.bytes, new CUstream(stream))
          tmp
        case "Float"  =>
          val tmp = new Array[Float](nextInputNumElem(streamN))
          cuMemcpyDtoHAsync(Pointer.to(tmp), nextInputDevPtr(streamN),
            nextInputNumElem(streamN) * funcDetailLast._4 * FLOAT_COLUMN.bytes, new CUstream(stream))
          tmp
        case "Double" =>
          val tmp = new Array[Double](nextInputNumElem(streamN))
          cuMemcpyDtoHAsync(Pointer.to(tmp), nextInputDevPtr(streamN),
            nextInputNumElem(streamN) * funcDetailLast._4 * DOUBLE_COLUMN.bytes, new CUstream(stream))
          tmp
      }
      */
      if(!inputHyIter.gpuCache) JCuda.cudaStreamDestroy(stream)
    }
    println(s"thread $threadId D2H end time: ${System.nanoTime()}") // profiling

//    println(s"outputHyIter totalElementsCount = ${outputHyIter.totalElementCount}")
//    println(s"outputHyIter totalChunkCount = ${outputHyIter.totalChunkCount}")
//    println(s"outputHyIter subPartitionSize = ${outputHyIter.subPartitionSize}")
    outputHyIter.asInstanceOf[Iterator[U]]

    /*
    val tempArr = funcDetailLast._2 match {
      case "Int"    =>
        val tmp = new Array[Int](funcDetailLast._3 * funcDetailLast._4)
        cuMemcpyDtoH(Pointer.to(tmp), nextInputDevPtr, funcDetailLast._3 * funcDetailLast._4 * INT_COLUMN.bytes)
        tmp
      case "Long"   =>
        val tmp = new Array[Long](funcDetailLast._3 * funcDetailLast._4)
        cuMemcpyDtoH(Pointer.to(tmp), nextInputDevPtr, funcDetailLast._3 * funcDetailLast._4 * LONG_COLUMN.bytes)
        tmp
      case "Float"  =>
        val tmp = new Array[Float](funcDetailLast._3 * funcDetailLast._4)
        cuMemcpyDtoH(Pointer.to(tmp), nextInputDevPtr, funcDetailLast._3 * funcDetailLast._4 * FLOAT_COLUMN.bytes)
        tmp
      case "Double" =>
        val tmp = new Array[Double](funcDetailLast._3 * funcDetailLast._4)
        cuMemcpyDtoH(Pointer.to(tmp), nextInputDevPtr, funcDetailLast._3 * funcDetailLast._4 * DOUBLE_COLUMN.bytes)
        tmp
    }

    tempArr.toIterator.asInstanceOf[Iterator[U]]
    */


  }

}
