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

import java.nio.{ByteBuffer, ByteOrder}

import jcuda.driver.JCudaDriver._
import jcuda.driver.{CUdeviceptr, CUresult, CUstream}
import jcuda.runtime.{JCuda, cudaStream_t}
import jcuda.{CudaException, Pointer}
import org.apache.spark.storage.BlockId
import org.apache.spark.storage.RDDBlockId

import scala.collection.mutable.{ArrayBuffer, HashMap}
import scala.language.existentials
import scala.reflect.ClassTag
import scala.reflect.runtime._
import scala.reflect.runtime.universe.TermSymbol

// scalastyle:off no.finalize
private[gpuenabler] case class KernelParameterDesc(
                                cpuArr: Array[_ >: Byte with Short with Int
                                  with Float with Long with Double <: AnyVal],
                                cpuPtr: Pointer,
                                devPtr: CUdeviceptr,
                                gpuPtr: Pointer,
                                sz: Int,
                                symbol: TermSymbol)

private[gpuenabler] class HybridIterator[T: ClassTag](inputArr: Array[T],
                                                      iter: Iterator[T],
        colSchema: ColumnPartitionSchema,
        __columnsOrder: Seq[String],
        _blockId: Option[BlockId],
        numentries: Int = 0,
        outputArraySizes: Seq[Int] = null,
        isCol: Boolean = true,
        isPipelined: Boolean = false,
        subPartitionSizes: Int = 64 * 1024 * 1024 // default
                                                     ) extends Iterator[T] {

  private var _arr: Array[T] = inputArr

  val threadId = Thread.currentThread().getId // for debugging multi tasks
  /*
   * use the Iterator instead of the Array
   * to remove array overhead by qyu
   */
  private lazy val _iter: Iterator[T] = {
    if (iter == null) {
      if (!isPipelined) {
        println("number 1")
        copyGpuToCpu
        getResultIterator
      } else {
        println("number 2")
        getChunkResultIterator
        //        getChunkResultIteratorTest
      }
    } else {
      println("number 3")
      iter
    }
  }

  /*
   * To supporting row format, I create row pointer which is CPU pointer
   *
   */

  private lazy val _row_pointer = if(isCol) null else inputArr match {
    case h if h.isInstanceOf[Array[Int]] => Pointer.to(inputArr.asInstanceOf[Array[Int]])
    case h if h.isInstanceOf[Array[Short]] => Pointer.to(inputArr.asInstanceOf[Array[Short]])
    case h if h.isInstanceOf[Array[Long]] => Pointer.to(inputArr.asInstanceOf[Array[Long]])
    case h if h.isInstanceOf[Array[Byte]] => Pointer.to(inputArr.asInstanceOf[Array[Byte]])
    case h if h.isInstanceOf[Array[Float]] => Pointer.to(inputArr.asInstanceOf[Array[Float]])
    case h if h.isInstanceOf[Array[Double]] => Pointer.to(inputArr.asInstanceOf[Array[Double]])
  }

  def arr: Array[T] = if (_arr == null) {
    // Validate the CPU pointers before deserializing
    copyGpuToCpu
    _arr = getResultList // bottleneck
    _arr
  } else {
    _arr
  }

  private var _columnsOrder = __columnsOrder

  def columnsOrder: Seq[String] = if (_columnsOrder == null) {
    _columnsOrder = colSchema.getAllColumns()
    _columnsOrder
  } else {
    _columnsOrder

  }

  private val _outputArraySizes = if (outputArraySizes != null) {
    outputArraySizes 
  } else {
    val tempbuf = new ArrayBuffer[Int]()
    // if outputArraySizes is not provided by user program; create one
    // based on the number of columns and initialize it to '1' to denote
    // the object has only one element in it.
    columnsOrder.foreach(_ => tempbuf += 1)
    tempbuf
  }

  var _numElements = if (inputArr != null) inputArr.length else 0
  var idx: Int = -1

  val blockId: Option[BlockId] = _blockId match {
    case Some(x) => _blockId
    case None => {
      val r = scala.util.Random
      Some(RDDBlockId(r.nextInt(99999), r.nextInt(99999)))
    }
  }

  def rddId: Int = blockId.getOrElse(RDDBlockId(0, 0)).asRDDId.get.rddId

  def cachedGPUPointers: HashMap[String, KernelParameterDesc] =
    GPUSparkEnv.get.gpuMemoryManager.getCachedGPUPointers

  def numElements: Int = _numElements

  lazy val stream = new cudaStream_t
  if(inputArr != null) JCuda.cudaStreamCreateWithFlags(stream, JCuda.cudaStreamNonBlocking)
  lazy val cuStream = new CUstream(stream)



//  I removed this part because it makes spark's transformation/action methods performance fall down when
//  user use gpu + cpu transformation/action methods.
//  I think it could be used array.
/*
  def hasNext: Boolean = {
    idx < arr.length - 1
  }

  def next: T = if (hasNext){
    idx += 1
    arr(idx)
  } else Iterator.empty.next()
*/


  override def hasNext: Boolean = _iter.hasNext

  override def next(): T = _iter.next()


  def gpuCache: Boolean = GPUSparkEnv.get.gpuMemoryManager.cachedGPURDDs.contains(rddId)

  // Function to free the allocated GPU memory if the RDD is not cached.
  def freeGPUMemory: Unit = {
    println(s"GPUSparkEnv.get.gpuMemoryManager.cachedGPURDDs is empty = " +
      s"${GPUSparkEnv.get.gpuMemoryManager.cachedGPURDDs.isEmpty}")
    if (!gpuCache) {
      // Make sure the CPU ptrs are populated before GPU memory is freed up.
      copyGpuToCpu
      if (_listKernParmDesc == null) return
      println("devPtr freed from gpu") // debug
      _listKernParmDesc = _listKernParmDesc.map(kpd => {
        if (kpd.devPtr != null) {
          GPUSparkEnv.get.cudaManager.freeGPUMemory(kpd.devPtr)
        }
        KernelParameterDesc(kpd.cpuArr, kpd.cpuPtr, null, null, kpd.sz, kpd.symbol)
      })
      cachedGPUPointers.retain(
        (name, kernelParameterDesc) => !name.startsWith(blockId.get.toString))
    }
  }

  // TODO: Discuss the need for finalize; how to handle streams;
  override def finalize(): Unit = {
    JCuda.cudaStreamDestroy(stream)
    super.finalize
  }

  // This function is used to copy the CPU memory to GPU for
  // an existing Hybrid Iterator
  def copyCpuToGpu: Unit = {
    if (_listKernParmDesc == null) return
    _listKernParmDesc = _listKernParmDesc.map(kpd => {
      if (kpd.devPtr == null) {
        val devPtr = GPUSparkEnv.get.cudaManager.allocateGPUMemory(kpd.sz)
        cuMemcpyHtoDAsync(devPtr, kpd.cpuPtr, kpd.sz, cuStream)
        cuCtxSynchronize()
        val gPtr = Pointer.to(devPtr)
        KernelParameterDesc(kpd.cpuArr, kpd.cpuPtr, devPtr, gPtr, kpd.sz, kpd.symbol)
      } else {
        kpd
      }
    })
  }

  // This function is used to copy the GPU memory to CPU for
  // an existing Hybrid Iterator
  def copyGpuToCpu: Unit = {
    // Ensure main memory is allocated to hold the GPU data
    if (_listKernParmDesc == null) return
    _listKernParmDesc = (_listKernParmDesc, colSchema.orderedColumns(columnsOrder)).
        zipped.map((kpd, col) => {
      if (kpd.cpuArr == null && kpd.cpuPtr == null && kpd.devPtr != null) {
        val (cpuArr, cpuPtr: Pointer) = col.columnType match {
          case c if c == INT_COLUMN => {
            val y = new Array[Int](kpd.sz/ INT_COLUMN.bytes)
            (y, Pointer.to(y))
          }
          case c if c == SHORT_COLUMN => {
            val y = new Array[Short](kpd.sz/ SHORT_COLUMN.bytes)
            (y, Pointer.to(y))
          }
          case c if c == BYTE_COLUMN => {
            val y = new Array[Byte](kpd.sz/ BYTE_COLUMN.bytes)
            (y, Pointer.to(y))
          }
          case c if c == LONG_COLUMN => {
            val y = new Array[Long](kpd.sz/ LONG_COLUMN.bytes)
            (y, Pointer.to(y))
          }
          case c if c == FLOAT_COLUMN => {
            val y = new Array[Float](kpd.sz/ FLOAT_COLUMN.bytes)
            (y, Pointer.to(y))
          }
          case c if c == DOUBLE_COLUMN => {
            val y = new Array[Double](kpd.sz / DOUBLE_COLUMN.bytes)
            (y, Pointer.to(y))
          }
          case c if c == LONG_COLUMN => {
            val y = new Array[Long](kpd.sz / LONG_COLUMN.bytes)
            (y, Pointer.to(y))
          }
          case c if c == INT_ARRAY_COLUMN => {
            val y = new Array[Int](kpd.sz / INT_COLUMN.bytes)
            (y, Pointer.to(y))
          }
          case c if c == LONG_ARRAY_COLUMN => {
            val y = new Array[Long](kpd.sz / LONG_COLUMN.bytes)
            (y, Pointer.to(y))
          }
          case c if c == FLOAT_ARRAY_COLUMN => {
            val y = new Array[Float](kpd.sz / FLOAT_COLUMN.bytes)
            (y, Pointer.to(y))
          }
          case c if c == DOUBLE_ARRAY_COLUMN => {
            val y = new Array[Double](kpd.sz / DOUBLE_COLUMN.bytes)
            (y, Pointer.to(y))
          }
          case c if c == LONG_ARRAY_COLUMN => {
            val y = new Array[Long](kpd.sz / LONG_COLUMN.bytes)
            (y, Pointer.to(y))
          }
        }
        cuMemcpyDtoHAsync(cpuPtr, kpd.devPtr, kpd.sz, cuStream) 
        KernelParameterDesc(cpuArr, cpuPtr, kpd.devPtr, kpd.gpuPtr, kpd.sz, kpd.symbol)
      } else {
        kpd
      }
    })
    cuCtxSynchronize()
  }

  // Extract the getter method from the given object using reflection
  private def getter[C](obj: Any, symbol: TermSymbol): C = {
    currentMirror.reflect(obj).reflectField(symbol).get.asInstanceOf[C]
  }

  // valVarMembers will be available only for non-primitive types
  private val valVarMembers = if (colSchema.isPrimitive) {
    null
  } else {
    val runtimeCls = implicitly[ClassTag[T]].runtimeClass
    val clsSymbol = currentMirror.classSymbol(runtimeCls)
    clsSymbol.typeSignature.members.view
      .filter(p => !p.isMethod && p.isTerm).map(_.asTerm)
      .filter(p => p.isVar || p.isVal)
  }

  // Allocate Memory from Off-heap Pinned Memory and returns
  // the pointer & buffer address pointing to it
  private def allocPinnedHeap(size: Long) = {
    val ptr: Pointer = new Pointer()
    try {
      val result: Int = JCuda.cudaHostAlloc(ptr, size, JCuda.cudaHostAllocPortable)
//      val result: Int = JCuda.cudaHostAlloc(ptr, size, JCuda.cudaHostAllocWriteCombined) // best performance
      if (result != CUresult.CUDA_SUCCESS) {
        throw new CudaException(JCuda.cudaGetErrorString(result))
      }
    }
    catch {
      case ex: Exception => {
        throw new OutOfMemoryError("Could not alloc pinned memory: " + ex.getMessage)
      }
    }
    (ptr, ptr.getByteBuffer(0, size).order(ByteOrder.LITTLE_ENDIAN))
  }

  def listKernParmDesc: Seq[KernelParameterDesc] = _listKernParmDesc

  private var _listKernParmDesc = if (inputArr != null && inputArr.length > 0) {
    // initFromInputIterator
    println(s"thread $threadId listKernParmDesc start at ${System.nanoTime()}")
    val kernParamDesc = colSchema.orderedColumns(columnsOrder).map { col =>
      cachedGPUPointers.getOrElseUpdate(blockId.get + col.prettyAccessor, {
        val cname = col.prettyAccessor.split("\\.").reverse.head
        val symbol = if (colSchema.isPrimitive) {
          null
        } else {
          valVarMembers.find(_.name.toString.startsWith(cname)).get
        }

        val (hPtr: Pointer, colDataSize: Int) = {
          val mirror = ColumnPartitionSchema.mirror
          val priv_getter = col.terms.foldLeft(identity[Any] _)((r, term) =>
            ((obj: Any) => mirror.reflect(obj).reflectField(term).get) compose r)

          var bufferOffset = 0

          col.columnType match {
            case c if c == INT_COLUMN => {
              val size = col.memoryUsage(inputArr.length).toInt
              if (isCol) {
                println("isCol is true")
                val (ptr, buffer) = allocPinnedHeap(size)
                priv_getter(inputArr.head)
                inputArr.foreach(x => buffer.putInt(priv_getter(x).asInstanceOf[Int]))
                (ptr, size)
              } else {
                println("isRow is true")
                (_row_pointer, size)
              }
            }
            case c if c == LONG_COLUMN => {
              val size = col.memoryUsage(inputArr.length).toInt
              if (isCol) {
                val (ptr, buffer) = allocPinnedHeap(size)
                inputArr.foreach(x => buffer.putLong(priv_getter(x).asInstanceOf[Long]))
                (ptr, size)
              } else {
                (_row_pointer, size)
              }
            }
            case c if c == SHORT_COLUMN => {
              val size = col.memoryUsage(inputArr.length).toInt
              if (isCol) {
                val (ptr, buffer) = allocPinnedHeap(size)
                inputArr.foreach(x => buffer.putShort(priv_getter(x).asInstanceOf[Short]))
                (ptr, size)
              } else {
                (_row_pointer, size)
              }
            }
            case c if c == BYTE_COLUMN => {
              val size = col.memoryUsage(inputArr.length).toInt
              if (isCol) {
                val (ptr, buffer) = allocPinnedHeap(size)
                inputArr.foreach(x => buffer.put(priv_getter(x).asInstanceOf[Byte]))
                (ptr, size)
              } else {
                (_row_pointer, size)
              }
            }
            case c if c == FLOAT_COLUMN => {
              val size = col.memoryUsage(inputArr.length).toInt
              if (isCol) {
                val (ptr, buffer) = allocPinnedHeap(size)
                inputArr.foreach(x => buffer.putFloat(priv_getter(x).asInstanceOf[Float]))
                (ptr, size)
              } else {
                (_row_pointer, size)
              }
            }
            case c if c == DOUBLE_COLUMN => {
              val size = col.memoryUsage(inputArr.length).toInt
              if (isCol) {
                val (ptr, buffer) = allocPinnedHeap(size)
                inputArr.foreach(x => buffer.putDouble(priv_getter(x).asInstanceOf[Double]))
                (ptr, size)
              } else {
                (_row_pointer, size)
              }
            }
            case c if c == LONG_COLUMN => {
              val size = col.memoryUsage(inputArr.length).toInt
              if (isCol) {
                val (ptr, buffer) = allocPinnedHeap(size)
                inputArr.foreach(x => buffer.putLong(priv_getter(x).asInstanceOf[Long]))
                (ptr, size)
              } else {
                (_row_pointer, size)
              }
            }
            case c if c == INT_ARRAY_COLUMN => {
              // retrieve the first element to determine the array size.
              val arrLength = priv_getter(inputArr.head).asInstanceOf[Array[Int]].length
              val size = col.memoryUsage(inputArr.length * arrLength).toInt
              val (ptr, buffer) = allocPinnedHeap(size)
              inputArr.foreach(x => {
                buffer.position(bufferOffset)
                buffer.asIntBuffer().put(priv_getter(x).asInstanceOf[Array[Int]], 0, arrLength)
                // bufferOffset += col.memoryUsage(arrLength).toInt
                bufferOffset += arrLength * INT_COLUMN.bytes
              })
              (ptr, size)
            }
            case c if c == LONG_ARRAY_COLUMN => {
              // retrieve the first element to determine the array size.
              val arrLength = priv_getter(inputArr.head).asInstanceOf[Array[Long]].length
              val size = col.memoryUsage(inputArr.length * arrLength).toInt
              val (ptr, buffer) = allocPinnedHeap(size)
              inputArr.foreach(x => {
                buffer.position(bufferOffset)
                buffer.asLongBuffer().put(priv_getter(x).asInstanceOf[Array[Long]], 0, arrLength)
                // bufferOffset += col.memoryUsage(arrLength).toInt
                bufferOffset += arrLength * LONG_COLUMN.bytes
              })
              (ptr, size)
            }
            case c if c == FLOAT_ARRAY_COLUMN => {
              // retrieve the first element to determine the array size.
              val arrLength = priv_getter(inputArr.head).asInstanceOf[Array[Float]].length
              val size = col.memoryUsage(inputArr.length * arrLength).toInt
              val (ptr, buffer) = allocPinnedHeap(size)
              inputArr.foreach(x => {
                buffer.position(bufferOffset)
                buffer.asFloatBuffer().put(priv_getter(x).asInstanceOf[Array[Float]], 0, arrLength)
                bufferOffset += arrLength * FLOAT_COLUMN.bytes
                // bufferOffset += col.memoryUsage(arrLength).toInt
              })
              (ptr, size)
            }
            case c if c == DOUBLE_ARRAY_COLUMN => {
              // retrieve the first element to determine the array size.
              val arrLength = priv_getter(inputArr.head).asInstanceOf[Array[Double]].length
              val size = col.memoryUsage(inputArr.length * arrLength).toInt
              val (ptr, buffer) = allocPinnedHeap(size)
              inputArr.foreach(x => {
                buffer.position(bufferOffset)
                buffer.asDoubleBuffer().put(priv_getter(x).asInstanceOf[Array[Double]], 0, arrLength)
                bufferOffset += arrLength * DOUBLE_COLUMN.bytes
                // bufferOffset += col.memoryUsage(arrLength).toInt
              })
              (ptr, size)
            }
            case c if c == LONG_ARRAY_COLUMN => {
              // retrieve the first element to determine the array size.
              val arrLength = priv_getter(inputArr.head).asInstanceOf[Array[Long]].length
              val size = col.memoryUsage(inputArr.length * arrLength).toInt
              val (ptr, buffer) = allocPinnedHeap(size)
              inputArr.foreach(x => {
                buffer.position(bufferOffset)
                buffer.asLongBuffer().put(priv_getter(x).asInstanceOf[Array[Long]], 0, arrLength)
                bufferOffset += arrLength * LONG_COLUMN.bytes
                // bufferOffset += col.memoryUsage(arrLength).toInt
              })
              (ptr, size)
            }
          }
        }
        println(s"thread $threadId h2d start at ${System.nanoTime()}")
        val devPtr = GPUSparkEnv.get.cudaManager.allocateGPUMemory(colDataSize)
        cuMemcpyHtoDAsync(devPtr, hPtr, colDataSize, cuStream)
        val gPtr = Pointer.to(devPtr)

        // mark the cpuPtr null as we use pinned memory and got the Pointer directly
        new KernelParameterDesc(null, hPtr, devPtr, gPtr, colDataSize, symbol)
      })
    }
    cuCtxSynchronize()
    kernParamDesc


  } else if (numentries != 0) {
    // initEmptyArrays - mostly used by output argument list
    // set the number of entries to numentries as its initialized to '0'
    _numElements = numentries
    val colOrderSizes = colSchema.orderedColumns(columnsOrder) zip _outputArraySizes

    val kernParamDesc = colOrderSizes.map { col =>
      cachedGPUPointers.getOrElseUpdate(blockId.get + col._1.prettyAccessor, {
        val cname = col._1.prettyAccessor.split("\\.").reverse.head
        val symbol = if (colSchema.isPrimitive) {
          null
        } else {
          valVarMembers.find(_.name.toString.startsWith(cname)).get
        }

        val colDataSize: Int = col._1.columnType match {
          case c if c == INT_COLUMN => {
            numentries * INT_COLUMN.bytes
          }
          case c if c == LONG_COLUMN => {
            numentries * LONG_COLUMN.bytes
          }
          case c if c == SHORT_COLUMN => {
            numentries * SHORT_COLUMN.bytes
          }
          case c if c == BYTE_COLUMN => {
            numentries * BYTE_COLUMN.bytes
          }
          case c if c == FLOAT_COLUMN => {
            numentries * FLOAT_COLUMN.bytes
          }
          case c if c == DOUBLE_COLUMN => {
            numentries * DOUBLE_COLUMN.bytes
          }
          case c if c == LONG_COLUMN => {
            numentries * LONG_COLUMN.bytes
          }
          case c if c == INT_ARRAY_COLUMN => {
            col._2 * numentries * INT_COLUMN.bytes
          }
          case c if c == LONG_ARRAY_COLUMN => {
            col._2 * numentries * LONG_COLUMN.bytes
          }
          case c if c == FLOAT_ARRAY_COLUMN => {
            col._2 * numentries * FLOAT_COLUMN.bytes
          }
          case c if c == DOUBLE_ARRAY_COLUMN => {
            col._2 * numentries * DOUBLE_COLUMN.bytes
          }
          case c if c == LONG_ARRAY_COLUMN => {
            col._2 * numentries * LONG_COLUMN.bytes
          }
          // TODO : Handle error condition
        }
        val devPtr = GPUSparkEnv.get.cudaManager.allocateGPUMemory(colDataSize)
        cuMemsetD32Async(devPtr, 0, colDataSize / 4, cuStream)
        val gPtr = Pointer.to(devPtr)

        // Allocate only GPU memory; main memory will be allocated during deserialization
        new KernelParameterDesc(null, null, devPtr, gPtr, colDataSize, symbol)
      })
    }
    cuCtxSynchronize()
    kernParamDesc
  } else {
    null
  }

//  private val subPartitionSize = 128 * 1024 * 512 // 128 MB
  var subPartitionSize = subPartitionSizes // 128 MB

  val setOfkernParmDesc: HashMap[String, Seq[KernelParameterDesc]] = GPUSparkEnv.get.gpuMemoryManager.getCachedCPUPointers
  var totalChunkCount = 0
  var totalElementCount = 0
  var elementCount = 0
  val setElemeCount: HashMap[String, Int] = GPUSparkEnv.get.gpuMemoryManager.getSizeOfChunk
  def getNumStreams: Int = subPartitionSize

  /*
   * free chunk gpu memory for sub partition data
   * check stream number and remove from hash table named setOfkernParmDesc
   *
   * @param streamNumber stream number for the CUDA stream
   *
   */
  def freeChunkGPUMemory(streamNumber: Int): Unit = {
    if (!gpuCache) {
      // Make sure the CPU ptrs are populated before GPU memory is freed up.
//      copyGpuToCpuChunk(stream, streamNumber)
      if (!setOfkernParmDesc.contains(blockId.get + "_stream_"+streamNumber)) return
      setOfkernParmDesc(blockId.get + "_stream_"+streamNumber) = setOfkernParmDesc(blockId.get + "_stream_"+streamNumber).map(kpd => {
        if (kpd.devPtr != null) {
          GPUSparkEnv.get.cudaManager.freeGPUMemory(kpd.devPtr)
        }
        KernelParameterDesc(kpd.cpuArr, kpd.cpuPtr, null, null, kpd.sz, kpd.symbol)
      })
      cachedGPUPointers.retain(
        (name, kernelParameterDesc) => !name.startsWith(blockId.get.toString))
    }
  }

  /*
   * copy chunk cpu memory to gpu memory for sub partition data if this partition already in hash table
   * after checking stream number and stream, do H2D
   *
   * @param stream CUstream of sub partition
   * @param streamNumber stream number for the CUDA stream
   *
   */
  def copyCpuToGpuChunk(stream:CUstream, streamNumber: Int): Unit = {
    if (!setOfkernParmDesc.contains(blockId.get + "_stream_"+streamNumber)) {
//      println("hash map dosen't contains stream number ")
      return
    }
    setOfkernParmDesc(blockId.get + "_stream_"+streamNumber) = setOfkernParmDesc(blockId.get + "_stream_"+streamNumber).map(kpd => {
      if (kpd.devPtr == null) {
        val devPtr = GPUSparkEnv.get.cudaManager.allocateGPUMemory(kpd.sz)
        cuMemcpyHtoDAsync(devPtr, kpd.cpuPtr, kpd.sz, stream)
        val gPtr = Pointer.to(devPtr)
        KernelParameterDesc(kpd.cpuArr, kpd.cpuPtr, devPtr, gPtr, kpd.sz, kpd.symbol)
      } else {
        kpd
      }
    })

    (colSchema.orderedColumns(columnsOrder) zip setOfkernParmDesc(blockId.get + "_stream_"+streamNumber)).map { case (col, kpd) =>
      cachedGPUPointers.getOrElseUpdate(blockId.get + "_stream_"+streamNumber + "_" + col.prettyAccessor, kpd)
    }

  }

  /*
   * copy chunk gpu memory to cpu memory for sub partition data after CUDA stream finished
   * after checking stream number and stream, do D2H
   *
   * @param stream CUstream of sub partition
   * @param streamNumber stream number for the CUDA stream
   *
   */
  def copyGpuToCpuChunk(stream:CUstream, streamNumber: Int): Unit = {
    // Ensure main memory is allocated to hold the GPU data
    if (!setOfkernParmDesc.contains(blockId.get + "_stream_"+streamNumber)) {
//      println("hash map dosen't contains stream number ");
      return
    }

    println("" + setOfkernParmDesc.keySet) // debug

    setOfkernParmDesc(blockId.get + "_stream_"+streamNumber) = (setOfkernParmDesc(blockId.get + "_stream_"+streamNumber), colSchema.orderedColumns(columnsOrder)).
      zipped.map((kpd, col) => {
      println(s"stream number = ${streamNumber}") // debug
      if (kpd.cpuArr == null && kpd.cpuPtr == null && kpd.devPtr != null) {
        println("null if")
        val (cpuArr, cpuPtr: Pointer) = col.columnType match {
          case c if c == INT_COLUMN => {
//            val (cpuPtr, buf) = allocPinnedHeap(kpd.sz)
//            (buf, cpuPtr)
            val y = new Array[Int](kpd.sz/ INT_COLUMN.bytes)
            (y, Pointer.to(y))
          }
          case c if c == SHORT_COLUMN => {
            val y = new Array[Short](kpd.sz/ SHORT_COLUMN.bytes)
            (y, Pointer.to(y))
          }
          case c if c == BYTE_COLUMN => {
            val y = new Array[Byte](kpd.sz/ BYTE_COLUMN.bytes)
            (y, Pointer.to(y))
          }
          case c if c == LONG_COLUMN => {
            val y = new Array[Long](kpd.sz/ LONG_COLUMN.bytes)
            (y, Pointer.to(y))
          }
          case c if c == FLOAT_COLUMN => {
            val y = new Array[Float](kpd.sz/ FLOAT_COLUMN.bytes)
            (y, Pointer.to(y))
          }
          case c if c == DOUBLE_COLUMN => {
            val y = new Array[Double](kpd.sz / DOUBLE_COLUMN.bytes)
            (y, Pointer.to(y))
          }
          case c if c == LONG_COLUMN => {
            val y = new Array[Long](kpd.sz / LONG_COLUMN.bytes)
            (y, Pointer.to(y))
          }
          case c if c == INT_ARRAY_COLUMN => {
            val y = new Array[Int](kpd.sz / INT_COLUMN.bytes)
            (y, Pointer.to(y))
          }
          case c if c == LONG_ARRAY_COLUMN => {
            val y = new Array[Long](kpd.sz / LONG_COLUMN.bytes)
            (y, Pointer.to(y))
          }
          case c if c == FLOAT_ARRAY_COLUMN => {
            val y = new Array[Float](kpd.sz / FLOAT_COLUMN.bytes)
            (y, Pointer.to(y))
          }
          case c if c == DOUBLE_ARRAY_COLUMN => {
            val y = new Array[Double](kpd.sz / DOUBLE_COLUMN.bytes)
            (y, Pointer.to(y))
          }
          case c if c == LONG_ARRAY_COLUMN => {
            val y = new Array[Long](kpd.sz / LONG_COLUMN.bytes)
            (y, Pointer.to(y))
          }
        }
        cuMemcpyDtoHAsync(cpuPtr, kpd.devPtr, kpd.sz, stream)
//        KernelParameterDesc(cpuArr, cpuPtr, kpd.devPtr, kpd.gpuPtr, kpd.sz, kpd.symbol)
        KernelParameterDesc(cpuArr, cpuPtr, kpd.devPtr, kpd.gpuPtr, kpd.sz, kpd.symbol)
      } else {
        println("else")
        kpd
      }
    })
//    cuCtxSynchronize()
  }


  def copyGpuToCpuChunkTest(stream:CUstream, streamNumber: Int): Unit = {
    // Ensure main memory is allocated to hold the GPU data
    if (!setOfkernParmDesc.contains(blockId.get + "_stream_"+streamNumber)) {
//      println("hash map dosen't contains stream number ")
      return
    }

    println("" + setOfkernParmDesc.keySet) // debug

    setOfkernParmDesc(blockId.get + "_stream_"+streamNumber) = (setOfkernParmDesc(blockId.get + "_stream_"+streamNumber), colSchema.orderedColumns(columnsOrder)).
      zipped.map((kpd, col) => {
//      println(s"stream number = ${streamNumber}") // debug
      if (kpd.cpuArr == null && kpd.cpuPtr == null && kpd.devPtr != null) {
//        println("null if")
        val (cpuArr, cpuPtr: Pointer) = col.columnType match {
          case c if c == INT_COLUMN => {
            val (cpuPtr, buf) = allocPinnedHeap(kpd.sz)
            (buf.asIntBuffer(), cpuPtr)
          }
          case c if c == SHORT_COLUMN => {
            val y = new Array[Short](kpd.sz/ SHORT_COLUMN.bytes)
            (y, Pointer.to(y))
          }
          case c if c == BYTE_COLUMN => {
            val y = new Array[Byte](kpd.sz/ BYTE_COLUMN.bytes)
            (y, Pointer.to(y))
          }
          case c if c == LONG_COLUMN => {
            val y = new Array[Long](kpd.sz/ LONG_COLUMN.bytes)
            (y, Pointer.to(y))
          }
          case c if c == FLOAT_COLUMN => {
            val y = new Array[Float](kpd.sz/ FLOAT_COLUMN.bytes)
            (y, Pointer.to(y))
          }
          case c if c == DOUBLE_COLUMN => {
            val y = new Array[Double](kpd.sz / DOUBLE_COLUMN.bytes)
            (y, Pointer.to(y))
          }
          case c if c == LONG_COLUMN => {
            val y = new Array[Long](kpd.sz / LONG_COLUMN.bytes)
            (y, Pointer.to(y))
          }
          case c if c == INT_ARRAY_COLUMN => {
            val y = new Array[Int](kpd.sz / INT_COLUMN.bytes)
            (y, Pointer.to(y))
          }
          case c if c == LONG_ARRAY_COLUMN => {
            val y = new Array[Long](kpd.sz / LONG_COLUMN.bytes)
            (y, Pointer.to(y))
          }
          case c if c == FLOAT_ARRAY_COLUMN => {
            val y = new Array[Float](kpd.sz / FLOAT_COLUMN.bytes)
            (y, Pointer.to(y))
          }
          case c if c == DOUBLE_ARRAY_COLUMN => {
            val y = new Array[Double](kpd.sz / DOUBLE_COLUMN.bytes)
            (y, Pointer.to(y))
          }
          case c if c == LONG_ARRAY_COLUMN => {
            val y = new Array[Long](kpd.sz / LONG_COLUMN.bytes)
            (y, Pointer.to(y))
          }
        }
        cuMemcpyDtoHAsync(cpuPtr, kpd.devPtr, kpd.sz, stream)
        //        KernelParameterDesc(cpuArr, cpuPtr, kpd.devPtr, kpd.gpuPtr, kpd.sz, kpd.symbol)
        KernelParameterDesc(null, cpuPtr, kpd.devPtr, kpd.gpuPtr, kpd.sz, kpd.symbol)
      } else {
        println("else")
        kpd
      }
    })
    //    cuCtxSynchronize()
  }

  /*
   * copy chunk gpu memory to cpu memory for sub partition data after CUDA stream finished
   * after checking stream number and stream, do D2H
   *
   * @param stream CUstream of sub partition
   * @param streamNumber stream number for the CUDA stream
   *
   */
  def copyGpuToCpuChunkTest2(stream:CUstream, streamNumber: Int, size: Int, devPtr: CUdeviceptr): Unit = {
    println(s"thread id = $threadId alloc host array start at ${System.nanoTime()}")
    setOfkernParmDesc.put(blockId.get + "_stream_"+streamNumber, colSchema.orderedColumns(columnsOrder).map { col =>
      val (cpuArr, cpuPtr: Pointer, byteSize: Int) = col.columnType match {
        case c if c == INT_COLUMN => {
          val y = new Array[Int](size)
          (y, Pointer.to(y), size * INT_COLUMN.bytes)
        }
        case c if c == SHORT_COLUMN => {
          val y = new Array[Short](size)
          (y, Pointer.to(y), size * SHORT_COLUMN.bytes)
        }
        case c if c == BYTE_COLUMN => {
          val y = new Array[Byte](size)
          (y, Pointer.to(y), size * BYTE_COLUMN.bytes)
        }
        case c if c == LONG_COLUMN => {
          val y = new Array[Long](size)
          (y, Pointer.to(y), size * LONG_COLUMN.bytes)
        }
        case c if c == FLOAT_COLUMN => {
          val y = new Array[Float](size)
          (y, Pointer.to(y), size * FLOAT_COLUMN.bytes)
        }
        case c if c == DOUBLE_COLUMN => {
          val y = new Array[Double](size)
          (y, Pointer.to(y), size * DOUBLE_COLUMN.bytes)
        }
        case c if c == INT_ARRAY_COLUMN => {
          val y = new Array[Int](size)
          (y, Pointer.to(y), size * INT_COLUMN.bytes)
        }
        case c if c == LONG_ARRAY_COLUMN => {
          val y = new Array[Long](size)
          (y, Pointer.to(y), size * LONG_COLUMN.bytes)
        }
        case c if c == FLOAT_ARRAY_COLUMN => {
          val y = new Array[Float](size)
          (y, Pointer.to(y), size * FLOAT_COLUMN.bytes)
        }
        case c if c == DOUBLE_ARRAY_COLUMN => {
          val y = new Array[Double](size)
          (y, Pointer.to(y), size * DOUBLE_COLUMN.bytes)
        }
      }

      println(s"thread id = $threadId D2H array start at ${System.nanoTime()}")
      cuMemcpyDtoHAsync(cpuPtr, devPtr, byteSize, stream)
//      println(cpuArr.deep)

      KernelParameterDesc(cpuArr, cpuPtr, devPtr, Pointer.to(devPtr), size, null)
    })


  }

  /* I want to create stream for pipe-lining cpu-gpu task
   * so I create lazy list of kernel paramemter descriptions
   * if it is called by CUDAFunction for sub-partition to launch kernel,
   * it will be calculated only needed sub-partition at that time
   * @param listKPD
   */
  def lazyOutputListKernParmDesc(listKPD: Seq[KernelParameterDesc], cuStream: CUstream, streamNumber: Int): Seq[KernelParameterDesc] = {
    // initEmptyArrays - mostly used by output argument list
    // set the number of entries to numentries as its initialized to '0'
    val colOrderSizes = colSchema.orderedColumns(columnsOrder) zip listKPD.map(_.sz)

    val kernParamDesc = colOrderSizes.map { col =>
      cachedGPUPointers.getOrElseUpdate(blockId.get + "_stream_"+streamNumber + "_" + col._1.prettyAccessor, {
        val cname = col._1.prettyAccessor.split("\\.").reverse.head
        val symbol = if (colSchema.isPrimitive) {
          null
        } else {
          valVarMembers.find(_.name.toString.startsWith(cname)).get
        }

        val colDataSize: Int = col._2
        val devPtr = GPUSparkEnv.get.cudaManager.allocateGPUMemory(colDataSize)
        cuMemsetD32Async(devPtr, 0, colDataSize / 4, cuStream)
        val gPtr = Pointer.to(devPtr)

        // Allocate only GPU memory; main memory will be allocated during deserialization
        new KernelParameterDesc(null, null, devPtr, gPtr, colDataSize, symbol)
      })
    }
//    cuCtxSynchronize()
    setOfkernParmDesc.put(blockId.get + "_stream_"+streamNumber, kernParamDesc)
    kernParamDesc
  }

  def lazyInputListKernParmDesc(stream: CUstream, streamNumber: Int): Iterator[(Seq[KernelParameterDesc])] = if (iter.hasNext) {
    // initFromInputIterator
    println(s"thread $threadId lazy list KernParmDesc start at ${System.nanoTime()}")
    Iterator.continually {


      val kernParamDesc = colSchema.orderedColumns(columnsOrder).map { col =>

        cachedGPUPointers.getOrElseUpdate(blockId.get + "_stream_"+streamNumber + "_" + col.prettyAccessor, {
          val cname = col.prettyAccessor.split("\\.").reverse.head
          val symbol = if (colSchema.isPrimitive) {
            null
          } else {
            valVarMembers.find(_.name.toString.startsWith(cname)).get
          }

          val (hPtr: Pointer, colDataSize: Int) = {
            val mirror = ColumnPartitionSchema.mirror
            val priv_getter = col.terms.foldLeft(identity[Any] _)((r, term) =>
              ((obj: Any) => mirror.reflect(obj).reflectField(term).get) compose r)

            var bufferOffset = 0
            var count = 0

            var start = 0L
            var tempEnd = 0L
//            var accumTempTime:Long = 0
//            var accumputIntTime: Long = 0

            val result = col.columnType match {
              case c if c == INT_COLUMN => {
                println(s"create buffer at ${System.nanoTime()}")
                val size = col.memoryUsage(subPartitionSize).toInt
                val (ptr, buffer) = allocPinnedHeap(size)
                println(s"put input into buffer at ${System.nanoTime()}")
                while(count < subPartitionSize && iter.hasNext) {
//                  start = System.nanoTime()
//                  val temp = priv_getter(iter.next()).asInstanceOf[Int]
//                  tempEnd = System.nanoTime()
//                  accumTempTime += tempEnd - start
                  buffer.putInt(priv_getter(iter.next()).asInstanceOf[Int])
//                  accumputIntTime += System.nanoTime() - tempEnd
                  count += 1
                }
//                println(s"accum temp time = ${accumTempTime}, accum putInt time = ${accumputIntTime}")
                (ptr, if(count == subPartitionSize) size else col.memoryUsage(count).toInt)
              }
              case c if c == LONG_COLUMN => {
                val size = col.memoryUsage(subPartitionSize).toInt
                val (ptr, buffer) = allocPinnedHeap(size)
                while(count < subPartitionSize && iter.hasNext) {
                  buffer.putLong(priv_getter(iter.next()).asInstanceOf[Long])
                  count += 1
                }
                (ptr, if(count == subPartitionSize) size else col.memoryUsage(count).toInt)
              }
              case c if c == SHORT_COLUMN => {
                val size = col.memoryUsage(subPartitionSize).toInt
                val (ptr, buffer) = allocPinnedHeap(size)
                while(count < subPartitionSize && iter.hasNext) {
                  buffer.putShort(priv_getter(iter.next()).asInstanceOf[Short])
                  count += 1
                }
                (ptr, if(count == subPartitionSize) size else col.memoryUsage(count).toInt)
              }
              case c if c == BYTE_COLUMN => {
                val size = col.memoryUsage(subPartitionSize).toInt
                val (ptr, buffer) = allocPinnedHeap(size)
                while(count < subPartitionSize && iter.hasNext) {
                  buffer.put(priv_getter(iter.next()).asInstanceOf[Byte])
                  count += 1
                }
                (ptr, if(count == subPartitionSize) size else col.memoryUsage(count).toInt)
              }
              case c if c == FLOAT_COLUMN => {
                val size = col.memoryUsage(subPartitionSize).toInt
                val (ptr, buffer) = allocPinnedHeap(size)
                while(count < subPartitionSize && iter.hasNext) {
                  buffer.putFloat(priv_getter(iter.next()).asInstanceOf[Float])
                  count += 1
                }
                (ptr, if(count == subPartitionSize) size else col.memoryUsage(count).toInt)
              }
              case c if c == DOUBLE_COLUMN => {
                val size = col.memoryUsage(subPartitionSize).toInt
                val (ptr, buffer) = allocPinnedHeap(size)
                while(count < subPartitionSize && iter.hasNext) {
                  buffer.putDouble(priv_getter(iter.next()).asInstanceOf[Double])
                  count += 1
                }
                (ptr, if(count == subPartitionSize) size else col.memoryUsage(count).toInt)
              }
              case c if c == INT_ARRAY_COLUMN => {
                // retrieve the first element to determine the array size.
                val temp = iter.next()
                val arrLength = priv_getter(temp).asInstanceOf[Array[Int]].length
                val size = col.memoryUsage(subPartitionSize * arrLength).toInt
                val (ptr, buffer) = allocPinnedHeap(size)

                while(count < subPartitionSize && iter.hasNext) {
                  buffer.position(bufferOffset)
                  if(count == 0)
                    buffer.asIntBuffer().put(priv_getter(temp).asInstanceOf[Array[Int]])
                  else
                    buffer.asIntBuffer().put(priv_getter(iter.next()).asInstanceOf[Array[Int]])
                  count += 1
                  bufferOffset += arrLength * INT_COLUMN.bytes
                }
                (ptr, if(count == subPartitionSize) size else col.memoryUsage(count).toInt)
              }
              case c if c == LONG_ARRAY_COLUMN => {
                // retrieve the first element to determine the array size.
                val temp = iter.next()
                val arrLength = priv_getter(temp).asInstanceOf[Array[Long]].length
                val size = col.memoryUsage(subPartitionSize * arrLength).toInt
                val (ptr, buffer) = allocPinnedHeap(size)

                while(count < subPartitionSize && iter.hasNext) {
                  buffer.position(bufferOffset)
                  if(count == 0)
                    buffer.asLongBuffer().put(priv_getter(temp).asInstanceOf[Array[Long]])
                  else
                    buffer.asLongBuffer().put(priv_getter(iter.next()).asInstanceOf[Array[Long]])
                  count += 1
                  bufferOffset += arrLength * INT_COLUMN.bytes
                }
                (ptr, if(count == subPartitionSize) size else col.memoryUsage(count).toInt)
              }
              case c if c == FLOAT_ARRAY_COLUMN => {
                // retrieve the first element to determine the array size.
                val temp = iter.next()
                val arrLength = priv_getter(temp).asInstanceOf[Array[Float]].length
                val size = col.memoryUsage(subPartitionSize * arrLength).toInt
                val (ptr, buffer) = allocPinnedHeap(size)

                while(count < subPartitionSize && iter.hasNext) {
                  buffer.position(bufferOffset)
                  if(count == 0)
                    buffer.asFloatBuffer().put(priv_getter(temp).asInstanceOf[Array[Float]])
                  else
                    buffer.asFloatBuffer().put(priv_getter(iter.next()).asInstanceOf[Array[Float]])
                  count += 1
                  bufferOffset += arrLength * INT_COLUMN.bytes
                }
                (ptr, if(count == subPartitionSize) size else col.memoryUsage(count).toInt)
              }
              case c if c == DOUBLE_ARRAY_COLUMN => {
                // retrieve the first element to determine the array size.
                val temp = iter.next()
                val arrLength = priv_getter(temp).asInstanceOf[Array[Double]].length
                val size = col.memoryUsage(subPartitionSize * arrLength).toInt
                val (ptr, buffer) = allocPinnedHeap(size)

                while(count < subPartitionSize && iter.hasNext) {
                  buffer.position(bufferOffset)
                  if(count == 0)
                    buffer.asDoubleBuffer().put(priv_getter(temp).asInstanceOf[Array[Double]])
                  else
                    buffer.asDoubleBuffer().put(priv_getter(iter.next()).asInstanceOf[Array[Double]])
                  count += 1
                  bufferOffset += arrLength * INT_COLUMN.bytes
                }
                (ptr, if(count == subPartitionSize) size else col.memoryUsage(count).toInt)
              }
            }
            elementCount = count
            result
          }
          setElemeCount.put(blockId.get + "_stream_" + streamNumber, elementCount)
          println(s"thread $threadId HybridIterator H2D start at ${System.nanoTime()}")
          val devPtr = GPUSparkEnv.get.cudaManager.allocateGPUMemory(colDataSize)
          cuMemcpyHtoDAsync(devPtr, hPtr, colDataSize, stream)
          val gPtr = Pointer.to(devPtr)

          // mark the cpuPtr null as we use pinned memory and got the Pointer directly
          new KernelParameterDesc(null, hPtr, devPtr, gPtr, colDataSize, symbol)
        })
      }

      setOfkernParmDesc.put(blockId.get + "_stream_"+streamNumber, kernParamDesc)
      totalElementCount = totalElementCount / kernParamDesc.length // TODO this have to be changed more elegantly for multi-columns
      totalChunkCount += 1
      kernParamDesc
    }.takeWhile(seq => seq(0).sz != 0)
  } else {
    null
  }

  def lazyInputListKernParmDescTest(stream: CUstream, streamNumber: Int): Iterator[(Seq[KernelParameterDesc])] = if (iter.hasNext) {
    // initFromInputIterator
    println(s"thread $threadId lazy list KernParmDesc Test version start at ${System.nanoTime()}")
    Iterator.continually {

      val check = colSchema.orderedColumns(columnsOrder).map{ col =>
        cachedGPUPointers.get(blockId.get + "_stream_"+streamNumber + "_" + col.prettyAccessor)
      }

      if(check.forall(_ == None)) {

        val tempIterHead = iter.next() // for Array elements

        println(s"thread $threadId create buffer start at ${System.nanoTime()}")
        val kernParamBuffer = colSchema.orderedColumns(columnsOrder).map { col =>
          val mirror = ColumnPartitionSchema.mirror
          val priv_getter = col.terms.foldLeft(identity[Any] _)((r, term) =>
            ((obj: Any) => mirror.reflect(obj).reflectField(term).get) compose r)

          col.columnType match {
            case c if c == INT_COLUMN => {
              val size = col.memoryUsage(subPartitionSize).toInt
              val (ptr, buffer) = allocPinnedHeap(size)
              (ptr, buffer, col, priv_getter, 1)
            }
            case c if c == LONG_COLUMN => {
              val size = col.memoryUsage(subPartitionSize).toInt
              val (ptr, buffer) = allocPinnedHeap(size)
              (ptr, buffer, col, priv_getter, 1)
            }
            case c if c == SHORT_COLUMN => {
              val size = col.memoryUsage(subPartitionSize).toInt
              val (ptr, buffer) = allocPinnedHeap(size)
              (ptr, buffer, col, priv_getter, 1)
            }
            case c if c == BYTE_COLUMN => {
              val size = col.memoryUsage(subPartitionSize).toInt
              val (ptr, buffer) = allocPinnedHeap(size)
              (ptr, buffer, col, priv_getter, 1)
            }
            case c if c == FLOAT_COLUMN => {
              val size = col.memoryUsage(subPartitionSize).toInt
              val (ptr, buffer) = allocPinnedHeap(size)
              (ptr, buffer, col, priv_getter, 1)
            }
            case c if c == DOUBLE_COLUMN => {
              val size = col.memoryUsage(subPartitionSize).toInt
              val (ptr, buffer) = allocPinnedHeap(size)
              (ptr, buffer, col, priv_getter, 1)
            }
            case c if c == INT_ARRAY_COLUMN => {
              // retrieve the first element to determine the array size.
              val arrLength = priv_getter(tempIterHead).asInstanceOf[Array[Int]].length
              val size = col.memoryUsage(subPartitionSize * arrLength).toInt
              val (ptr, buffer) = allocPinnedHeap(size)
              (ptr, buffer, col, priv_getter, arrLength)
            }
            case c if c == LONG_ARRAY_COLUMN => {
              // retrieve the first element to determine the array size.
              val arrLength = priv_getter(tempIterHead).asInstanceOf[Array[Long]].length
              val size = col.memoryUsage(subPartitionSize * arrLength).toInt
              val (ptr, buffer) = allocPinnedHeap(size)
              (ptr, buffer, col, priv_getter, arrLength)
            }
            case c if c == FLOAT_ARRAY_COLUMN => {
              // retrieve the first element to determine the array size.
              val arrLength = priv_getter(tempIterHead).asInstanceOf[Array[Float]].length
              val size = col.memoryUsage(subPartitionSize * arrLength).toInt
              val (ptr, buffer) = allocPinnedHeap(size)
              (ptr, buffer, col, priv_getter, arrLength)
            }
            case c if c == DOUBLE_ARRAY_COLUMN => {
              // retrieve the first element to determine the array size.
              val arrLength = priv_getter(tempIterHead).asInstanceOf[Array[Double]].length
              val size = col.memoryUsage(subPartitionSize * arrLength).toInt
              val (ptr, buffer) = allocPinnedHeap(size)
              (ptr, buffer, col, priv_getter, arrLength)
            }
          }
        }

        println(s"thread $threadId put input into buffer start at ${System.nanoTime()}")
        var countElem = 0
        var bufferOffset2 = 0
//        var getterAccumTime: Long = 0
//        var putAccumTime: Long = 0
        while (countElem < subPartitionSize && iter.hasNext) {
          val elem = if (countElem == 0) tempIterHead else iter.next()
          kernParamBuffer.map { case (cpuPtr, buffer, col, getter, dim) =>
            col.columnType match {
              case c if c == INT_COLUMN => {
//                val start = System.nanoTime()
//                val temp = getter(elem).asInstanceOf[Int]
//                val tempEnd = System.nanoTime()
//                getterAccumTime += tempEnd - start
                buffer.putInt(getter(elem).asInstanceOf[Int])
//                putAccumTime += System.nanoTime() - tempEnd
              }
              case c if c == LONG_COLUMN => {
                buffer.putLong(getter(elem).asInstanceOf[Long])
              }
              case c if c == SHORT_COLUMN => {
                buffer.putShort(getter(elem).asInstanceOf[Short])
              }
              case c if c == BYTE_COLUMN => {
                buffer.put(getter(elem).asInstanceOf[Byte])
              }
              case c if c == FLOAT_COLUMN => {
                buffer.putFloat(getter(elem).asInstanceOf[Float])
              }
              case c if c == DOUBLE_COLUMN => {
                buffer.putDouble(getter(elem).asInstanceOf[Double])
              }
              case c if c == INT_ARRAY_COLUMN => {
                buffer.position(bufferOffset2)
                buffer.asIntBuffer().put(getter(elem).asInstanceOf[Array[Int]])
                bufferOffset2 += dim * INT_COLUMN.bytes
              }
              case c if c == LONG_ARRAY_COLUMN => {
                buffer.position(bufferOffset2)
                buffer.asLongBuffer().put(getter(elem).asInstanceOf[Array[Long]])
                bufferOffset2 += dim * LONG_COLUMN.bytes
              }
              case c if c == FLOAT_ARRAY_COLUMN => {
                buffer.position(bufferOffset2)
                buffer.asFloatBuffer().put(getter(elem).asInstanceOf[Array[Float]])
                bufferOffset2 += dim * FLOAT_COLUMN.bytes
              }
              case c if c == DOUBLE_ARRAY_COLUMN => {
                buffer.position(bufferOffset2)
                buffer.asDoubleBuffer().put(getter(elem).asInstanceOf[Array[Double]])
                bufferOffset2 += dim * DOUBLE_COLUMN.bytes
              }
            }
          }
          countElem += 1
        }

//        println(s"thread $threadId getter accum time = ${getterAccumTime / 1000000}, " +
//          s"put accum time = ${putAccumTime / 1000000}")

        println(s"thread $threadId memcpy HtoD async at ${System.nanoTime()}")
        val kernParamDesc2 = kernParamBuffer.map { case (cpuPtr, buffer, col, getter, dim) =>
          cachedGPUPointers.getOrElseUpdate(blockId.get + "_stream_" + streamNumber + "_" + col.prettyAccessor, {
            val cname = col.prettyAccessor.split("\\.").reverse.head
            val symbol = if (colSchema.isPrimitive) {
              null
            } else {
              valVarMembers.find(_.name.toString.startsWith(cname)).get
            }

            val colDataSize = col.memoryUsage(countElem).toInt * dim
            val devPtr = GPUSparkEnv.get.cudaManager.allocateGPUMemory(colDataSize)
            cuMemcpyHtoDAsync(devPtr, cpuPtr, colDataSize, stream)
            val gPtr = Pointer.to(devPtr)

            // mark the cpuPtr null as we use pinned memory and got the Pointer directly
            new KernelParameterDesc(null, cpuPtr, devPtr, gPtr, colDataSize, symbol)
          })
        }

        println(s"thread $threadId ended memcpy HtoD async at ${System.nanoTime()}")
        setOfkernParmDesc.put(blockId.get + "_stream_" + streamNumber, kernParamDesc2)
        setElemeCount.put(blockId.get + "_stream_" + streamNumber, countElem)
        elementCount = countElem // TODO this have to be changed more elegantly for multi-columns
        totalChunkCount += 1

        kernParamDesc2
      } else {
        check.flatten
      }
    }.takeWhile(seq => seq(0).sz != 0)
  } else {
    null
  }

  // Use reflection to instantiate object without calling constructor
  private def instantiateClass(cls: Class[_]): AnyRef = {
    val rf = sun.reflect.ReflectionFactory.getReflectionFactory
    val parentCtor = classOf[java.lang.Object].getDeclaredConstructor()
    val newCtor = rf.newConstructorForSerialization(cls, parentCtor)
    val obj = newCtor.newInstance().asInstanceOf[AnyRef]
    obj
  }

  // Extract the setter method from the given object using reflection
  private def setter[C](obj: Any, value: C, symbol: TermSymbol) = {
    currentMirror.reflect(obj).reflectField(symbol).set(value)
  }

  def deserializeColumnValue(columnType: ColumnType, cpuArr: Array[_ >: Byte with Short with Int
    with Float with Long with Double <: AnyVal], index: Int, outsize: Int = 0): Any = {
    columnType match {
      case  INT_COLUMN => cpuArr(index).asInstanceOf[Int]
      case  LONG_COLUMN => cpuArr(index).asInstanceOf[Long]
      case  SHORT_COLUMN => cpuArr(index).asInstanceOf[Short]
      case  BYTE_COLUMN => cpuArr(index).asInstanceOf[Byte]
      case  FLOAT_COLUMN => cpuArr(index).asInstanceOf[Float]
      case  DOUBLE_COLUMN => cpuArr(index).asInstanceOf[Double]
      case  LONG_COLUMN => cpuArr(index).asInstanceOf[Long]
      case  INT_ARRAY_COLUMN => {
        val array = new Array[Int](outsize)
        var runIndex = index
        for (i <- 0 to outsize - 1) {
          array(i) = cpuArr(runIndex).asInstanceOf[Int]
          runIndex += 1
        }
        array
      }
      case  LONG_ARRAY_COLUMN => {
        val array = new Array[Long](outsize)
        var runIndex = index
        for (i <- 0 to outsize - 1) {
          array(i) = cpuArr(runIndex).asInstanceOf[Long]
          runIndex += 1
        }
        array
      }
      case  FLOAT_ARRAY_COLUMN => {
        val array = new Array[Float](outsize)
        var runIndex = index

        for (i <- 0 to outsize - 1) {
          array(i) = cpuArr(runIndex).asInstanceOf[Float]
          runIndex += 1
        }
        array
      }
      case  DOUBLE_ARRAY_COLUMN => {
        val array = new Array[Double](outsize)
        var runIndex = index

        for (i <- 0 to outsize - 1) {
          array(i) = cpuArr(runIndex).asInstanceOf[Double]
          runIndex += 1
        }
        array
      }
      case  LONG_ARRAY_COLUMN => {
        val array = new Array[Long](outsize)
        var runIndex = index

        for (i <- 0 to outsize - 1) {
          array(i) = cpuArr(runIndex).asInstanceOf[Long]
          runIndex += 1
        }
        array
      }
    }
  }

  def deserializeColumnValueTest(columnType: ColumnType, cpuArr: ByteBuffer, index: Int, outsize: Int = 0): Any = {
    columnType match {
      case  INT_COLUMN => cpuArr.getInt(index*INT_COLUMN.bytes)
      case  LONG_COLUMN => cpuArr.getLong(index*LONG_COLUMN.bytes)
      case  SHORT_COLUMN => cpuArr.getShort(index*SHORT_COLUMN.bytes)
      case  BYTE_COLUMN => cpuArr.get(index)
      case  FLOAT_COLUMN => cpuArr.getFloat(index*FLOAT_COLUMN.bytes)
      case  DOUBLE_COLUMN => cpuArr.getDouble(index*DOUBLE_COLUMN.bytes)
      case  INT_ARRAY_COLUMN => {
        val array = new Array[Int](outsize)
        var runIndex = index
        for (i <- 0 to outsize - 1) {
          array(i) = cpuArr.getInt(runIndex*INT_COLUMN.bytes).asInstanceOf[Int]
          runIndex += 1
        }
        array
      }
      case  LONG_ARRAY_COLUMN => {
        val array = new Array[Long](outsize)
        var runIndex = index
        for (i <- 0 to outsize - 1) {
          array(i) = cpuArr.getLong(runIndex*LONG_COLUMN.bytes).asInstanceOf[Long]
          runIndex += 1
        }
        array
      }
      case  FLOAT_ARRAY_COLUMN => {
        val array = new Array[Float](outsize)
        var runIndex = index

        for (i <- 0 to outsize - 1) {
          array(i) = cpuArr.getFloat(runIndex*FLOAT_COLUMN.bytes).asInstanceOf[Float]
          runIndex += 1
        }
        array
      }
      case  DOUBLE_ARRAY_COLUMN => {
        val array = new Array[Double](outsize)
        var runIndex = index

        for (i <- 0 to outsize - 1) {
          array(i) = cpuArr.getDouble(runIndex*DOUBLE_COLUMN.bytes).asInstanceOf[Double]
          runIndex += 1
        }
        array
      }
    }
  }

  // if hybridIterator copy the data from gpu to cpu pinned memory, you have to use LITTLE ENDIAN byteorder
  def getChunkResultIteratorTest: Iterator[T] = {
    println(s"thread $threadId iterator deserialize start time : ${System.nanoTime()}") // debug
    val runtimeCls = implicitly[ClassTag[T]].runtimeClass

    var index = 0
    var kernIndex = 0
    val tempArr = Array.tabulate(totalChunkCount){ idx =>
      (colSchema.orderedColumns(columnsOrder),
        setOfkernParmDesc(blockId.get + "_stream_"+idx).map(kpd => (kpd.cpuPtr.getByteBuffer(0, kpd.sz).order(ByteOrder.LITTLE_ENDIAN), kpd.symbol)),
        _outputArraySizes
        ).zipped
    }
    val result = if (colSchema.isPrimitive) {
      Iterator.continually {
        val result = tempArr(kernIndex).map(
          (col, cbuf, outsize) => {
            val retObj = deserializeColumnValueTest(col.columnType,
              cbuf._1, index * outsize, outsize)
            retObj
          }).head
        index += 1
        if(index % subPartitionSize == 0) {
          kernIndex += 1
          index = 0
        }
        result.asInstanceOf[T]
      } take totalElementCount
    } else {
      // For non-primitive types create an object on the fly and populate the values
      Iterator.continually {
        val retObj = instantiateClass(runtimeCls)
        tempArr(kernIndex).foreach(
          (col, cbuf, outsize) => {
            setter(retObj, deserializeColumnValueTest(col.columnType, cbuf._1,
              index * outsize, outsize), cbuf._2)
          })
        index += 1
        if(index % subPartitionSize == 0 || numElements == index) {
          kernIndex += 1
          index = 0
        }
        retObj.asInstanceOf[T]
      } take (numElements * kernIndex)
    }

    println(s"thread $threadId iterator deserialize end time : ${System.nanoTime()}") // debug

    result
  }


  // create iterator for reduce bottleneck of create array by lazy
  // getChunkResultIterator duration will be added to map task record write time
  def getChunkResultIterator: Iterator[T] = {
    println(s"thread $threadId chunk iterator deserialize start time : ${System.nanoTime()}") // debug
    val runtimeCls = implicitly[ClassTag[T]].runtimeClass

    var index = 0
    var kernIndex = 0
    val tempArr = Array.tabulate(totalChunkCount){ idx =>
      (colSchema.orderedColumns(columnsOrder), setOfkernParmDesc(blockId.get + "_stream_"+idx), _outputArraySizes).zipped
    }
    val tempLengthArr = Array.tabulate(totalChunkCount){ idx =>
      setElemeCount(blockId.get + "_stream_" + idx)
    }

    val result = if (colSchema.isPrimitive) {
      Iterator.continually {
        val result = tempArr(kernIndex).map(
          (col, cdesc, outsize) => {
            val retObj = deserializeColumnValue(col.columnType,
              cdesc.cpuArr, index * outsize, outsize)
            retObj
          }).head
        index += 1
        if(index % tempLengthArr(kernIndex) == 0 || numentries == index) {
          kernIndex += 1
          index = 0
        }
        result.asInstanceOf[T]
      } take (if(numentries != 0) numentries * totalChunkCount else totalElementCount)
    } else {
      // For non-primitive types create an object on the fly and populate the values
      Iterator.continually {
        val retObj = instantiateClass(runtimeCls)
        tempArr(kernIndex).foreach(
          (col, cdesc, outsize) => {
            setter(retObj, deserializeColumnValue(col.columnType, cdesc.cpuArr,
              index * outsize, outsize), cdesc.symbol)
          })
        index += 1
        if(index % subPartitionSize == 0) {
          kernIndex += 1
          index = 0
        }
        retObj.asInstanceOf[T]
      } take totalElementCount
    }

    println(s"thread $threadId iterator deserialize end time : ${System.nanoTime()}") // debug

    result
  }


  def getResultIterator: Iterator[T] = {
    println(s"thread $threadId iterator deserialize start time : ${System.nanoTime()}") // debug
    val runtimeCls = implicitly[ClassTag[T]].runtimeClass

    var index = 0
    val result = if (colSchema.isPrimitive) {
      val temp = (colSchema.orderedColumns(columnsOrder), listKernParmDesc, _outputArraySizes).zipped //create obj for every iterator
      Iterator.continually {
        val result = temp.map(
          (col, cdesc, outsize) => {
            val retObj = deserializeColumnValue(col.columnType,
              cdesc.cpuArr, index * outsize, outsize)
            retObj
          }).head
        index += 1
        result.asInstanceOf[T]
      } take numElements
    } else {
      // For non-primitive types create an object on the fly and populate the values
      Iterator.continually {
        val retObj = instantiateClass(runtimeCls)

        (colSchema.orderedColumns(columnsOrder), listKernParmDesc,
          _outputArraySizes).zipped.foreach(
          (col, cdesc, outsize) => {
            setter(retObj, deserializeColumnValue(col.columnType, cdesc.cpuArr,
              index * outsize, outsize), cdesc.symbol)
          })
        index += 1
        retObj.asInstanceOf[T]
      } take numElements
    }

    println(s"thread $threadId iterator deserialize end time : ${System.nanoTime()}") // debug

    result
  }

  def getResultList: Array[T] = {
    println(s"thread $threadId array deserialize start time : ${System.nanoTime()}") // debug
    val resultsArray = new Array[T](numElements)
    val runtimeCls = implicitly[ClassTag[T]].runtimeClass

    val temp = (colSchema.orderedColumns(columnsOrder), listKernParmDesc, _outputArraySizes).zipped

    for (index <- 0 to numElements - 1) {
      val obj = if (colSchema.isPrimitive) {
        temp.map(
          (col, cdesc, outsize) => {
            val retObj = deserializeColumnValue(col.columnType,
              cdesc.cpuArr, index * outsize, outsize)
            retObj
        }).head
      } else {
        // For non-primitive types create an object on the fly and populate the values
        val retObj = instantiateClass(runtimeCls)

        temp.foreach(
          (col, cdesc, outsize) => {
            setter(retObj, deserializeColumnValue(col.columnType, cdesc.cpuArr,
              index * outsize, outsize), cdesc.symbol)
          })
        retObj
      }
      resultsArray(index) = obj.asInstanceOf[T]
    }
    println(s"thread $threadId array deserialize end time : ${System.nanoTime()}") // debug
    resultsArray
  }
}



