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

package org.apache.spark.sql.execution.streaming

import java.io._
import java.nio.charset.StandardCharsets
import java.util.{ConcurrentModificationException, EnumSet, UUID}

import scala.reflect.ClassTag

import org.apache.commons.io.IOUtils
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs._
import org.apache.hadoop.fs.permission.FsPermission
import org.json4s.NoTypeHints
import org.json4s.jackson.Serialization

import org.apache.spark.internal.Logging
import org.apache.spark.sql.SparkSession
import org.apache.spark.util.UninterruptibleThread


/**
 * A [[MetadataLog]] implementation based on HDFS. [[HDFSMetadataLog]] uses the specified `path`
 * as the metadata storage.
 *
 * When writing a new batch, [[HDFSMetadataLog]] will firstly write to a temp file and then rename
 * it to the final batch file. If the rename step fails, there must be multiple writers and only
 * one of them will succeed and the others will fail.
 *
 * Note: [[HDFSMetadataLog]] doesn't support S3-like file systems as they don't guarantee listing
 * files in a directory always shows the latest files.
 */
class HDFSMetadataLog[T <: AnyRef : ClassTag](sparkSession: SparkSession, path: String)
  extends MetadataLog[T] with Logging {

  private implicit val formats = Serialization.formats(NoTypeHints)

  /** Needed to serialize type T into JSON when using Jackson */
  private implicit val manifest = Manifest.classType[T](implicitly[ClassTag[T]].runtimeClass)

  // Avoid serializing generic sequences, see SPARK-17372
  require(implicitly[ClassTag[T]].runtimeClass != classOf[Seq[_]],
    "Should not create a log with type Seq, use Arrays instead - see SPARK-17372")

  import HDFSMetadataLog._

  val metadataPath = new Path(path)
  protected val fileManager = createFileManager()

  if (!fileManager.exists(metadataPath)) {
    fileManager.mkdirs(metadataPath)
  }

  /**
   * A `PathFilter` to filter only batch files
   */
  protected val batchFilesFilter = new PathFilter {
    override def accept(path: Path): Boolean = isBatchFile(path)
  }

  protected def batchIdToPath(batchId: Long): Path = {
    new Path(metadataPath, batchId.toString)
  }

  protected def pathToBatchId(path: Path) = {
    path.getName.toLong
  }

  protected def isBatchFile(path: Path) = {
    try {
      path.getName.toLong
      true
    } catch {
      case _: NumberFormatException => false
    }
  }

  protected def serialize(metadata: T, out: OutputStream): Unit = {
    // called inside a try-finally where the underlying stream is closed in the caller
    Serialization.write(metadata, out)
  }

  protected def deserialize(in: InputStream): T = {
    // called inside a try-finally where the underlying stream is closed in the caller
    val reader = new InputStreamReader(in, StandardCharsets.UTF_8)
    Serialization.read[T](reader)
  }

  /**
   * Store the metadata for the specified batchId and return `true` if successful. If the batchId's
   * metadata has already been stored, this method will return `false`.
   */
  override def add(batchId: Long, metadata: T): Boolean = {
    get(batchId).map(_ => false).getOrElse {
      // Only write metadata when the batch has not yet been written
      if (fileManager.isLocalFileSystem) {
        Thread.currentThread match {
          case ut: UninterruptibleThread =>
            // When using a local file system, "writeBatch" must be called on a
            // [[org.apache.spark.util.UninterruptibleThread]] so that interrupts can be disabled
            // while writing the batch file. This is because there is a potential dead-lock in
            // Hadoop "Shell.runCommand" before 2.5.0 (HADOOP-10622). If the thread running
            // "Shell.runCommand" is interrupted, then the thread can get deadlocked. In our case,
            // `writeBatch` creates a file using HDFS API and will call "Shell.runCommand" to set
            // the file permission if using the local file system, and can get deadlocked if the
            // stream execution thread is stopped by interrupt. Hence, we make sure that
            // "writeBatch" is called on [[UninterruptibleThread]] which allows us to disable
            // interrupts here. Also see SPARK-14131.
            ut.runUninterruptibly { writeBatch(batchId, metadata, serialize) }
          case _ =>
            throw new IllegalStateException(
              "HDFSMetadataLog.add() on a local file system must be executed on " +
                "a o.a.spark.util.UninterruptibleThread")
        }
      } else {
        // For a distributed file system, such as HDFS or S3, if the network is broken, write
        // operations may just hang until timeout. We should enable interrupts to allow stopping
        // the query fast.
        writeBatch(batchId, metadata, serialize)
      }
      true
    }
  }

  def writeTempBatch(metadata: T, writer: (T, OutputStream) => Unit = serialize): Option[Path] = {
    var nextId = 0
    while (true) {
      val tempPath = new Path(metadataPath, s".${UUID.randomUUID.toString}.tmp")
      try {
        val output = fileManager.create(tempPath)
        try {
          writer(metadata, output)
          return Some(tempPath)
        } finally {
          IOUtils.closeQuietly(output)
        }
      } catch {
        case e: IOException if isFileAlreadyExistsException(e) =>
          // Failed to create "tempPath". There are two cases:
          // 1. Someone is creating "tempPath" too.
          // 2. This is a restart. "tempPath" has already been created but not moved to the final
          // batch file (not committed).
          //
          // For both cases, the batch has not yet been committed. So we can retry it.
          //
          // Note: there is a potential risk here: if HDFSMetadataLog A is running, people can use
          // the same metadata path to create "HDFSMetadataLog" and fail A. However, this is not a
          // big problem because it requires the attacker must have the permission to write the
          // metadata path. In addition, the old Streaming also have this issue, people can create
          // malicious checkpoint files to crash a Streaming application too.
          nextId += 1
      }
    }
    None
  }

  /**
   * Write a batch to a temp file then rename it to the batch file.
   *
   * There may be multiple [[HDFSMetadataLog]] using the same metadata path. Although it is not a
   * valid behavior, we still need to prevent it from destroying the files.
   */
  private def writeBatch(batchId: Long, metadata: T, writer: (T, OutputStream) => Unit): Unit = {
    val tempPath = writeTempBatch(metadata, writer).getOrElse(
      throw new IllegalStateException(s"Unable to create temp batch file $batchId"))
    try {
      // Try to commit the batch
      // It will fail if there is an existing file (someone has committed the batch)
      logDebug(s"Attempting to write log #${batchIdToPath(batchId)}")
      fileManager.rename(tempPath, batchIdToPath(batchId))

      // SPARK-17475: HDFSMetadataLog should not leak CRC files
      // If the underlying filesystem didn't rename the CRC file, delete it.
      val crcPath = new Path(tempPath.getParent(), s".${tempPath.getName()}.crc")
      if (fileManager.exists(crcPath)) fileManager.delete(crcPath)
    } catch {
      case e: IOException if isFileAlreadyExistsException(e) =>
        // If "rename" fails, it means some other "HDFSMetadataLog" has committed the batch.
        // So throw an exception to tell the user this is not a valid behavior.
        throw new ConcurrentModificationException(
          s"Multiple HDFSMetadataLog are using $path", e)
      case e: FileNotFoundException =>
        // Sometimes, "create" will succeed when multiple writers are calling it at the same
        // time. However, only one writer can call "rename" successfully, others will get
        // FileNotFoundException because the first writer has removed it.
        throw new ConcurrentModificationException(
          s"Multiple HDFSMetadataLog are using $path", e)
    } finally {
      fileManager.delete(tempPath)
    }
  }

  private def isFileAlreadyExistsException(e: IOException): Boolean = {
    e.isInstanceOf[FileAlreadyExistsException] ||
      // Old Hadoop versions don't throw FileAlreadyExistsException. Although it's fixed in
      // HADOOP-9361 in Hadoop 2.5, we still need to support old Hadoop versions.
      (e.getMessage != null && e.getMessage.startsWith("File already exists: "))
  }

  /**
   * @return the deserialized metadata in a batch file, or None if file not exist.
   * @throws IllegalArgumentException when path does not point to a batch file.
   */
  def get(batchFile: Path): Option[T] = {
    if (fileManager.exists(batchFile)) {
      if (isBatchFile(batchFile)) {
        get(pathToBatchId(batchFile))
      } else {
        throw new IllegalArgumentException(s"File ${batchFile} is not a batch file!")
      }
    } else {
      None
    }
  }

  override def get(batchId: Long): Option[T] = {
    val batchMetadataFile = batchIdToPath(batchId)
    if (fileManager.exists(batchMetadataFile)) {
      val input = fileManager.open(batchMetadataFile)
      try {
        Some(deserialize(input))
      } finally {
        IOUtils.closeQuietly(input)
      }
    } else {
      logDebug(s"Unable to find batch $batchMetadataFile")
      None
    }
  }

  override def get(startId: Option[Long], endId: Option[Long]): Array[(Long, T)] = {
    val files = fileManager.list(metadataPath, batchFilesFilter)
    val batchIds = files
      .map(f => pathToBatchId(f.getPath))
      .filter { batchId =>
        (endId.isEmpty || batchId <= endId.get) && (startId.isEmpty || batchId >= startId.get)
    }
    batchIds.sorted.map(batchId => (batchId, get(batchId))).filter(_._2.isDefined).map {
      case (batchId, metadataOption) =>
        (batchId, metadataOption.get)
    }
  }

  override def getLatest(): Option[(Long, T)] = {
    val batchIds = fileManager.list(metadataPath, batchFilesFilter)
      .map(f => pathToBatchId(f.getPath))
      .sorted
      .reverse
    for (batchId <- batchIds) {
      val batch = get(batchId)
      if (batch.isDefined) {
        return Some((batchId, batch.get))
      }
    }
    None
  }

  /**
   * Get an array of [FileStatus] referencing batch files.
   * The array is sorted by most recent batch file first to
   * oldest batch file.
   */
  def getOrderedBatchFiles(): Array[FileStatus] = {
    fileManager.list(metadataPath, batchFilesFilter)
      .sortBy(f => pathToBatchId(f.getPath))
      .reverse
  }

  /**
   * Removes all the log entry earlier than thresholdBatchId (exclusive).
   */
  override def purge(thresholdBatchId: Long): Unit = {
    val batchIds = fileManager.list(metadataPath, batchFilesFilter)
      .map(f => pathToBatchId(f.getPath))

    for (batchId <- batchIds if batchId < thresholdBatchId) {
      val path = batchIdToPath(batchId)
      fileManager.delete(path)
      logTrace(s"Removed metadata log file: $path")
    }
  }

  private def createFileManager(): FileManager = {
    val hadoopConf = sparkSession.sessionState.newHadoopConf()
    try {
      new FileContextManager(metadataPath, hadoopConf)
    } catch {
      case e: UnsupportedFileSystemException =>
        logWarning("Could not use FileContext API for managing metadata log files at path " +
          s"$metadataPath. Using FileSystem API instead for managing log files. The log may be " +
          s"inconsistent under failures.")
        new FileSystemManager(metadataPath, hadoopConf)
    }
  }
}

object HDFSMetadataLog {

  /** A simple trait to abstract out the file management operations needed by HDFSMetadataLog. */
  trait FileManager {

    /** List the files in a path that matches a filter. */
    def list(path: Path, filter: PathFilter): Array[FileStatus]

    /** Make directory at the give path and all its parent directories as needed. */
    def mkdirs(path: Path): Unit

    /** Whether path exists */
    def exists(path: Path): Boolean

    /** Open a file for reading, or throw exception if it does not exist. */
    def open(path: Path): FSDataInputStream

    /** Create path, or throw exception if it already exists */
    def create(path: Path): FSDataOutputStream

    /**
     * Atomically rename path, or throw exception if it cannot be done.
     * Should throw FileNotFoundException if srcPath does not exist.
     * Should throw FileAlreadyExistsException if destPath already exists.
     */
    def rename(srcPath: Path, destPath: Path): Unit

    /** Recursively delete a path if it exists. Should not throw exception if file doesn't exist. */
    def delete(path: Path): Unit

    /** Whether the file systme is a local FS. */
    def isLocalFileSystem: Boolean
  }

  /**
   * Default implementation of FileManager using newer FileContext API.
   */
  class FileContextManager(path: Path, hadoopConf: Configuration) extends FileManager {
    private val fc = if (path.toUri.getScheme == null) {
      FileContext.getFileContext(hadoopConf)
    } else {
      FileContext.getFileContext(path.toUri, hadoopConf)
    }

    override def list(path: Path, filter: PathFilter): Array[FileStatus] = {
      fc.util.listStatus(path, filter)
    }

    override def rename(srcPath: Path, destPath: Path): Unit = {
      fc.rename(srcPath, destPath)
    }

    override def mkdirs(path: Path): Unit = {
      fc.mkdir(path, FsPermission.getDirDefault, true)
    }

    override def open(path: Path): FSDataInputStream = {
      fc.open(path)
    }

    override def create(path: Path): FSDataOutputStream = {
      fc.create(path, EnumSet.of(CreateFlag.CREATE))
    }

    override def exists(path: Path): Boolean = {
      fc.util().exists(path)
    }

    override def delete(path: Path): Unit = {
      try {
        fc.delete(path, true)
      } catch {
        case e: FileNotFoundException =>
        // ignore if file has already been deleted
      }
    }

    override def isLocalFileSystem: Boolean = fc.getDefaultFileSystem match {
      case _: local.LocalFs | _: local.RawLocalFs =>
        // LocalFs = RawLocalFs + ChecksumFs
        true
      case _ => false
    }
  }

  /**
   * Implementation of FileManager using older FileSystem API. Note that this implementation
   * cannot provide atomic renaming of paths, hence can lead to consistency issues. This
   * should be used only as a backup option, when FileContextManager cannot be used.
   */
  class FileSystemManager(path: Path, hadoopConf: Configuration) extends FileManager {
    private val fs = path.getFileSystem(hadoopConf)

    override def list(path: Path, filter: PathFilter): Array[FileStatus] = {
      fs.listStatus(path, filter)
    }

    /**
     * Rename a path. Note that this implementation is not atomic.
     * @throws FileNotFoundException if source path does not exist.
     * @throws FileAlreadyExistsException if destination path already exists.
     * @throws IOException if renaming fails for some unknown reason.
     */
    override def rename(srcPath: Path, destPath: Path): Unit = {
      if (!fs.exists(srcPath)) {
        throw new FileNotFoundException(s"Source path does not exist: $srcPath")
      }
      if (fs.exists(destPath)) {
        throw new FileAlreadyExistsException(s"Destination path already exists: $destPath")
      }
      if (!fs.rename(srcPath, destPath)) {
        throw new IOException(s"Failed to rename $srcPath to $destPath")
      }
    }

    override def mkdirs(path: Path): Unit = {
      fs.mkdirs(path, FsPermission.getDirDefault)
    }

    override def open(path: Path): FSDataInputStream = {
      fs.open(path)
    }

    override def create(path: Path): FSDataOutputStream = {
      fs.create(path, false)
    }

    override def exists(path: Path): Boolean = {
      fs.exists(path)
    }

    override def delete(path: Path): Unit = {
      try {
        fs.delete(path, true)
      } catch {
        case e: FileNotFoundException =>
          // ignore if file has already been deleted
      }
    }

    override def isLocalFileSystem: Boolean = fs match {
      case _: LocalFileSystem | _: RawLocalFileSystem =>
        // LocalFileSystem = RawLocalFileSystem + ChecksumFileSystem
        true
      case _ => false
    }
  }
}
