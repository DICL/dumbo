/**
 * Lustre file system for Hadoop
 * 
 * Uses lustrefs:/// as the file system.
 *
 * Based on the glusterfs-hadoop plugin
 * https://forge.gluster.org/hadoop/pages/Architecture
 * https://forge.gluster.org/hadoop
 *
 * Modifications by:
 * Seagate Technology
 * December 2014
 */


package org.apache.hadoop.fs.lustrefs;

import java.io.File;
import java.io.IOException;
import java.net.URI;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.fs.FilterFileSystem;
import org.apache.hadoop.fs.Path;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class LustreFileSystem extends FilterFileSystem{

    protected static final Logger log=LoggerFactory.getLogger(LustreFileSystem.class);
   
    public LustreFileSystem(){
        super(new LustreVolume());
        log.info("Initializing LustreFS.");
    }

    public LustreFileSystem(Configuration conf) {
        super(new LustreVolume(conf));
        log.info("Initializing LustreFS with conf.");
    }
   
    
    /** Convert a path to a File. */
    public File pathToFile(Path path){
        return ((LustreVolume) fs).pathToFile(path);
    }

    /**
     * Get file status.
     */
    public boolean exists(Path f) throws IOException{
        File path=pathToFile(f);
        if(path.exists()){
            return true;
        }else{
            return false;
        }
    }

    public void setConf(Configuration conf){
        log.info("Configuring LustreFS");
        super.setConf(conf);
    }

    /*
     * if LusreFileSystem is the default filesystem, real local URLs come back
     * without a file:/ scheme name (BUG!). the lustrefs file system is
     * assumed. force a schema.
     */

    public void copyFromLocalFile(boolean delSrc,Path src,Path dst) throws IOException{
        FileSystem srcFs=new Path("file:/"+src.toString()).getFileSystem(getConf());
        FileSystem dstFs=dst.getFileSystem(getConf());
        FileUtil.copy(srcFs, src, dstFs, dst, delSrc, getConf());
    }

    public void copyToLocalFile(boolean delSrc,Path src,Path dst) throws IOException{
        FileSystem srcFs=src.getFileSystem(getConf());
        FileSystem dstFs=new Path("file:/"+dst.toString()).getFileSystem(getConf());
        FileUtil.copy(srcFs, src, dstFs, dst, delSrc, getConf());
    }

    public String toString(){
        return "Lustre File System";
    }
}
