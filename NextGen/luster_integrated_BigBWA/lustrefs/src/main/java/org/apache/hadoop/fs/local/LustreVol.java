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

package org.apache.hadoop.fs.local;

import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FsConstants;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.RawLocalFileSystem;
import org.apache.hadoop.fs.lustrefs.LustreFileSystem;
import org.apache.hadoop.fs.lustrefs.LustreVolume;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class LustreVol extends RawLocalFsL{
   
    protected static final Logger log = LoggerFactory.getLogger(LustreFileSystem.class);
    
    LustreVol(final Configuration conf) throws IOException, URISyntaxException {
        this(LustreVolume.NAME, conf);
        
    }
      
      /**
       * This constructor has the signature needed by
       * {@link AbstractFileSystem#createFileSystem(URI, Configuration)}.
       * 
       * @param theUri which must be that of lustreFs
       * @param conf
       * @throws IOException
       * @throws URISyntaxException 
       */
    LustreVol(final URI theUri, final Configuration conf) throws IOException, URISyntaxException {
        super(theUri, new LustreVolume(), conf, false);
    }
    
    
    
    

}
