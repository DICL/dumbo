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
import org.apache.hadoop.fs.FilterFs;

public class LustreFs extends FilterFs {

    LustreFs(Configuration conf) throws IOException, URISyntaxException{
        super(new LustreVol(conf));
    }

    LustreFs(final URI theUri, final Configuration conf) throws IOException, URISyntaxException{
        this(conf);
    }


}
