package org.apache.hadoop.fs.local;

import junit.framework.AssertionFailedError;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystemContractBaseTest;
import org.apache.hadoop.fs.Path;

import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;

import org.apache.hadoop.fs.lustrefs.LustreFileSystem;

/** 
 * A collection of tests for the LustreFS plugin.
 * Because LustreFS is a thin wrapper around the built-in Hadoop 
 * RawLocalFileSystem, this test class need only be a thin wrapper as well.
 */

public class LustreFileSystemContractBaseTest
        extends FileSystemContractBaseTest {
    private static final Log log =
        LogFactory.getLog(LustreFileSystemContractBaseTest.class);
  
    @Override
    protected void setUp() throws Exception {
        final Configuration conf = new Configuration();
        conf.set("fs.lustrefs.mount", "/tmp");
        fs = createLustreFS(conf);
        final URI uri = new URI("lustrefs:///");
        fs.initialize(uri, conf);
        /* try {
            fs.setConf(conf);
        } catch (RuntimeException e) {
            // Init failed, set fs to null so shutdown doesn't use it.
            fs = null;
            throw e;
        } */
        super.setUp();
    }
    
    @Override
    protected String getDefaultWorkingDirectory() {
        return "/tmp/user/" + System.getProperty("user.name");
    }
    
    protected LustreFileSystem createLustreFS() throws IOException {
        LustreFileSystem lustreFS = new LustreFileSystem();
    return lustreFS;
    }

    protected LustreFileSystem createLustreFS(Configuration conf) throws IOException {
        LustreFileSystem lustreFS = new LustreFileSystem(conf);
    return lustreFS;
    }
    
}