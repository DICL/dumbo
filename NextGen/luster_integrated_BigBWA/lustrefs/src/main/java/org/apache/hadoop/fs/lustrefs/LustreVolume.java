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
import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.URI;
import java.util.Arrays;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.BlockLocation;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.RawLocalFileSystem;
import org.apache.hadoop.fs.permission.FsPermission;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class LustreVolume extends RawLocalFileSystem{

    static final Logger log = LoggerFactory.getLogger(LustreVolume.class);

    /**
     * General reason for these constants is to help us decide
     * when to override the specified buffer size.  See implementation 
     * of logic below, which might change overtime.
     */
    public static final int OVERRIDE_WRITE_BUFFER_SIZE = 1024 * 4;
    public static final int OPTIMAL_WRITE_BUFFER_SIZE = 1024 * 128;
    
    public static final String fname = "lustrefs:";
    
    public static final URI NAME = URI.create(fname + "///");
    
    protected String root=null;
    protected String superUser=null;
    protected AclPathFilter aclFilter = null;
    
    public LustreVolume(){
    }
    
    public LustreVolume(Configuration conf){
        this();
        try {
            this.initialize(NAME, conf);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        this.setConf(conf);
    }
    public URI getUri() { return NAME; }
    
    public void setConf(Configuration conf){
        super.setConf(conf);
        if(conf!=null){
         
            try{
                root=conf.get("fs.lustrefs.mount", null);
                log.info("Root of Lustre file system is " + root);
                String jtSysDir = conf.get("mapreduce.jobtracker.system.dir", null);
                Path mapredSysDirectory = null;
                
                if(jtSysDir!=null)
                    mapredSysDirectory = new Path(jtSysDir);
                else{
                    mapredSysDirectory = new Path(conf.get("mapred.system.dir", "lustrefs:///mapred/system"));
                }
                
                if(!exists(mapredSysDirectory)){
                    mkdirs(mapredSysDirectory);
                }
                //ACL setup
                aclFilter = new AclPathFilter(conf);
		//superUser =  conf.get("lustrefs.daemon.user", null);
		//log.info("mapreduce/superuser daemon : " + superUser);

                //Working directory setup
                Path workingDirectory = getInitialWorkingDirectory();
                mkdirs(workingDirectory);
                setWorkingDirectory(workingDirectory);
                log.info("Working directory is : "+ getWorkingDirectory());

                /**
                 * Write Buffering
                 */
                Integer userBufferSize=conf.getInt("io.file.buffer.size", -1);
                if(userBufferSize == OVERRIDE_WRITE_BUFFER_SIZE || userBufferSize == -1) {
                	conf.setInt("io.file.buffer.size", OPTIMAL_WRITE_BUFFER_SIZE);
                }
                log.info("Write buffer size : " +conf.getInt("io.file.buffer.size",-1)) ;
            }
            catch (Exception e){
                throw new RuntimeException(e);
            }
        }
        
    }
    
    public File pathToFile(Path path) {
      checkPath(path);
      if (!path.isAbsolute()) {
        path = new Path(getWorkingDirectory(), path);
      }
      String s = path.toString();
      if (s.startsWith(fname + root) || s.startsWith(fname + "/" + root) || 
            s.startsWith(fname + "//" + root) || s.startsWith(fname + "///" + root) ) {
        return new File(path.toUri().getPath());
      } else {
        return new File(root + path.toUri().getPath());
      }
    }
  
    /**
     * Note this method doesn't override anything in hadoop 1.2.0 and 
     * below.
     */
    protected Path getInitialWorkingDirectory() {
		/* apache's unit tests use a default working direcotry like this: */
       return new Path(this.NAME + root + "/" + "user/" + System.getProperty("user.name"));
        /* The super impl returns the users home directory in unix */
		//return super.getInitialWorkingDirectory();
	}

	public Path fileToPath(File path) {
        return new Path(NAME.toString() + path.toURI().getRawPath().substring(root.length()));
     }

    public boolean rename(Path src, Path dst) throws IOException {
	File dest = pathToFile(dst);
	/* two HCFS semantics java.io.File doesn't honor */
        if(dest.exists() && dest.isFile() || !(new File(dest.getParent()).exists())) return false;
	if (!dest.exists() && pathToFile(src).renameTo(dest)) {
	    return true;
	}
	return FileUtil.copy(this, src, this, dst, true, getConf());
    }
    
    /**
    * Delete the given path to a file or directory.
    * @param p the path to delete
    * @param recursive to delete sub-directories
    * @return true if the file or directory and all its contents were deleted
    * @throws IOException if p is non-empty and recursive is false 
    */
    @Override
    public boolean delete(Path p, boolean recursive) throws IOException {
        File f = pathToFile(p);
        if(!f.exists()){
            /* HCFS semantics expect 'false' if attempted file deletion on non existent file */
            return false;
        }else if (f.isFile()) {
            return f.delete();
        } else if (!recursive && f.isDirectory() && 
            (FileUtil.listFiles(f).length != 0)) {
            throw new IOException("Directory " + f.toString() + " is not empty");
        }
        return FileUtil.fullyDelete(f);
    }
	  
    public FileStatus[] listStatus(Path f) throws IOException {
        File localf = pathToFile(f);
        FileStatus[] results;
        if (!localf.exists()) {
          throw new FileNotFoundException("File " + f + " does not exist");
        }
        if (localf.isFile()) {
          return new FileStatus[] {
            new LustreFileStatus(localf, getDefaultBlockSize(), this) };
        }

        File[] names = localf.listFiles();
        if (names == null) {
          throw new FileNotFoundException("Directory " + f + " does not exist.");
	}
	Arrays.sort(names);
        results = new FileStatus[names.length];
        int j = 0;
        for (int i = 0; i < names.length; i++) {
          try {
            results[j] = getFileStatus(fileToPath(names[i]));
            j++;
          } catch (FileNotFoundException e) {
            // ignore the files not found since the dir list may have have changed
            // since the names[] list was generated.
          }
        }
        if (j == names.length) {
          return results;
        }
        return Arrays.copyOf(results, j);
    }
    
    @Override
    public FileStatus getFileStatus(Path f) throws IOException {
        File path = pathToFile(f);
        if (path.exists()) {
          return new LustreFileStatus(pathToFile(f), getDefaultBlockSize(), this);
        } else {
          throw new FileNotFoundException( "File " + f + " does not exist with path " + path);
        }
      }
    
    /*
     * ensures the 'super user' is given read/write access.  
     * the ACL drops off after a chmod or chown.
     */
    
    private void updateAcl(Path p){
    	if(superUser!=null && aclFilter.matches(p)  ){
    		File f = pathToFile(p);
    		String path = f.getAbsolutePath();
    		String command = "setfacl -m u:" + superUser + ":rwx " + path;
    		try{
    			Runtime.getRuntime().exec(command);
    		}catch(IOException ex){
    			throw new RuntimeException(ex);
    		}
    	}
    }
    
    public void setOwner(Path p, String username, String groupname)
            throws IOException {
    	super.setOwner(p,username,groupname);
    	updateAcl(p);
    	
    }
    
    public void setPermission(Path p, FsPermission permission)
            throws IOException {
    	super.setPermission(p,permission);
    	updateAcl(p);
    }

    public String toString(){
        return "Lustre Volume mounted at: " + root;
    }

}
