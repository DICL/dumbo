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

import java.io.DataOutput;
import java.io.File;
import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.permission.FsPermission;
import org.apache.hadoop.util.Shell;
import org.apache.hadoop.util.StringUtils;
/*
 * Copied from org.apache.fs.RawLocalFileSystem.RawFileStatus
 */
public class LustreFileStatus extends FileStatus{
    /*
     * We can add extra fields here. It breaks at least CopyFiles.FilePair(). We
     * recognize if the information is already loaded by check if
     * onwer.equals("").
     */
    protected LustreVolume fs;

    private boolean isPermissionLoaded(){
        return !super.getOwner().equals("");
    }

    
    /*
     * This constructor is the only difference than the RawLocalFileStatus impl.
     RawLocalFileSystem converts a raw file path to path with the same prefix.
     ends up with a double /mnt/lustre.
     */
    LustreFileStatus(File f, long defaultBlockSize, LustreVolume fs){
        super(f.length(), f.isDirectory(), 1, defaultBlockSize, f.lastModified(), fs.fileToPath(f));
        this.fs=fs;
    }

    @Override
    public FsPermission getPermission(){
        if(!isPermissionLoaded()){
            loadPermissionInfo();
        }
        return super.getPermission();
    }

    @Override
    public String getOwner(){
        if(!isPermissionLoaded()){
            loadPermissionInfo();
        }
        return super.getOwner();
    }

    @Override
    public String getGroup(){
        if(!isPermissionLoaded()){
            loadPermissionInfo();
        }
        return super.getGroup();
    }

    // / loads permissions, owner, and group from `ls -ld`
    private void loadPermissionInfo(){
        IOException e=null;
        try{
            StringTokenizer t=new StringTokenizer(Util.execCommand(fs.pathToFile(getPath()), Util.getGET_PERMISSION_COMMAND()));
            // expected format
            // -rw------- 1 username groupname ...
            String permission=t.nextToken();
            if(permission.length()>10){ // files with ACLs might have a '+'
                permission=permission.substring(0, 10);
            }
            setPermission(FsPermission.valueOf(permission));
            t.nextToken();
            setOwner(t.nextToken());
            setGroup(t.nextToken());
        }catch (Shell.ExitCodeException ioe){
            if(ioe.getExitCode()!=1){
                e=ioe;
            }else{
                setPermission(null);
                setOwner(null);
                setGroup(null);
            }
        }catch (IOException ioe){
            e=ioe;
        }finally{
            if(e!=null){
                throw new RuntimeException("Error while running command to get "+"file permissions : "+StringUtils.stringifyException(e));
            }
        }
    }

    @Override
    public void write(DataOutput out) throws IOException{
        if(!isPermissionLoaded()){
            loadPermissionInfo();
        }
        super.write(out);
    }

}
