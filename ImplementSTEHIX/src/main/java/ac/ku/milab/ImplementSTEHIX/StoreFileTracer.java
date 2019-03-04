package ac.ku.milab.ImplementSTEHIX;

import java.util.HashSet;
import java.util.Iterator;
import java.util.NavigableMap;
import java.util.Set;
import java.util.TreeMap;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

public class StoreFileTracer {
  private static final Log LOG = LogFactory.getLog(StoreFileTracer.class.getName());
  
  // region name
  private String regionId;
  
  // trace map
  private NavigableMap<String,StoreFileTrace> storeFileTraceMap;
  // revision trace
  private NavigableMap<String, String> revisedStoreFileRecordMap;
  
  // Constructor
  public StoreFileTracer(String regionId){
    this.regionId = regionId;
    
    this.storeFileTraceMap = new TreeMap<String,StoreFileTrace>();
    this.revisedStoreFileRecordMap = new TreeMap<String, String>();
  }
  
  // add StoreFileTrace
  public void addStoreFileTrace(String storeFileName, int num){
    if(!isExistInTracer(storeFileName)){
      StoreFileTrace trace = new StoreFileTrace(storeFileName, num);
      storeFileTraceMap.put(storeFileName, trace);
      //LOG.info("this StoreFileTrace is registered");
    }else{
      //LOG.info("this StoreFileTrace is already registered");
    }
  }
  
  //register revision record
  public void addRevisedRecord(String storeFileName, String revisedName){
    // register revision record
    revisedStoreFileRecordMap.put(storeFileName, revisedName);
    //LOG.info("revisedRecord from " + storeFileName + " to " + revisedName + " is registered");
    
    // if old revision record exist, directly connect it
    if(revisedStoreFileRecordMap.containsValue(storeFileName)){
      Iterator<String> iter = revisedStoreFileRecordMap.keySet().iterator();
      while(iter.hasNext()){
        String key = iter.next();
        String value = revisedStoreFileRecordMap.get(key);
        if(value.equals(storeFileName)){
          revisedStoreFileRecordMap.replace(key, revisedName);
          //LOG.info("revisedRecord of " + key + " is changed from " + storeFileName + " to " + storeFileName);
        }
      }
    }
  }
  
  // whether trace is already registered
  public boolean isExistInTracer(String storeFileName){
    boolean isExist = storeFileTraceMap.containsKey(storeFileName);
    //LOG.info("isExistInTracer " + storeFileName + " " + isExist);
    return isExist;
  }
  
  // whether StoreFile happened to compact
  public boolean isRevised(String storeFileName){
    boolean isRevised = revisedStoreFileRecordMap.containsKey(storeFileName);
    //LOG.info("isRevised " + storeFileName + " " + isRevised);
    return isRevised;
  }
  
  public StoreFileTrace getStoreFileTrace(String storeFileName){
    if(isExistInTracer(storeFileName)){
      return storeFileTraceMap.get(storeFileName);
    }
    return null;
  }
  
  public void deleteStoreFileTrace(String storeFileName){
    if(isExistInTracer(storeFileName)){
      storeFileTraceMap.remove(storeFileName);
      //LOG.info("deleteStoreFileTrace " + storeFileName + " is deleted");
    }
  }
  
  public void deleteRevisedRecord(String storeFileName){
    if(isRevised(storeFileName)){
      revisedStoreFileRecordMap.remove(storeFileName);
      //LOG.info("deleteRevisedRecord " + storeFileName + " is deleted");
    }
  }
  
  public String getRegionId(){
    return this.regionId;
  }
  
  // get current StoreFile name of target StoreFile
  public String getCurrentStoreFileName(String targetName){
    if(isExistInTracer(targetName)){
      String currentName;
      if(isRevised(targetName)){
        currentName = revisedStoreFileRecordMap.get(targetName);
      }else{
        currentName = targetName;
      }
      return currentName;
    }else{
      return null;
    }
  }
  
  public Set<String> getCurrentStoreFileName(Set<String> targetNameList){
    Set<String> resultSet = new HashSet<String>();
    for(String storeFileName : targetNameList){
      String currentName = getCurrentStoreFileName(storeFileName);
      resultSet.add(currentName);
    }
    return resultSet;
  }
  
  public void decrementNumber(String storeFileName){
    if(isExistInTracer(storeFileName)){
      StoreFileTrace trace = storeFileTraceMap.get(storeFileName);
      boolean isZero = trace.decrementNumber();
      //LOG.info("decrementNumber " + storeFileName + " is decremented and " + isZero);
      
      if(isZero){
        deleteStoreFileTrace(storeFileName);
        deleteRevisedRecord(storeFileName);
        //LOG.info("decrementNumber " + storeFileName + " is deleted");
      }
    }
  }
  
  public void decrementNumber(String storeFileName, int num){
    if(isExistInTracer(storeFileName)){
      StoreFileTrace trace = storeFileTraceMap.get(storeFileName);
      boolean isZero = trace.decrementNumber(num);
      //LOG.info("decrementNumber " + storeFileName + " is decremented and " + isZero);
      
      if(isZero){
        deleteStoreFileTrace(storeFileName);
        deleteRevisedRecord(storeFileName);
        //LOG.info("decrementNumber " + storeFileName + " is deleted");
      }
    }
  }
  
  public void printStoreFileTraceMap(){
    Set<String> set = this.storeFileTraceMap.keySet();
    for(String storeFileName : set){
      StoreFileTrace trace = storeFileTraceMap.get(storeFileName);
      //LOG.info("storeFileName : " + storeFileName + ", number : " + trace.getNumData());
    }
  }
  
  public void printRevisedRecord(){
    Set<String> set = this.revisedStoreFileRecordMap.keySet();
    for(String storeFileName : set){
      String newName = revisedStoreFileRecordMap.get(storeFileName);
      //LOG.info("storeFileName : " + storeFileName + ", newName : " + newName);
    }
  }
  
  public void printStoreFileTracer(){
    printRevisedRecord();
   printStoreFileTraceMap();
  }
}
