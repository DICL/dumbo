package ac.ku.milab.ImplementSTEHIX;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.NavigableMap;
import java.util.Set;
import java.util.TreeMap;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.hbase.Cell;
import org.apache.hadoop.hbase.CellUtil;
import org.apache.hadoop.hbase.HRegionInfo;
import org.apache.hadoop.hbase.client.Scan;
import org.apache.hadoop.hbase.io.hfile.HFileScanner;
import org.apache.hadoop.hbase.regionserver.HRegion;
import org.apache.hadoop.hbase.regionserver.KeyValueScanner;
import org.apache.hadoop.hbase.regionserver.RegionScanner;
import org.apache.hadoop.hbase.regionserver.StoreFile;
import org.apache.hadoop.hbase.util.Bytes;
import org.davidmoten.hilbert.SmallHilbertCurve;

import com.github.davidmoten.rtree.Entry;
import com.github.davidmoten.rtree.RTree;
import com.github.davidmoten.rtree.geometry.Geometries;
import com.github.davidmoten.rtree.geometry.Geometry;
import com.github.davidmoten.rtree.geometry.Point;
import com.github.davidmoten.rtree.geometry.Rectangle;
import com.google.common.collect.ImmutableList;

import ac.ku.milab.ImplementSTEHIX.util.TableUtils;
import rx.Observable;

// Region Index Manager
public class RegionIndexManager {

  private static final Log LOG = LogFactory.getLog(RegionIndexManager.class.getName());

  // Region Name
  private String regionId;

  // Index using hilbert value and rtree
  private NavigableMap<Long, RTree<String, Geometry>> hilbertRTree;
  // StoreFileTracer
  private StoreFileTracer storeFileTracer;

  // Constructor
  public RegionIndexManager(String regionId) {
    this.regionId = regionId;
    this.storeFileTracer = new StoreFileTracer(regionId);
    this.hilbertRTree = new TreeMap<Long, RTree<String, Geometry>>();
  }

  // get regionId
  public String getRegionId() {
    return this.regionId;
  }

  // add rtree in specific hilbert value
  public void addRTree(long key, RTree<String, Geometry> rTree) {
    if (this.hilbertRTree.containsKey(key)) {
      //LOG.info("key " + key + " was already registered");
      return;
    } else {
      //LOG.info("key " + key + " registers index manager");
      this.hilbertRTree.put(key, rTree);
      System.out.println("put");
    }
  }

  // update to new rtree
  public void updateRTree(long key, RTree<String, Geometry> rTree) {
    if (isExist(key)) {
      this.hilbertRTree.replace(key, rTree);
    } else {
      return;
    }
  }

  // add new rtree
  public void addRTree(long key) {
    if (this.hilbertRTree.containsKey(key)) {
      //LOG.info("key " + key + " was already registered");
      return;
    } else {
      //LOG.info("key " + key + " registers index manager");
      RTree<String, Geometry> rTree = RTree.create();
      this.hilbertRTree.put(key, rTree);
      //System.out.println("put");
    }
  }

  // get rtree of hilbert value
  public RTree<String, Geometry> getRTree(long key) {
    if (isExist(key)) {
      //LOG.info("key " + key + " RTree");
      return this.hilbertRTree.get(key);
    } else {
      return null;
    }
  }

  // whether rtree of key already exists
  public boolean isExist(long key) {
    if (this.hilbertRTree.containsKey(key)) {
      return true;
    } else {
      return false;
    }
  }

  // whether trace is already registered
  public boolean isExistInTracer(String storeFileName) {
    return this.storeFileTracer.isExistInTracer(storeFileName);
  }

  // get current file of target storefile
  public String getCurrentFileName(String targetName) {
    return this.storeFileTracer.getCurrentStoreFileName(targetName);
  }

  // add revisedRecord
  public void addRevisedStoreFile(String name, String currentName) {
    this.storeFileTracer.addRevisedRecord(name, currentName);
    //LOG.info(name + " is currently " + currentName);
  }
  
 // add revisedRecords
  public void addRevisedStoreFiles(ImmutableList<StoreFile> selected, String currentName){
    for(StoreFile storeFile : selected){
      String storeFileName = storeFile.getFileInfo().getPath().getName();
      addRevisedStoreFile(storeFileName, currentName);
    }
  }

  // add one point
  public void addData(long key, Point p, String storeFileName) {

    RTree<String, Geometry> rTree;
    if (isExist(key)) {
      rTree = getRTree(key);
    } else {
      addRTree(key);
      rTree = getRTree(key);
    }

    rTree = rTree.add(storeFileName, p);
    updateRTree(key, rTree);
    //LOG.info("rtree add (" + p.x() + "," + p.y() + ")");
  }

  // add all data in storeFile
  public void addDatas(StoreFile storeFile) {
    boolean hasNext = true;
    double lat = 0.0;
    double lon = 0.0;
    int count = 0;

    String storeFileName = storeFile.getFileInfo().getPath().getName();

    HilbertCurveManager hilbertManager = HilbertCurveManager.getInstance();

    try {
      HFileScanner scanner = storeFile.createReader().getHFileReader().getScanner(false, false);
      scanner.seekTo();
      
      byte[] bLon = Bytes.toBytes("lon");
      byte[] bLat = Bytes.toBytes("lat");
      
      while (hasNext) {
        Cell cell = scanner.getKeyValue();
        if (cell == null) {
          //LOG.info("cell is null");
          break;
        }
        byte[] qualifier = CellUtil.cloneQualifier(cell);
        // if (latORlon) {
        if (Bytes.equals(qualifier, bLat)) {
          lat = Bytes.toDouble(CellUtil.cloneValue(cell));
          // latORlon = false;
        } else if (Bytes.equals(qualifier, bLon)) {
          // latORlon = true;
          lon = Bytes.toDouble(CellUtil.cloneValue(cell));
          long hilbertValue = hilbertManager.getConvetedHilbertValue(lon, lat);

          RTree<String, Geometry> rTree;
          if (isExist(hilbertValue)) {
            rTree = getRTree(hilbertValue);
          } else {
            addRTree(hilbertValue);
            rTree = getRTree(hilbertValue);
          }

          Point p = Geometries.point(lon, lat);
          rTree = rTree.add(storeFileName, p);
          updateRTree(hilbertValue, rTree);
          //LOG.info("rtree add (" + lon + "," + lat + ")");
          count++;
        }
        hasNext = scanner.next();
      }
      storeFileTracer.addStoreFileTrace(storeFileName, count);
      //LOG.info("storefile " + storeFileName + " has the number of data : " + count);
    } catch (Exception e) {

    }
  }

  // get regionscanner of targets
  public RegionScanner getIndexRegionScanner(HRegion region, Scan scan, double[] queryRect) {

    LOG.info("query range : " + queryRect[0] + "," + queryRect[1] + "," + queryRect[2] + "," + queryRect[3]);

    HilbertCurveManager hilbertManager = HilbertCurveManager.getInstance();
    
    ArrayList<Long> hilbertValues = hilbertManager.rangeToHilbert(queryRect);

    Set<String> storeFileNameSet = new HashSet<String>();

    int count = 0;
    for (long value : hilbertValues) {
      double[] cor = hilbertManager.getConvertedCordinate(value);

      if (!isExist(value)) {
        //LOG.info("no R-tree : " + hilbertManager.toString(cor));
        continue;
      }

      //LOG.info("search : " + hilbertManager.toString(cor));
      RTree<String, Geometry> rTree = getRTree(value);
      Rectangle r = Geometries.rectangle(queryRect[0], queryRect[1], queryRect[2], queryRect[3]);
      Observable<Entry<String, Geometry>> ob = rTree.search(r);
      Iterable<Entry<String, Geometry>> iter = ob.toBlocking().toIterable();

      for (Entry<String, Geometry> entry : iter) {
        String storeFileName = entry.value();
        //LOG.info("entry : " + entry.geometry().mbr());
        count++;
        storeFileNameSet.add(storeFileName);
        //LOG.info("storeFile " + storeFileName + " added in list " + count);
        if(storeFileTracer.isRevised(storeFileName)){
          String newName = storeFileTracer.getCurrentStoreFileName(storeFileName);
          Geometry geo = entry.geometry();
          rTree = rTree.delete(entry).add(newName, geo);
          storeFileTracer.decrementNumber(storeFileName);
        }
      }
      updateRTree(value, rTree);
    }
    LOG.info("added in list " + count);
    
    storeFileNameSet = this.storeFileTracer.getCurrentStoreFileName(storeFileNameSet);
    
    ArrayList<String> lastStoreFileList = new ArrayList<String>();
    Iterator<String> iter = storeFileNameSet.iterator();
    
    while(iter.hasNext()){
      String storeFileName = iter.next();
      lastStoreFileList.add(storeFileName);
      LOG.info("Index scanner includes " + storeFileName);
    }
    //Scan scan = new Scan();
    
    RegionScanner regionScanner = null;
    try{
      regionScanner = region.getSelectedScanner(scan, lastStoreFileList);
    }catch(Exception e){
      
    }
    
    LOG.info("target data : " + count);
    return regionScanner;
  }
  
  public int getNumberInStoreFileTrace(String storeFileName){
    StoreFileTrace trace = storeFileTracer.getStoreFileTrace(storeFileName);
    int number = trace.getNumData();
    return number;
  }
  
  public void addNewStoreFileTrace(String storeFileName, int num){
    this.storeFileTracer.addStoreFileTrace(storeFileName, num);
  }
  
  public void printStoreFileTracer(){
    storeFileTracer.printStoreFileTracer();
  }
  
}
