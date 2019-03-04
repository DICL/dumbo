package ac.ku.milab.ImplementSTEHIX.regionserver;

import java.io.IOException;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.NavigableSet;
import java.util.Set;
import java.util.TreeMap;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.hbase.Cell;
import org.apache.hadoop.hbase.CellUtil;
import org.apache.hadoop.hbase.CoprocessorEnvironment;
import org.apache.hadoop.hbase.HRegionInfo;
import org.apache.hadoop.hbase.KeyValue;
import org.apache.hadoop.hbase.Server;
import org.apache.hadoop.hbase.TableName;
import org.apache.hadoop.hbase.client.Durability;
import org.apache.hadoop.hbase.client.Put;
import org.apache.hadoop.hbase.client.Scan;
import org.apache.hadoop.hbase.coprocessor.BaseRegionObserver;
import org.apache.hadoop.hbase.coprocessor.ObserverContext;
import org.apache.hadoop.hbase.coprocessor.RegionCoprocessorEnvironment;
import org.apache.hadoop.hbase.filter.Filter;
import org.apache.hadoop.hbase.io.hfile.HFile;
import org.apache.hadoop.hbase.io.hfile.HFileScanner;
import org.apache.hadoop.hbase.regionserver.HRegion;
import org.apache.hadoop.hbase.regionserver.InternalScan;
import org.apache.hadoop.hbase.regionserver.InternalScanner;
import org.apache.hadoop.hbase.regionserver.KeyValueScanner;
import org.apache.hadoop.hbase.regionserver.Region;
import org.apache.hadoop.hbase.regionserver.Region.Operation;
import org.apache.hadoop.hbase.regionserver.SplitTransaction.SplitTransactionPhase;
import org.apache.hadoop.hbase.regionserver.RegionScanner;
import org.apache.hadoop.hbase.regionserver.RegionServerServices;
import org.apache.hadoop.hbase.regionserver.ScanInfo;
import org.apache.hadoop.hbase.regionserver.SplitTransaction;
import org.apache.hadoop.hbase.regionserver.SplitTransactionFactory;
import org.apache.hadoop.hbase.regionserver.SplitTransactionImpl;
import org.apache.hadoop.hbase.regionserver.Store;
import org.apache.hadoop.hbase.regionserver.StoreFile;
import org.apache.hadoop.hbase.regionserver.StoreFileInfo;
import org.apache.hadoop.hbase.regionserver.StoreFileScanner;
import org.apache.hadoop.hbase.regionserver.wal.WALEdit;
import org.apache.hadoop.hbase.util.Bytes;
import org.apache.hadoop.hbase.util.Pair;
import org.apache.hadoop.hbase.util.RegionSplitter;
import org.apache.hadoop.hbase.util.RegionSplitter.HexStringSplit;

import com.github.davidmoten.rtree.Entry;
import com.github.davidmoten.rtree.RTree;
import com.github.davidmoten.rtree.geometry.Geometries;
import com.github.davidmoten.rtree.geometry.Geometry;
import com.github.davidmoten.rtree.geometry.Point;
import com.github.davidmoten.rtree.geometry.Rectangle;
import com.google.common.collect.ImmutableList;

import ac.ku.milab.ImplementSTEHIX.RegionIndexManager;
import ac.ku.milab.ImplementSTEHIX.SpaceFilter;
import ac.ku.milab.ImplementSTEHIX.TimeFilter;
import ac.ku.milab.ImplementSTEHIX.TimeSpaceFilter;
import ac.ku.milab.ImplementSTEHIX.util.TableUtils;
import rx.Observable;

public class IndexRegionObserver extends BaseRegionObserver {

  private static final Log LOG = LogFactory.getLog(IndexRegionObserver.class.getName());
  
  private static Map<String, RegionIndexManager> regionIndexManagerMap = new TreeMap<String, RegionIndexManager>();
  private List<String> compactCandidates = null;

  private static final byte[] COLUMN_FAMILY = Bytes.toBytes("cf1");
  
  private static Map<String, Long> flushTimeCheckMap = new TreeMap<String, Long>();

  @Override
  public void start(CoprocessorEnvironment e) throws IOException {
    // TODO Auto-generated method stub
    LOG.info("START Coprocessor");
    super.start(e);
  }

  @Override
  public void postOpen(ObserverContext<RegionCoprocessorEnvironment> ctx) {
    // TODO Auto-generated method stub
    HRegionInfo regionInfo = ctx.getEnvironment().getRegionInfo();
    String tableName = regionInfo.getTable().getNameAsString();

    LOG.info("PostOpen : " + tableName);

    boolean isSystemTable = TableUtils.isSystemTable(tableName);
    if (!isSystemTable) {
      String regionName = regionInfo.getRegionNameAsString();
      if(!regionIndexManagerMap.containsKey(regionName)){
        RegionIndexManager regionIndexManager = new RegionIndexManager(regionName);
        regionIndexManagerMap.put(regionName, regionIndexManager);
        
        Region region = ctx.getEnvironment().getRegion();
        Store store = region.getStore(COLUMN_FAMILY);

        if (store == null) {
          LOG.info("PostOpen store is null");
          return;
        }

        Collection<StoreFile> storeFiles = store.getStorefiles();

        if (storeFiles == null) {
          LOG.info("PostOpen storeFiles is null");
          return;
        }
        Iterator<StoreFile> iter = storeFiles.iterator();
        while (iter.hasNext()) {
          LOG.info(ctx.getEnvironment().getRegionInfo().getRegionNameAsString() + " index ready");
          StoreFile storeFile = iter.next();
          regionIndexManager.addDatas(storeFile);
        }
        LOG.info("Index is ready");
      }
      
    }
  }

  // before put implements, call this function
  @Override
  public void prePut(ObserverContext<RegionCoprocessorEnvironment> ctx, Put put, WALEdit edit,
      Durability durability) throws IOException {

    // get table's information
    TableName tName = ctx.getEnvironment().getRegionInfo().getTable();
    String tableName = tName.getNameAsString();

    //LOG.info("PrePut START : " + tableName);

    // if table is not user table, it is not performed
    boolean isSystemTable = TableUtils.isSystemTable(Bytes.toBytes(tableName));
    if (!isSystemTable) {
      if (Bytes.equals(put.getRow(), Bytes.toBytes("flush"))) {
        Region r = ctx.getEnvironment().getRegion();
        r.flush(false);
        ctx.bypass();
        ctx.complete();
      }
      
      if (Bytes.equals(put.getRow(), Bytes.toBytes("compact"))) {
        Region r = ctx.getEnvironment().getRegion();
        r.compact(false);
        ctx.bypass();
        ctx.complete();
      }
      
      if(Bytes.equals(put.getRow(), Bytes.toBytes("split"))) {
        Region r = ctx.getEnvironment().getRegion();
        RegionServerServices rss = ctx.getEnvironment().getRegionServerServices();
        
        byte[] splitKey = getSplitKey(r);
        //LOG.info("splitKey is "+splitKey);
        //SplitTransactionFactory factory = new SplitTransactionFactory(ctx.getEnvironment().getConfiguration());
        //SplitTransaction splitTransaction = factory.create(r, splitKey);
        SplitTransaction splitTransaction = new SplitTransactionImpl(r, splitKey);
        splitTransaction.registerTransactionListener(new SplitTransaction.TransactionListener() {
          
          public void transition(SplitTransaction transaction, SplitTransactionPhase from,
              SplitTransactionPhase to) throws IOException {
            // TODO Auto-generated method stub
            LOG.info("split transition");
            LOG.info(from.name()+" to " + to.name());
          }
          
          public void rollback(SplitTransaction transaction, SplitTransactionPhase from,
              SplitTransactionPhase to) {
            // TODO Auto-generated method stub
            LOG.info("split rollback");
          }
        });
        ctx.bypass();
        ctx.complete();
        r.startRegionOperation(Operation.SPLIT_REGION);
        if(splitTransaction.prepare()){
          try{
            LOG.info("Split is ready");
            splitTransaction.execute((Server)rss, rss);
            LOG.info("Split is done");
          }catch(Exception e){
            e.printStackTrace();
            try{
              splitTransaction.rollback((Server)rss, rss);
            }catch(Exception e1){
              e1.printStackTrace();
            }
          }
          
        }
        r.closeRegionOperation();
      }
    }
  }

  // after flushing, register information in index manager
  @Override
  public void postFlush(ObserverContext<RegionCoprocessorEnvironment> ctx, Store store,
      StoreFile resultFile) throws IOException {

    HRegionInfo regionInfo = ctx.getEnvironment().getRegionInfo();
    String regionName = regionInfo.getRegionNameAsString();
    String tableName = regionInfo.getTable().getNameAsString();

    boolean isSystemTable = TableUtils.isSystemTable(tableName);

    if (!isSystemTable) {
      LOG.info("PostFlush : " + tableName);

      String storeFileName = resultFile.getFileInfo().getPath().getName();
      LOG.info("storefilename is " + storeFileName);

      RegionIndexManager regionIndexManager = null;
      if(!regionIndexManagerMap.containsKey(regionName)){
        regionIndexManager = new RegionIndexManager(regionName);
        regionIndexManagerMap.put(regionName, regionIndexManager);
      }else{
        regionIndexManager = regionIndexManagerMap.get(regionName);
      }
      regionIndexManager.addDatas(resultFile);
      
      long curTime = System.currentTimeMillis();
      long preTime = flushTimeCheckMap.get(regionName);
      
      long timeDiff = curTime - preTime;
      flushTimeCheckMap.remove(regionName);
      LOG.info("region "+regionName + " flush time : "+timeDiff + "ms");
    }
  }

  @Override
  public KeyValueScanner preStoreScannerOpen(ObserverContext<RegionCoprocessorEnvironment> ctx,
      Store store, Scan scan, NavigableSet<byte[]> targetCols, KeyValueScanner s)
      throws IOException {
    TableName tName = ctx.getEnvironment().getRegionInfo().getTable();
    String tableName = tName.getNameAsString();

    //LOG.info("preStoreScannerOpen START : " + tableName);

    // if table is not user table, it is not performed
    boolean isSystemTable = TableUtils.isSystemTable(Bytes.toBytes(tableName));
    if (!isSystemTable) {
      if (scan instanceof InternalScan) {

      }

      if (scan.getFilter() != null) {
        LOG.info("this scan has filter");
      }
    }
    return s;
  }
  
  @Override
  public void postSplit(ObserverContext<RegionCoprocessorEnvironment> ctx, Region l, Region r)
      throws IOException {
    // TODO Auto-generated method stub
    
    TableName tableName = ctx.getEnvironment().getRegionInfo().getTable();
    String sTableName = tableName.getNameAsString();
    
    LOG.info("postSplit START : " + sTableName);
    
    boolean isSystemTable = TableUtils.isSystemTable(Bytes.toBytes(sTableName));

    if (!isSystemTable) {
       String regionName = ctx.getEnvironment().getRegionInfo().getRegionNameAsString();
       regionIndexManagerMap.remove(regionName);
    }
    
    super.postSplit(ctx, l, r);
  }

  @Override
  public void preSplit(ObserverContext<RegionCoprocessorEnvironment> ctx) throws IOException {
    // TODO Auto-generated method stub
    TableName tableName = ctx.getEnvironment().getRegionInfo().getTable();
    String sTableName = tableName.getNameAsString();
    
    LOG.info("preSplit START : " + sTableName);

    boolean isSystemTable = TableUtils.isSystemTable(Bytes.toBytes(sTableName));

    if (!isSystemTable) {

    }
    super.preSplit(ctx);
  }

  // before regionscanner open, if index scanner is needed, then return index scanner
  @Override
  public RegionScanner preScannerOpen(ObserverContext<RegionCoprocessorEnvironment> ctx, Scan scan,
      RegionScanner s) throws IOException {
    TableName tName = ctx.getEnvironment().getRegionInfo().getTable();
    String tableName = tName.getNameAsString();

    LOG.info("preScannerOpen START : " + tableName);

    // if table is not user table, it is not performed
    boolean isSystemTable = TableUtils.isSystemTable(Bytes.toBytes(tableName));
    if (!isSystemTable) {
      Filter f = scan.getFilter();

      if (f == null) {
        return super.preScannerOpen(ctx, scan, s);
      }
      //LOG.info("scan has filter");
      if (f instanceof SpaceFilter) {
        SpaceFilter idxFilter = (SpaceFilter) f;
        double x1 = idxFilter.getX1();
        double y1 = idxFilter.getY1();
        double x2 = idxFilter.getX2();
        double y2 = idxFilter.getY2();
        double[] queryRect = new double[] { x1, y1, x2, y2 };

        String regionName = ctx.getEnvironment().getRegionInfo().getRegionNameAsString();
        HRegion region = (HRegion) ctx.getEnvironment().getRegion();
        
        RegionIndexManager regionIndexManager = regionIndexManagerMap.get(regionName);

        RegionScanner regionScanner =
            regionIndexManager.getIndexRegionScanner(region, scan, queryRect);
        ctx.bypass();
        return regionScanner;
      } 
      
      else if(f instanceof TimeFilter){
        LOG.info("TimeFilter input");
        return super.preScannerOpen(ctx, scan, s);
      } 
      
      else if(f instanceof TimeSpaceFilter){
        LOG.info("TimeSpaceFilter input");
        TimeSpaceFilter idxFilter = (TimeSpaceFilter) f;
        double x1 = idxFilter.getX1();
        double y1 = idxFilter.getY1();
        double x2 = idxFilter.getX2();
        double y2 = idxFilter.getY2();
        double[] queryRect = new double[] { x1, y1, x2, y2 };

        String regionName = ctx.getEnvironment().getRegionInfo().getRegionNameAsString();
        HRegion region = (HRegion) ctx.getEnvironment().getRegion();
        
        RegionIndexManager regionIndexManager = regionIndexManagerMap.get(regionName);

        RegionScanner regionScanner =
            regionIndexManager.getIndexRegionScanner(region, scan, queryRect);
        ctx.bypass();
        return regionScanner;
      }
      else{
        return super.preScannerOpen(ctx, scan, s);
      }
    }
    return super.preScannerOpen(ctx, scan, s);
  }

  @Override
  public void preCompactSelection(ObserverContext<RegionCoprocessorEnvironment> ctx, Store store,
      List<StoreFile> candidates) throws IOException {
    // TODO Auto-generated method stub

    TableName tName = ctx.getEnvironment().getRegionInfo().getTable();
    String tableName = tName.getNameAsString();

    LOG.info("preCompactSelection START : " + tableName);

    boolean isSystemTable = TableUtils.isSystemTable(Bytes.toBytes(tableName));
    if (!isSystemTable) {
      if (compactCandidates == null) {
        compactCandidates = new ArrayList<String>();
      }

      StringBuilder builder = new StringBuilder();
      builder.append("candidates : ");
      for (StoreFile storeFile : candidates) {
        String storeFileName = storeFile.getFileInfo().getPath().getName();
        compactCandidates.add(storeFileName);
        builder.append(storeFileName);
        builder.append(",");
      }
      LOG.info(builder.toString());
    }
  }
  
  @Override
  public InternalScanner preFlush(ObserverContext<RegionCoprocessorEnvironment> ctx, Store store,
      InternalScanner scanner) throws IOException {
    // TODO Auto-generated method stub
    
    HRegionInfo regionInfo = ctx.getEnvironment().getRegionInfo();
    String regionName = regionInfo.getRegionNameAsString();
    String tableName = regionInfo.getTable().getNameAsString();

    boolean isSystemTable = TableUtils.isSystemTable(tableName);

    if (!isSystemTable) {
      flushTimeCheckMap.put(regionName, System.currentTimeMillis());
    }
    return super.preFlush(ctx, store, scanner);
  }
 

  @Override
  public void postCompact(ObserverContext<RegionCoprocessorEnvironment> ctx, Store store,
      StoreFile resultFile) throws IOException {
    // TODO Auto-generated method stub
    TableName tName = ctx.getEnvironment().getRegionInfo().getTable();
    String tableName = tName.getNameAsString();

    // e.getEnvironment().getRegion().getStore(Bytes.toBytes("cf1")).getStorefiles();
    LOG.info("postCompact START : " + tableName);

    boolean isSystemTable = TableUtils.isSystemTable(Bytes.toBytes(tableName));
    if (!isSystemTable) {
      String regionName = ctx.getEnvironment().getRegionInfo().getRegionNameAsString();
      String resultFileName = resultFile.getFileInfo().getPath().getName();

      LOG.info("postCompact storefile - " + resultFileName);

      if (compactCandidates != null) {
        RegionIndexManager regionIndexManager = regionIndexManagerMap.get(regionName);
        
        int totalNum = 0;
        for (String name : compactCandidates) {
          regionIndexManager.addRevisedStoreFile(name, resultFileName);
          int num = regionIndexManager.getNumberInStoreFileTrace(name);
          totalNum += num;
        }
        compactCandidates.clear();
        compactCandidates = null;
        regionIndexManager.addNewStoreFileTrace(resultFileName, totalNum);
      }
    }
  }

  public byte[] getSplitKey(Region r) {
    Store store = r.getStore(COLUMN_FAMILY);
    List<Pair<byte[], Long>> samples = new ArrayList<Pair<byte[], Long>>();

    for (StoreFile sf : store.getStorefiles()) {
      try {
        byte[] midkey = sf.createReader().midkey();
        KeyValue kv = KeyValue.createKeyValueFromKey(midkey);
        byte[] rowkey = kv.getRow();
        long size = sf.createReader().length();
        samples.add(Pair.newPair(rowkey, size));
      } catch (IOException e) {
        // Empty StoreFile, or problem reading it
        LOG.error("Encountered problem reading store file: " + sf.toString());
        break;
      }
    }

    LOG.info("Combining " + samples.size() + " samples");

    long total_weight = 0;
    int max_sample_len = 0;

    // Find the max sample array length, and sum the sample weights
    for (Pair<byte[], Long> s : samples) {
      byte[] sample = s.getFirst();
      long weight = s.getSecond();

      max_sample_len = Math.max(max_sample_len, sample.length);
      total_weight += weight;
    }

    BigInteger weighted_samples_sum = BigInteger.ZERO;

    for (Pair<byte[], Long> s : samples) {
      byte[] sample = s.getFirst();
      long size = s.getSecond();

      byte[] normalized_sample = Bytes.padTail(sample, max_sample_len - sample.length);
      BigInteger sample_val = new BigInteger(1, normalized_sample);

      weighted_samples_sum =
          weighted_samples_sum.add(sample_val.multiply(BigInteger.valueOf(size)));
    }

    BigInteger combined_val = weighted_samples_sum.divide(BigInteger.valueOf(total_weight));
    byte[] combined = combined_val.toByteArray();

    // If the leading byte is 0, it came from BigInteger adding a byte to
    // indicate a positive two's complement value. Strip it.
    if (combined[0] == 0x00) {
      combined = Bytes.tail(combined, 1);
    }
    
//    byte[] longPart = new byte[8];
//    System.arraycopy(combined, combined.length-8, longPart, 0, 8);
//
//    String number = Bytes.toString(combined, 0, combined.length-8);
//    LOG.info("Combined value: " + number + ","+Bytes.toLong(longPart));
    LOG.info("Combined value: " + Bytes.toString(combined));
    return combined;
  }
}
