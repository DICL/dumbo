package ac.ku.milab.ImplementSTEHIX;

import java.io.IOException;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.hbase.Cell;
import org.apache.hadoop.hbase.CellUtil;
import org.apache.hadoop.hbase.exceptions.DeserializationException;
import org.apache.hadoop.hbase.filter.FilterBase;
import org.apache.hadoop.hbase.filter.Filter.ReturnCode;
import org.apache.hadoop.hbase.util.Bytes;

public class TimeFilter extends FilterBase {

  private long startTime = 0;
  private long endTime = 0;

  private boolean filterRow = false;

  private static final Log LOG = LogFactory.getLog(TimeFilter.class.getName());

  public TimeFilter() {
    super();
  }

  public TimeFilter(long startTime, long endTime) {
    this.startTime = startTime;
    this.endTime = endTime;
  }

  public TimeFilter(long startTime, long endTime, boolean filterRow) {
    this.startTime = startTime;
    this.endTime = endTime;
    this.filterRow = filterRow;
  }

  @Override
  public boolean filterRowKey(byte[] buffer, int offset, int length) throws IOException {
    // TODO Auto-generated method stub
    byte[] rowkey = Bytes.copy(buffer, offset, length);

    // LOG.info("buffer is : "+Bytes.toString(rowkey));
    byte[] bTime = Bytes.copy(rowkey, 9, 8);

    long time = Bytes.toLong(bTime);
    //LOG.info("Time Filter " + time);

    if (startTime <= time && endTime >= time) {
      filterRow = false;
      return false;
    } else {
      filterRow = true;
      return true;
    }
  }

  @Override
  public boolean filterRow() {
    // TODO Auto-generated method stub
    return filterRow;
  }

  @Override
  public void reset() {
    // TODO Auto-generated method stub
    this.filterRow = false;
  }

  @Override
  public boolean hasFilterRow() {
    // TODO Auto-generated method stub
    return true;
  }

  public void setFilterRow(boolean filterRow) {
    this.filterRow = filterRow;
  }

  @Override
  public byte[] toByteArray() {

    byte[] array = new byte[0];
    array = Bytes.add(array, Bytes.toBytes(this.startTime));
    array = Bytes.add(array, Bytes.toBytes(this.endTime));
    array = Bytes.add(array, Bytes.toBytes(this.filterRow));
    return array;
  }

  public static TimeFilter parseFrom(byte[] bytes) throws DeserializationException {
    TimeFilter filter = null;
    int length = bytes.length;

    long startTime = Bytes.toLong(Bytes.copy(bytes, 0, 8));
    long endTime = Bytes.toLong(Bytes.copy(bytes, 8, 8));

    boolean filterRow = Bytes.toBoolean(Bytes.copy(bytes, 16, 1));

    filter = new TimeFilter(startTime, endTime, filterRow);

    return filter;
  }

  public double getStartTime() {
    return this.startTime;
  }

  public double getEndTime() {
    return this.endTime;
  }

  @Override
  public ReturnCode filterKeyValue(Cell v) throws IOException {
    // TODO Auto-generated method stub
    return ReturnCode.INCLUDE;
  }
}
