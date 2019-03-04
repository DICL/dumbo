package ac.ku.milab.ImplementSTEHIX;

import java.util.ArrayList;

import org.davidmoten.hilbert.HilbertCurve;
import org.davidmoten.hilbert.SmallHilbertCurve;

public class HilbertCurveManager {

	private static HilbertCurveManager manager = null;
	private static SmallHilbertCurve hilbertCurve = null;

	private static final double MIN_X = 0.0;
	private static final double MAX_X = 10.0;
	private static final double MIN_Y = 0.0;
	private static final double MAX_Y = 10.0;

	private static final double MAX_CONVERTED_VALUE = 127.0;
	private static final double MIN_CONVERTED_VALUE = 0.0;
	
	private static final double X_TICK = (1.0 - MIN_CONVERTED_VALUE) * (MAX_X - MIN_X) / (MAX_CONVERTED_VALUE - MIN_CONVERTED_VALUE);
	private static final double Y_TICK = (1.0 - MIN_CONVERTED_VALUE) * (MAX_X - MIN_X) / (MAX_CONVERTED_VALUE - MIN_CONVERTED_VALUE);

	private HilbertCurveManager() {
		hilbertCurve = HilbertCurve.small().bits(7).dimensions(2);
	}

	public static HilbertCurveManager getInstance() {
		if (manager == null) {
			manager = new HilbertCurveManager();
		}
		return manager;
	}

	// get hilbert value of x, y
	public long getConvetedHilbertValue(double x, double y) {
		double convertedX = MIN_CONVERTED_VALUE
				+ (x - MIN_X) * (MAX_CONVERTED_VALUE - MIN_CONVERTED_VALUE) / (MAX_X - MIN_X);
		double convertedY = MIN_CONVERTED_VALUE
				+ (y - MIN_Y) * (MAX_CONVERTED_VALUE - MIN_CONVERTED_VALUE) / (MAX_Y - MIN_Y);

		return hilbertCurve.index((long) convertedX, (long) convertedY);
	}

	// get original cordinate
	public double[] getConvertedCordinate(long convertedValue) {

		long[] cor = hilbertCurve.point(convertedValue);

		long x1 = cor[0];
		long y1 = cor[1];

		double x_1 = MIN_X + (x1 - MIN_CONVERTED_VALUE) * (MAX_X - MIN_X) / (MAX_CONVERTED_VALUE - MIN_CONVERTED_VALUE);
		double x_2 = x_1 + X_TICK;
		double y_1 = MIN_Y + (y1 - MIN_CONVERTED_VALUE) * (MAX_Y - MIN_Y) / (MAX_CONVERTED_VALUE - MIN_CONVERTED_VALUE);
		double y_2 = y_1 + Y_TICK;
		
		return new double[] { x_1, y_1, x_2, y_2 };
	}
	
	// whether or not both are overlapped
	public boolean isOverlapped(double[] r1, double[] r2){
		if(r2[0]>r1[2] || r2[2]<r1[0] || r2[1]>r1[3] || r2[3]<r1[1]){
			return false;
		}else{
			return true;
		}
	}
	
	// to express cordinate
	public String toString(double[] cor){
		StringBuilder sb = new StringBuilder();
		sb.append("(").append(cor[0]).append(",").append(cor[1]).append(")-(").append(cor[2]).append(",").append(cor[3]).append(")");
		return sb.toString();
	}
	
	// get hilbert value between two cordinates
	public ArrayList<Long> rangeToHilbert(double[] cor){
		ArrayList<Long> values = new ArrayList<Long>();
		
		double startX = cor[0];
		double startY = cor[1];
		
		double endX = cor[2];
		double endY = cor[3];
		
		double x=startX;
		double y=startY;
		for(;x<endX;x=x+X_TICK){
			for(;y<endY;y=y+Y_TICK){
				values.add(getConvetedHilbertValue(x, y));
			}
			if(getConvetedHilbertValue(x, y)==getConvetedHilbertValue(x, endY)){
				values.add(getConvetedHilbertValue(x, y));
			}
			y=startY;
		}
		if(getConvetedHilbertValue(x, y)==getConvetedHilbertValue(endX, y)){
			for(;y<endY;y=y+Y_TICK){
				values.add(getConvetedHilbertValue(x, y));
			}
			if(getConvetedHilbertValue(x, y)==getConvetedHilbertValue(x, endY)){
				values.add(getConvetedHilbertValue(x, y));
			}
		}
		
		return values;
		
	}
}
