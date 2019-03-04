package ac.ku.milab.ImplementSTEHIX;

import java.util.ArrayList;
import java.util.Collection;

import org.davidmoten.hilbert.HilbertCurve;
import org.davidmoten.hilbert.HilbertCurveRenderer;
import org.davidmoten.hilbert.SmallHilbertCurve;

import com.github.davidmoten.rtree.Entry;
import com.github.davidmoten.rtree.RTree;
import com.github.davidmoten.rtree.Selector;
import com.github.davidmoten.rtree.geometry.Geometries;
import com.github.davidmoten.rtree.geometry.Geometry;
import com.github.davidmoten.rtree.geometry.Point;

import rx.Observable;

public class App {
	public static void main(String[] args) {
		// SmallHilbertCurve c = HilbertCurve.small().bits(7).dimensions(2);
		// long index = c.index(3,8);
		// System.out.println(index);

		// HilbertCurveRenderer.renderToFile(7, 1000, "abc.jpg");
		
//		String regionName = "region";
//		ServerIndexManager serverManager = new ServerIndexManager();
//		serverManager.addRegionIndexManager(regionName);
//		
//		//long one = serverManager.getHilbertValue(5, 6);
//		//System.out.println(one);
//		RTree<String, Geometry> rtree = RTree.create();
//		rtree = rtree.add("aa", Geometries.rectangle(0, 0, 2, 2));
//		rtree = rtree.add("cc", Geometries.rectangle(1, 0, 2, 3));
//		rtree = rtree.add("bb", Geometries.rectangle(2, 0, 5, 4));
//		rtree = rtree.add("dd", Geometries.rectangle(3, 0, 8, 5));
//		rtree = rtree.add("ab", Geometries.rectangle(1.5, 1, 2, 3));
//		rtree = rtree.add("cd", Geometries.rectangle(3, 0, 7, 2));
//		rtree = rtree.add("ef", Geometries.rectangle(5, 4, 6, 5));
//		
//		System.out.println(rtree.asString());
//		//rtree.visualize(600, 600).save("mytree.png");
//		
//		RTree<String, Geometry> rtree1 = RTree.create();
//		rtree1 = rtree1.add("aa", Geometries.rectangle(0, 0, 2, 2));
//		rtree1 = rtree1.add("cc", Geometries.rectangle(1, 0, 2, 3));
//		System.out.println(rtree1.asString());
//		
//		HilbertCurveManager a = HilbertCurveManager.getInstance();
//		long ab = a.getConvetedHilbertValue(5.8, 2.3);
//		
//		System.out.println(ab);
//		ab = HilbertCurve.small().bits(7).dimensions(2).index(73,29);
//		System.out.println(ab);
//		
//		RegionIndexManager manager = serverManager.getRegionIndexManager(regionName);
//		Observable<Entry<String, Geometry>> ob = rtree.entries();
//		Iterable<Entry<String, Geometry>> b = ob.toBlocking().toIterable();
//		for (Entry<String, Geometry> entry : b) {
//			System.out.println(entry.value());
//			System.out.println(entry.geometry().mbr());
//		}
		
		
//		HilbertCurveManager manager = HilbertCurveManager.getInstance();
//		long hilbert = manager.getConvetedHilbertValue(3.5, 4.2);
//		double[] conv = manager.getConvertedCordinate(hilbert);
//		System.out.println(hilbert);
//		System.out.println(conv[0]+","+conv[1]+","+conv[2]+","+conv[3]);
//		System.out.println((conv[2]-conv[0]) + " " + (conv[3]-conv[1]));
//		
//		long hilbert1 = manager.getConvetedHilbertValue(3.6, 4.1);
//		double[] conv1 = manager.getConvertedCordinate(hilbert1);
//		System.out.println(hilbert1);
//		System.out.println(conv1[0]+","+conv1[1]+","+conv1[2]+","+conv1[3]);
//		
//		System.out.println("overlap " + manager.isOverlapped(conv, conv1));
		
		HilbertCurveManager manager = HilbertCurveManager.getInstance();
		ArrayList<Long> values = manager.rangeToHilbert(new double[]{3.0, 1.8, 3.2, 1.9});
		
		for(long a : values){
			double[] cor = manager.getConvertedCordinate(a);
			System.out.println(a+":"+manager.toString(cor));
		}
				
	}
}
