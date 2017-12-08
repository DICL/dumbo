/*
 * JITPTX Project.
 * 
 * @author Byeongcheol Lee
 */
package org.ptx;

import java.io.FileOutputStream;
import static org.ptx.Util.*;

public class PTXJIT {
	public static int verbose = 0;
	static Util a;
	static org.ptx.ir.PIR b;
	static org.ptx.dispatch.Dispatcher c;
	static org.ptx.dispatch.LIR2PIR d;

	static {
		String _verbose =System.getenv("PTXJIT_VERBOSE");
		if (_verbose != null) {
			verbose = Integer.parseInt(_verbose);
		}
	}
	
	private static String PTXFILENAME="a.ptx";
	public static void runNVCC(byte[] ptxImage) {
		try {
		_runNVCC(ptxImage);
		} catch(Exception e) {
			e.printStackTrace();
		}
	}
	static void _runNVCC(byte[] ptxImage) throws Exception {
		FileOutputStream fos = new FileOutputStream(PTXFILENAME);
		fos.write(ptxImage);
		fos.close();
		String[] nvccCMD = new String[] {
				"nvcc", "--cubin", "-o", "/dev/null",
				PTXFILENAME
		};
		printf("executing:");
		for(String a: nvccCMD)
			printf( " %s", a);
		printf("\n");
//		ProcessBuilder pb = new ProcessBuilder(nvccCMD);
//		pb.redirectErrorStream(true);
//		Process p =pb.start();
		Process p = Runtime.getRuntime().exec(nvccCMD);
		int r = p.waitFor();
		printf("nvcc returns %d\n", r);
	}
	
}
