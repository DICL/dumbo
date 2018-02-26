package org.grvm;

import java.io.File;
import java.io.FileInputStream;

public class Util {

	public static void _assert(boolean cond) {
		if (cond)
			return;
		Throwable t = new Throwable();
		StackTraceElement[] frames = t.getStackTrace();
		StackTraceElement pos = frames[1];
	    System.out.printf("%s:%d: assertion fail: ",
	    		pos.getFileName(), pos.getLineNumber());
		for(int i = 1; i < frames.length;i++) {
			System.out.printf(" at %s\n", frames[i]);
		}
		System.exit(1);
	}

	public static void _assert(boolean cond, String fmt, Object... args) {
		if (cond)
			return;
		Throwable t = new Throwable();
		StackTraceElement[] frames = t.getStackTrace();
		StackTraceElement pos = frames[1];
	    System.out.printf("%s:%d: assertion fail: ",
	    		pos.getFileName(), pos.getLineNumber());
		System.out.printf(fmt + "\n", args);
		for(int i = 1; i < frames.length;i++) {
			System.out.printf(" at %s\n", frames[i]);
		}
		System.exit(1);
	}
	
	public static void warning(String fmt, Object...args) {
		Throwable t = new Throwable();
		StackTraceElement[] frames = t.getStackTrace();
		StackTraceElement pos = frames[1];
	    System.out.printf("%s:%d: warning: ",
	    		pos.getFileName(), pos.getLineNumber());
		System.out.printf(fmt + "\n", args);
	}

	public static void printf(String fmt, Object...args) {
		System.out.printf(fmt, args);
	}

	public static byte[] file2bytes(String fname) {
		int flen = (int)new File(fname).length();
		byte[] buf = new byte[flen];
		FileInputStream fis;
		try {
			fis = new FileInputStream(fname);
			int nread = fis.read(buf);
			_assert(nread == flen);
			fis.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
		return buf;
	}
}
