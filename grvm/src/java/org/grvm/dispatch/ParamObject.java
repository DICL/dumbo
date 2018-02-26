package org.grvm.dispatch;

import org.jikesrvm.runtime.Magic;

public class ParamObject {
	private final long value;
	
	private ParamObject(Object o) {
		value = Magic.objectAsAddress(o).toLong();
	}
	
	public static ParamObject valueOf(Object o) {
		return new ParamObject(o);
	}
}
