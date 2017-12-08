/*
 * JITPTX Project.
 * 
 * @author Byeongcheol Lee 
 */
package org.ptx.ir;

import static org.ptx.Util._assert;
import org.ptx.ir.PTX.PTXRegType;

public class PTXRegister {
	public final PTXRegType type;
	public final int ordinal;
	public PTXRegister(PTXRegType type, int ordinal) {
		_assert(type != null);
		this.type = type;
		this.ordinal = ordinal;
	}
	public String toString() {
		return "%r" + ordinal;
	}
}
