/*
 * JITPTX Project.
 * 
 * @author Byeongcheol Lee 
 */
package org.ptx.ir;

public class PTXIntConstantOperand extends PTXOperand {
	final int value;

	public PTXIntConstantOperand(int value) {
		this.value = value;
	}
	
	public String toString() {
		return "" + value;
	}
}
