/*
 * JITPTX Project.
 * 
 * @author Byeongcheol Lee 
 */
package org.ptx.ir;

public class PTXRegisterOperand extends PTXOperand {
	final PTXRegister reg;

	public PTXRegisterOperand(PTXRegister reg) {
		this.reg = reg;
	}

	public String toString() {
		return reg.toString();
	}
}
