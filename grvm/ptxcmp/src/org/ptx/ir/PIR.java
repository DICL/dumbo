/*
 * JITPTX Project.
 * 
 * @author Byeongcheol Lee
 * @author Bongsuk Ko
 */
package org.ptx.ir;

import java.io.*;
import java.util.LinkedList;

import org.jikesrvm.classloader.*;
import static org.ptx.Util.*;

public class PIR {
	protected final NormalMethod method;
	protected final String entryPointName = "ptxentry";
	public final LinkedList<PTXRegister> registers = new LinkedList<PTXRegister>();
	public final LinkedList<PInstruction> instructions = new LinkedList<PInstruction>();

	public PIR(NormalMethod method) {
		this.method = method;
	}

	public NormalMethod getMethod() {
		return method;
	}

	public void printInstructions() {
		warning("TBI");
	}

	public String getEntryPointName() {
		return entryPointName;
	}

	public void emitCode(PrintWriter os) {
		os.printf(".version 4.3\n");
		os.printf(".target sm_20\n");
		os.printf(".address_size 64\n");
		os.printf(".visible .entry %s(\n", entryPointName);
		int nextParamOrdinal = 0;
		for(TypeReference tref:method.getParameterTypes()) {
			int ordinal = nextParamOrdinal++;
			if (tref.isArrayType()) {
				TypeReference etref = tref.getArrayElementType();
				_assert(etref.isPrimitiveType(), "support array of primitive types");
				os.printf("  .param .u64 param%d", ordinal);
			} else if (tref.isIntType()) {
				os.printf("  .param .s32 param%d", ordinal);
			} else {
				_assert(false,"TBI");
			}
			if (ordinal != method.getParameterTypes().length-1) {
				os.printf(",\n");
			}
		}
		os.printf(")\n");
		os.printf("{\n");
		
		for(PTXRegister r: registers) {
			os.printf("  .reg .%s %s;\n", r.type, r);
		}
		for(PInstruction i: instructions) {
			os.printf("  %s\n", i.ptxLine);
		}
		//os.printf("   ret;\n");
		os.printf("}\n");
	}
}
