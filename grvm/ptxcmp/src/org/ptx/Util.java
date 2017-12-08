/*
 * JITPTX Project.
 * 
 * @author Byeongcheol Lee
 * @author Bongsuk Ko
 */
package org.ptx;

import org.jikesrvm.classloader.*;

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

	public static void disassemble(NormalMethod m) {
		printf("Java byte code of %s\n", m.toString());
		BytecodeStream bs =  m.getBytecodes();
		while (bs.hasMoreBytecodes()) {
			int bci = bs.index();
			int opcode = bs.nextInstruction();
			String code = BytecodeConstants.JBC_name(opcode);
			int Local,Branch,Inc = 0;
		
			if(code=="iconst_0"){
				printf("%3d: %3s\n",bci, code);
			}
			
			else if(code=="istore"){
				Local = bs.getLocalNumber();
				printf("%3d: %3s %7d\n",bci, code, Local);
			}
			
			else if(code=="iload"){
				Local = bs.getLocalNumber();
				printf("%3d: %3s %8d\n",bci,code,Local);
			}	
			
			else if(code=="iload_3"){
				printf("%3d: %3s\n",bci, code);
			}

			else if(code=="if_icmpge"){
				Branch = bs.getBranchOffset()+bci;
				printf("%3d: %3s %5d\n",bci,code,Branch);
			}
			
			else if(code=="aload_0"|code=="aload_1"|code=="aload_2"){
				printf("%3d: %3s\n",bci,code);
			}
			
			else if(code=="daload"){
				printf("%3d: %3s\n",bci,code);
			}
			else if(code=="dadd"){
				printf("%3d: %3s\n",bci,code);
			}
			else if(code=="dastore"){
				printf("%3d: %3s\n",bci,code);
			}
			
			else if(code=="iinc"){
				Local = bs.getLocalNumber();
				Inc = bs.getIncrement();
				printf("%3d: %3s %9d, %d\n",bci,code,Local,Inc);

			}
			else if(code=="goto"){
				Branch = bs.getBranchOffset()+bci;
				printf("%3d: %3s %9d\n",bci,code,Branch);
			}
			
			else if(code=="return"){
				printf("%3d: %3s\n",bci,code);
			}
				
			//bs.skipInstruction();
		}
	}
}
