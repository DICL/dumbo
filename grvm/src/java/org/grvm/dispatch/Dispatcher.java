package org.grvm.dispatch;



import java.lang.reflect.Method;
import java.util.function.IntConsumer;

import org.jikesrvm.classloader.*;
import org.jikesrvm.compilers.common.CodeArray;
import org.jikesrvm.compilers.common.CompiledMethod;
import org.jikesrvm.runtime.Magic;
import org.jikesrvm.scheduler.RVMThread;
import org.grvm.compiler.OffloadCompiler;
import org.vmmagic.pragma.Entrypoint;
import org.vmmagic.unboxed.WordArray;

import static org.jikesrvm.runtime.Reflection.REFLECTION_GPRS_MASK;
import static org.jikesrvm.runtime.Reflection.REFLECTION_GPRS_BITS;
import static org.jikesrvm.runtime.Reflection.REFLECTION_FPRS_BITS;

import static org.grvm.Util.*;

public class Dispatcher {
	private static final boolean DEBUG = false;

	static {
		System.loadLibrary("grvm");
		init();
	}

	private static native void init();
	static native void _launch(
			byte[] ptxkernel, byte[] paramTypes,
			int from, int to, Object[] paramValues);
	

	public static void offloadIntStream(int from, int last, IntConsumer action) throws Exception {
		if (DEBUG) printf("action => %s\n", action.getClass());
			
		Method m = Dispatcher.class.getMethod("invoke", int.class, int.class, IntConsumer.class);
		NormalMethod method = (NormalMethod)java.lang.reflect.JikesRVMSupport.getMethodOf(m);
		RVMClass cls = java.lang.JikesRVMSupport.getTypeForClass(action.getClass()).asClass();
		OffloadCompiler comp = new OffloadCompiler(method, cls);
		comp.compile();
		CompiledMethod cm = comp.hostCompiledMethod;
		outOfLineInvoke(cm, from, last, action);
	}

	static void outOfLineInvoke(CompiledMethod cm, int from, int last, IntConsumer action) {
		RVMMethod method = cm.getMethod();
		Object[] arguments = new Object[] {
				Integer.valueOf(from),
				Integer.valueOf(last),
				action,
		};
		int triple = org.jikesrvm.ia32.MachineReflection.countParameters(method);
		int gprs = triple & REFLECTION_GPRS_MASK;
		int spills = triple >> (REFLECTION_GPRS_BITS + REFLECTION_FPRS_BITS);
		WordArray GPRs = WordArray.create(gprs);
		WordArray Spills = WordArray.create(spills);
		byte[] fprMeta = new byte[0];
		double[] FPRs = new double[0];
		RVMThread.getCurrentThread().disableYieldpoints();
		org.jikesrvm.ia32.MachineReflection.packageParameters(
				method, null, arguments, GPRs, FPRs, fprMeta, Spills);
		CodeArray code = cm.getEntryCodeArray();
		RVMThread.getCurrentThread().enableYieldpoints();
		if (DEBUG) 	printf("code = 0x%012x\n", Magic.objectAsAddress(code).toLong());
		Magic.invokeMethodReturningVoid(code, GPRs, FPRs, fprMeta, Spills);
	}

	public static void invoke(int from, int last, IntConsumer action) {
		for(int i = from; i < last;i++) {
			action.accept(i);
		}
	}

	@Entrypoint
	public static boolean shouldOffload() {
		if (DEBUG) printf("shouldoffload => true\n");
		return true;
	}

	@Entrypoint
	public static void launch(
			GPUKernelMethod gmethod,
			int from, int to, Object[] paramValues) {
		if (DEBUG) {
			printf("launching a GPU kernel %d-%d\n", from, to);
			printf("kernel:\n%s\n", new String(gmethod.ptxcode));
			for(int i = 0; i < paramValues.length;i++) {
				printf("param%d: %s\n", i, paramValues[i].toString());
			}
		}

		byte[] ptxkernel = gmethod.ptxcode;
		byte[] paramTypes = gmethod.paramTypes;
		_launch(ptxkernel, paramTypes, from, to, paramValues);
	}			
}
