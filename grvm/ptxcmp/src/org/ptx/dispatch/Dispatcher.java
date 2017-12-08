/*
 * JITPTX Project.
 *
 * @author Byeongcheol Lee
 * @author Bongsuk Ko
 */
package org.ptx.dispatch;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.ArrayList;

import org.jikesrvm.classloader.*;
import org.jikesrvm.compilers.opt.*;
import org.jikesrvm.compilers.opt.bc2ir.ConvertBCtoHIR;
import org.jikesrvm.compilers.opt.controlflow.BranchOptimizations;
import org.jikesrvm.compilers.opt.driver.*;
import org.jikesrvm.compilers.opt.hir2lir.ConvertHIRtoLIR;
import org.jikesrvm.compilers.opt.hir2lir.ExpandRuntimeServices;
import org.jikesrvm.compilers.opt.ir.IR;

import org.ptx.PTXJIT;
import org.ptx.ir.PIR;
import static org.ptx.Util.*;

public class Dispatcher {

	public static Method getMethod(Class<?> cls, String methodName, Class<?>...types) {
		try {
			Method m = cls.getDeclaredMethod(methodName, types);
			return m;
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(1);
			return null;
		}
	}

	public static void invoke(Method method, Object...args) {
		try {
			_invoke(method, args);
		} catch(Exception e) {
			e.printStackTrace();
			System.exit(1);
		}
	}

	private static void _invoke(Method method, Object...args) throws Exception {
		Field f = Method.class.getDeclaredField("m");
		f.setAccessible(true);
		Object vmmethod = f.get(method);
		Field f2 = vmmethod.getClass().getDeclaredField("method");
		f2.setAccessible(true);
		NormalMethod m = (NormalMethod)f2.get(vmmethod);
		if (PTXJIT.verbose >= 1)
			disassemble(m);

		OptOptions options = new OptOptions();
		options.setOptLevel(0);
		OptimizationPlanElement[] optPlan = createOptPlan(options);
		CompilationPlan plan = new CompilationPlan(m, optPlan, null, options);
		IR ir = plan.execute();
		if (PTXJIT.verbose >= 1) {
			printf("Jikes LIR of %s\n", ir.getMethod().toString());
			ir.printInstructions();
		}
		LIR2PIR c = new LIR2PIR(ir);
		PIR pir = c.execute();
		if (PTXJIT.verbose >= 2)
			pir.printInstructions();
		PTXCompiledMethod cm = new PTXCompiledMethod(pir);

		if (PTXJIT.verbose >= 1) {
			cm.dumpPTXAssembly();
			//PTXJIT.runNVCC(cm.ptxAssemblyCode);
		}

		cm.invoke(args);
	}

	private static OptimizationPlanElement[] createOptPlan(OptOptions opt) {
		ArrayList<OptimizationPlanElement> master = new ArrayList<OptimizationPlanElement>();
		for(OptimizationPlanElement e: optPlan) {
			if (e.shouldPerform(opt)) {
				master.add(e);
			}
		}
		return master.toArray(new OptimizationPlanElement[0]);
	}

	static final CompilerPhase[] phases = new CompilerPhase[] {
		new ConvertBCtoHIR(),
		new BranchOptimizations(0, true, false),
		new LocalCopyProp(),
		new LocalConstantProp(),
		new LocalCSE(true),
		
		new ExpandRuntimeServices(),
       new BranchOptimizations(1, true, true),
       new LocalCastOptimization(),
       new ConvertHIRtoLIR(),
       new BranchOptimizations(0, true, true),
       new LocalCopyProp(),
		new LocalConstantProp(),
		new LocalCSE(false),
	};
	static final OptimizationPlanElement[] optPlan;
	static {
		ArrayList<OptimizationPlanElement> master = new ArrayList<OptimizationPlanElement>();
		for(CompilerPhase p: phases) {
			OptimizationPlanAtomicElement e = new OptimizationPlanAtomicElement(p);
			master.add(e);
		}
		optPlan = master.toArray(new OptimizationPlanElement[0]);
	}
}
