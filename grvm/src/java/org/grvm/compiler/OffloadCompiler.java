package org.grvm.compiler;

import org.jikesrvm.classloader.NormalMethod;
import org.jikesrvm.classloader.RVMClass;
import org.jikesrvm.classloader.TypeReference;
import org.jikesrvm.compilers.common.CompiledMethod;
import org.jikesrvm.compilers.opt.LocalCSE;
import org.jikesrvm.compilers.opt.LocalCastOptimization;
import org.jikesrvm.compilers.opt.LocalConstantProp;
import org.jikesrvm.compilers.opt.LocalCopyProp;
import org.jikesrvm.compilers.opt.OptOptions;
import org.jikesrvm.compilers.opt.Simple;
import org.jikesrvm.compilers.opt.bc2ir.ConvertBCtoHIR;
import org.jikesrvm.compilers.opt.controlflow.BranchOptimizations;
import org.jikesrvm.compilers.opt.controlflow.CFGTransformations;
import org.jikesrvm.compilers.opt.driver.CompilationPlan;
import org.jikesrvm.compilers.opt.driver.CompilerPhase;
import org.jikesrvm.compilers.opt.driver.OptimizationPlanAtomicElement;
import org.jikesrvm.compilers.opt.driver.OptimizationPlanCompositeElement;
import org.jikesrvm.compilers.opt.driver.OptimizationPlanElement;
import org.jikesrvm.compilers.opt.hir2lir.ConvertHIRtoLIR;
import org.jikesrvm.compilers.opt.hir2lir.ExpandRuntimeServices;
import org.jikesrvm.compilers.opt.ir.IR;
import org.jikesrvm.compilers.opt.lir2mir.ConvertLIRtoMIR;
import org.jikesrvm.compilers.opt.liveness.LiveAnalysis;
import org.jikesrvm.compilers.opt.mir2mc.ConvertMIRtoMC;
import org.jikesrvm.compilers.opt.regalloc.ExpandCallingConvention;
import org.jikesrvm.compilers.opt.regalloc.PrologueEpilogueCreator;
import org.jikesrvm.compilers.opt.regalloc.RegisterAllocator;
import org.jikesrvm.compilers.opt.regalloc.ia32.MIRSplitRanges;
import org.jikesrvm.compilers.opt.regalloc.ia32.RewriteMemoryOperandsWithOversizedDisplacements;

public final class OffloadCompiler {
	final NormalMethod rootMethod;
	final RVMClass actionClass;
	final TypeReference[] paramTypes;
	public CompiledMethod hostCompiledMethod;

	public OffloadCompiler(NormalMethod method, RVMClass cls) {
		rootMethod = method;
		actionClass = cls;
		paramTypes = new TypeReference[] {
				TypeReference.Int,
				TypeReference.Int,
				cls.getTypeRef(),
		};
	}

	public void compile() {
		OptOptions options = new OptOptions();
		options.setOptLevel(3); // enable aggressive inlining and CFGTransformations
		options.INLINE_MAX_TARGET_SIZE = 30;
		CompilationPlan plan = new CompilationPlan(rootMethod, paramTypes, optPlan, null, options);
		IR ir = plan.execute();
		ir.compiledMethod.compileComplete(ir.MIRInfo.machinecode);
		hostCompiledMethod = ir.compiledMethod;
	}

	
	static final OptimizationPlanElement[] optPlan = new OptimizationPlanElement[] {
		c("Convert Bytecodes to HIR",
				p(new ConvertBCtoHIR()),
				p(new BranchOptimizations(0, true, false)),
				p(new LocalCopyProp()),
				p(new LocalConstantProp()),
				p(new LocalCSE(true))),
				p(new Simple(0, false, false, false, false)),
		c("GPU Offloading",
				p(new CFGTransformations()),
				p(new GPUParallelizer())),
		c("Convert HIR to LIR",
				p(new ExpandRuntimeServices()),
				p(new BranchOptimizations(1, true, true)),
				p(new LocalCastOptimization()),
				p(new ConvertHIRtoLIR()),
				p(new BranchOptimizations(0, true, true))),
    p(new LocalCopyProp()),
    p(new LocalConstantProp()),
    p(new LocalCSE(false)),
    new ConvertLIRtoMIR(),
    c("Register Mapping", 
    		p(new RewriteMemoryOperandsWithOversizedDisplacements()),
    		p(new MIRSplitRanges()),
    		p(new ExpandCallingConvention()),
    		p(new LiveAnalysis(true, false)),
    		new RegisterAllocator(),
    		p(new PrologueEpilogueCreator())),
    new ConvertMIRtoMC(),
	};

	static OptimizationPlanElement p(CompilerPhase cp) {
		return new OptimizationPlanAtomicElement(cp);
	}

	static OptimizationPlanElement c(String name,
			OptimizationPlanElement... elements) {
		return new OptimizationPlanCompositeElement(name, elements);
	}

	static class OptDebug extends CompilerPhase {

		@Override
		public String getName() {
			return "OptDebug";
		}

		@Override
		public void perform(IR ir) {
			((RegionTree)ir.HIRInfo.loopStructureTree).print();
			ir.printInstructions();
		}

		@Override
	  public CompilerPhase newExecution(IR ir) {
	    return this;
	  }
	}
	static class OptPrinter extends CompilerPhase {
		public String getName() {
			return "OptPrinter";
		}
		public void perform(IR ir) {
			ir.printInstructions();
		}
	  public CompilerPhase newExecution(IR ir) {
	    return this;
	  }		
	}
}
