package org.grvm.compiler;

import java.util.Enumeration;
import java.util.LinkedList;

import static org.jikesrvm.compilers.opt.ir.Operators.CALL;
import static org.jikesrvm.compilers.opt.ir.Operators.INT_IFCMP;
import static org.jikesrvm.compilers.opt.ir.Operators.NEWARRAY;
import static org.jikesrvm.compilers.opt.ir.Operators.REF_ASTORE;

import org.jikesrvm.classloader.TypeReference;
import org.jikesrvm.compilers.opt.DefUse;
import org.jikesrvm.compilers.opt.controlflow.DominatorsPhase;
import org.jikesrvm.compilers.opt.controlflow.LSTNode;
import org.jikesrvm.compilers.opt.driver.CompilerPhase;
import org.jikesrvm.compilers.opt.inlining.InlineSequence;
import org.jikesrvm.compilers.opt.ir.AStore;
import org.jikesrvm.compilers.opt.ir.BasicBlock;
import org.jikesrvm.compilers.opt.ir.Call;
import org.jikesrvm.compilers.opt.ir.Goto;
import org.jikesrvm.compilers.opt.ir.IR;
import org.jikesrvm.compilers.opt.ir.IRTools;
import org.jikesrvm.compilers.opt.ir.IfCmp;
import org.jikesrvm.compilers.opt.ir.Instruction;
import org.jikesrvm.compilers.opt.ir.NewArray;
import org.jikesrvm.compilers.opt.ir.Register;
import org.jikesrvm.compilers.opt.ir.operand.BranchProfileOperand;
import org.jikesrvm.compilers.opt.ir.operand.ConditionOperand;
import org.jikesrvm.compilers.opt.ir.operand.LocationOperand;
import org.jikesrvm.compilers.opt.ir.operand.MethodOperand;
import org.jikesrvm.compilers.opt.ir.operand.RegisterOperand;
import org.jikesrvm.compilers.opt.ir.operand.TypeOperand;
import org.grvm.dispatch.GPUEntrypoints;
import org.grvm.dispatch.GPUKernelMethod;
import org.grvm.dispatch.GPUKernelMethods;

import static org.grvm.Util._assert;
import static org.grvm.Util.printf;

public class GPUParallelizer extends CompilerPhase {
	private static final boolean DEBUG = false;

	IR ir;
	final LinkedList<GPUKernelInfo> gpuInfoList = 
			new LinkedList<GPUKernelInfo>();

	@Override
	public String getName() {
		return "GPU Parallelizer";
	}

  @Override
  public CompilerPhase newExecution(IR ir) {
    return this;
  }

	@Override
	public void perform(IR ir) {
    if (ir.hasReachableExceptionHandlers()) {
    	return;
    }
    this.ir =ir;
    new DominatorsPhase(false).perform(ir);
    DefUse.computeDU(ir);
    RegionTree regionTree = new RegionTree(ir, ir.HIRInfo.loopStructureTree);
    if (DEBUG) regionTree.print();

		collect(regionTree);
		for(GPUKernelInfo gki: gpuInfoList) {
			processGPURegion(gki);
		}
		if (DEBUG) {
			ir.printInstructions();
		}
	}

	private void collect(RegionTree t) {
		for(Enumeration<LSTNode> e = t.getRoot().getChildren();
				e.hasMoreElements();){
			RegionNode c = (RegionNode)e.nextElement();
			collect(c);
		}
	}

	private void collect(RegionNode lstNode) {
		BasicBlock header = lstNode.getHeader();
		Instruction i = header.firstRealInstruction();

		if (i.position().getMethod() == GPUEntrypoints.dispatcherInvokeMethod) {
			GPUKernelInfo ki = new GPUKernelInfo(lstNode);
			gpuInfoList.add(ki);
			return;
		}
		for(Enumeration<LSTNode> e = lstNode.getChildren();
				e.hasMoreElements();){
			RegionNode c = (RegionNode)e.nextElement();
			collect(c);
		}
	}

	private void processGPURegion(GPUKernelInfo gi) {		
		String entry = "kmain";
		GPUCodeGen gcgen = new GPUCodeGen(ir, gi, entry);
		gcgen.perform();
		byte[] ptxcode = gcgen.getCode();
		byte[] paramTypes = new byte[gi.params.size()];
		int i = 0;
		for(Register p: gi.params) {
			if (p.isInteger()) {
				paramTypes[i] = 'I';
			} else if (p.isAddress()) {
				paramTypes[i] = 'A';
			} else {
				_assert(false, "TBI: %s", p);
			}
			i = i + 1;
		}
		GPUKernelMethod gmethod = GPUKernelMethods.allocate(
				entry, ptxcode, paramTypes);

		RegionNode region = gi.loop;
		
		int bcidx = region.header.firstInstruction().getBytecodeIndex();
		InlineSequence pos = region.header.firstInstruction().position();
		BasicBlock predBB = region.predecessor;
		BasicBlock succBB = region.successor;
		_assert(1 == predBB.getNumberOfOut());

		BasicBlock gpuEntryBB = new BasicBlock(bcidx, pos, ir.cfg);
		BasicBlock gpuExitBB = new BasicBlock(bcidx, pos, ir.cfg);
		ir.cfg.insertAfterInCodeOrder(predBB, gpuEntryBB);
		ir.cfg.insertAfterInCodeOrder(gpuEntryBB, gpuExitBB);
		gpuEntryBB.insertOut(gpuExitBB);

		Instruction branchToRegionInst = predBB.lastRealInstruction();
		_assert(Goto.conforms(branchToRegionInst));
		Register regOffload = ir.regpool.getCondition();
		Instruction instCall = Call.create0(CALL, IRTools.I(regOffload), 
			IRTools.AC(GPUEntrypoints.shouldOffloadMethod.getOffset()), 
			MethodOperand.STATIC(GPUEntrypoints.shouldOffloadMethod));
		instCall.setPosition(pos);
		instCall.setBytecodeIndex(bcidx);
		branchToRegionInst.insertBefore(instCall);
		Instruction instCondBranch = IfCmp.create(INT_IFCMP, 
			ir.regpool.makeTempValidation(), 
			IRTools.I(regOffload), 
			IRTools.IC(0), ConditionOperand.NOT_EQUAL(), 
			gpuEntryBB.makeJumpTarget(),
			BranchProfileOperand.likely());
		instCondBranch.setPosition(pos);
		instCondBranch.setBytecodeIndex(bcidx);
		branchToRegionInst.insertBefore(instCondBranch);
		predBB.insertOut(gpuEntryBB);

		gpuExitBB.appendInstruction(succBB.makeGOTO());
		gpuExitBB.insertOut(succBB);
		generateGPUKernelLaunch(gi,gmethod, gpuEntryBB, gpuExitBB);

		if (DEBUG) {
			for (BasicBlock bb = gpuEntryBB; true; bb = bb.nextBasicBlockInCodeOrder()) {
				for (Enumeration<Instruction> e = bb.forwardInstrEnumerator(); e.hasMoreElements();) {
					Instruction inst = e.nextElement();
					printf("BB%02d: %4d %s\n", bb.getNumber(), inst.getBytecodeIndex(), inst.toString());
				}
				if (bb == gpuEntryBB) {
					break;
				}
			}
		}
	}

	private void generateGPUKernelLaunch(
			GPUKernelInfo gi, GPUKernelMethod gmethod,
			BasicBlock klEntry, BasicBlock klExit) {
		BasicBlock bb = klEntry;

		int numParams = gmethod.paramTypes.length;
		RegisterOperand ropDefParams = ir.regpool.makeTemp(
				TypeReference.JavaLangObjectArray);
		Instruction instNewParamArray = NewArray.create(NEWARRAY, 
				ropDefParams, 
				new TypeOperand(TypeReference.JavaLangObjectArray), 
				IRTools.IC(numParams));
		bb.appendInstruction(instNewParamArray);
		int i = 0;
		for(Register preg: gi.params) {
			RegisterOperand ropParam;
			switch(gmethod.paramTypes[i]) {
			case 'I': {
				ropParam = ir.regpool.makeTemp(
						GPUEntrypoints.typeRefInteger);
				Instruction instInvokeValueOf = Call.create1(
						CALL, ropParam, 
						IRTools.AC(GPUEntrypoints.integerValueofmethod.getOffset()),
						MethodOperand.STATIC(GPUEntrypoints.integerValueofmethod),
						IRTools.I(preg));
				bb.appendInstruction(instInvokeValueOf);
				break;
			}
			case 'A': {
				ropParam = ir.regpool.makeTemp(
						GPUEntrypoints.typeRefParamObject);
				Instruction instInvokeValueOf = Call.create1(
						CALL, ropParam, 
						IRTools.AC(GPUEntrypoints.paramObjectValueofMethod.getOffset()),
						MethodOperand.STATIC(GPUEntrypoints.paramObjectValueofMethod),
						IRTools.A(preg));
				bb.appendInstruction(instInvokeValueOf);
				break;
			}
			default:
				ropParam = null;
				_assert(false, "TBI:%c", gmethod.paramTypes[i]);
				break;
			}
			
			Instruction instAstore = AStore.create(
					REF_ASTORE, ropParam.copyD2U(), 
					ropDefParams.copyD2U(),
					IRTools.IC(i), 
					new LocationOperand(ropDefParams.getType().getArrayElementType()),
					IRTools.TG());
			bb.appendInstruction(instAstore);

			i = i + 1;
		}

		RegisterOperand ropDefKernelMethod = ir.regpool.makeTemp(
				GPUEntrypoints.typeRefGPUKernelMethod);
		Instruction instGetKernelMethod = Call.create1(
				CALL, ropDefKernelMethod, 
				IRTools.AC(GPUEntrypoints.getGPUKernelMethod.getOffset()),
				MethodOperand.STATIC(GPUEntrypoints.getGPUKernelMethod), 
				IRTools.IC(gmethod.id));
		bb.appendInstruction(instGetKernelMethod);

		Instruction instCallLaunch = Call.create4(
				CALL, null, 
				IRTools.AC(GPUEntrypoints.launchGPUKernelMethod.getOffset()), 
				MethodOperand.STATIC(GPUEntrypoints.launchGPUKernelMethod), 
				ropDefKernelMethod.copyD2U(), // kernel method
				gi.loop.initialIteratorValue.copy(), //from
				gi.loop.terminalIteratorValue.copy(), //to
				ropDefParams.copyD2U()); // param values
		bb.appendInstruction(instCallLaunch);

		InlineSequence posEntry = klEntry.firstInstruction().position();
		int bcidxEntry = klEntry.firstInstruction().getBytecodeIndex();		
		for(Enumeration<Instruction> e = bb.forwardRealInstrEnumerator();
				e.hasMoreElements();) {
			Instruction inst = e.nextElement();
			inst.setPosition(posEntry);
			inst.setBytecodeIndex(bcidxEntry);
		}
	}
}
