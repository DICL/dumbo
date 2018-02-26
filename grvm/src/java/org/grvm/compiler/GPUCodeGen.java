package org.grvm.compiler;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;

import org.jikesrvm.classloader.FieldReference;
import org.jikesrvm.classloader.RVMField;
import org.jikesrvm.classloader.TypeReference;
import org.jikesrvm.compilers.opt.DefUse;
import org.jikesrvm.compilers.opt.ir.ALoad;
import org.jikesrvm.compilers.opt.ir.AStore;
import org.jikesrvm.compilers.opt.ir.BasicBlock;
import org.jikesrvm.compilers.opt.ir.Binary;
import org.jikesrvm.compilers.opt.ir.GetField;
import org.jikesrvm.compilers.opt.ir.Goto;
import org.jikesrvm.compilers.opt.ir.IR;
import org.jikesrvm.compilers.opt.ir.IfCmp;
import org.jikesrvm.compilers.opt.ir.Instruction;
import org.jikesrvm.compilers.opt.ir.Label;
import org.jikesrvm.compilers.opt.ir.Move;
import org.jikesrvm.compilers.opt.ir.Operators;
import org.jikesrvm.compilers.opt.ir.Register;
import org.jikesrvm.compilers.opt.ir.operand.ConditionOperand;
import org.jikesrvm.compilers.opt.ir.operand.Operand;
import org.jikesrvm.compilers.opt.ir.operand.RegisterOperand;

import static org.grvm.Util._assert;
import static org.grvm.Util.printf;

public class GPUCodeGen {
	private final static boolean DEBUG = false;
	final IR ir;
	final GPUKernelInfo gi;
	final String entry;
	final LinkedList<BasicBlock> bbList = new LinkedList<BasicBlock>();
	final HashSet<BasicBlock> basicBlocks = new HashSet<BasicBlock>();

	final HashMap<Register,String> vmap = new HashMap<Register,String>();
	final LinkedList<Register> params = new LinkedList<Register>();
	final LinkedList<Register> locals = new LinkedList<Register>(); 
	final HashMap<Instruction,String> i2predreg = new HashMap<Instruction,String>();
	final Register ivReg;
	int next32bitRegID = 0;
	int next64bitRegID = 0;
	int nextPredRegID = 0;
	final String predRegTermCheck = allocatePredReg();

	final String pregBlockIDX = allocateReg32();
	final String pregBlockDimX = allocateReg32();
	final String pregThreadIDX = allocateReg32();
	final String pregS64 = allocateReg64();
	
	final ByteArrayOutputStream baOut = new ByteArrayOutputStream(); 
	final PrintWriter pout = new PrintWriter(baOut);
	byte[] gpucode;

	public GPUCodeGen(IR ir, GPUKernelInfo gi, String entry) {
		this.ir = ir;
		this.gi = gi;
		this.entry = entry;
		this.ivReg = gi.loop.carriedLoopIterator.asRegister().getRegister();
	}

	private String allocatePredReg() {
		return "%p" + nextPredRegID++;
	}

	private String allocateReg32() {
		return "%r" + next32bitRegID++;
	}

	private String allocateReg64() {
		return "%rd" + next64bitRegID++;
	}

	public byte[] getCode() {
		return gpucode;
	}

	void perform() {

		RegionNode loop = gi.loop;
		for(Enumeration<BasicBlock> e = ir.forwardBlockEnumerator();
				e.hasMoreElements();) {
			BasicBlock bb = e.nextElement();
			if (loop.contains(bb)) {
				bbList.add(bb);
				basicBlocks.add(bb);
			}
		}

		if (DEBUG) {
			ir.printInstructions();
		}
		analyze();
		generate();

		if (DEBUG) {
			try {System.out.write(gpucode);} catch (IOException e) {e.printStackTrace();}
		}
	}

	private void analyze() {
		for(BasicBlock bb : bbList) {
			for(Enumeration<Instruction> ei = bb.forwardInstrEnumerator();
					ei.hasMoreElements();) {
				Instruction i = ei.nextElement();
				if (DEBUG) printf("BB%d: %3d %s\n", bb.getNumber(), i.getNumberOfUses(), i);
				for(Enumeration<Operand> eo = i.getOperands();
						eo.hasMoreElements();) {
					Operand op = eo.nextElement();
					if (!op.isRegister()) {
						continue;
					}
					RegisterOperand rop = op.asRegister();
					Register reg = rop.getRegister();
					if (reg.isValidation()) {
						continue;
					}
					ensureRegMapping(reg);
				}
				if (i.isConditionalBranch()) {
						String preg = allocatePredReg();
						i2predreg.put(i, preg);
				}
			}
		}

		for(Register reg: vmap.keySet()) {
			if (reg == ivReg) {
				continue;
			}
			if (isRegionLocal(reg)) {
				locals.add(reg);
			} else if (isReadOnlyInTheRegion(reg)){
				params.add(reg);
				gi.params.add(reg);
			} else {
				_assert(false, "TBI: %s", reg);
			}
		}

		if (DEBUG) {
			printf("IV reg: %s\n", ivReg);
			printf("params\n");
			for(Register reg: params) {
				printf("reg: %s -> %s\n", reg, vmap.get(reg));
			}
			printf("locals\n");
			for(Register reg: locals) {
				printf("reg: %s -> %s\n", reg, vmap.get(reg));
			}
		}
	}
	
	private boolean isRegionLocal(Register reg) {
		if (reg.isSSA() && !reg.spansBasicBlock() 
				&& basicBlocks.contains(reg.getFirstDef().getBasicBlock())) {
			return true;
		}
		for(Enumeration<RegisterOperand> e = DefUse.defs(reg);
				e.hasMoreElements();) {
			RegisterOperand drop = e.nextElement();
			Instruction def = drop.instruction;
			BasicBlock bb = def.getBasicBlock();
			if (!basicBlocks.contains(bb)) {
				return false;
			}
		}
		for(Enumeration<RegisterOperand> e = DefUse.uses(reg);
				e.hasMoreElements();) {
			RegisterOperand urop = e.nextElement();
			Instruction use = urop.instruction;
			BasicBlock bb = use.getBasicBlock();
			if (!basicBlocks.contains(bb)) {
				return false;
			}
		}
		return true;
	}

	private boolean isReadOnlyInTheRegion(Register reg) {
		for (Enumeration<RegisterOperand> e = DefUse.defs(reg);
				e.hasMoreElements();) {
			RegisterOperand drop = e.nextElement();
			BasicBlock bb = drop.instruction.getBasicBlock();
			if (basicBlocks.contains(bb)) {
				return false;
			}
		}
		return true;
	}

	private void ensureRegMapping(Register reg) {
		if (vmap.containsKey(reg)) {
			return;
		}
		if (reg.isInteger()) {
			String vreg = allocateReg32();
			vmap.put(reg, vreg);
		} else if (reg.isAddress()) {
			String vreg = allocateReg64();
			vmap.put(reg, vreg);
		} else {
			_assert(false, "TBI: %s", reg);
		}
	}

	private void generate() {
		int i;
		
		emit(".visible .entry kmain(");
		i = 0;
		for(Register reg: params) {
			boolean isLast = (params.size() - i) <= 1;
			emit("  .param %s p%d%s", 
					toPTXType(reg), i, isLast ? "": ",");
			i++;
		}
		emit(") {");
		if (next32bitRegID > 0) {
			emit("  .reg .b32 %%r<%d>;", next32bitRegID);
		}
		if (next64bitRegID > 0) {
			emit("  .reg .b64 %%rd<%d>;", next64bitRegID);
		}
		if (nextPredRegID > 0) {
			emit("  .reg .pred %%p<%d>;", nextPredRegID);
		}
		emit("");

		emit("BBENTRY:");
		i = 0;
		for(Register reg: params) {
			emit("  ld.param%s %s, [p%d];", toPTXType(reg), 
					toPTXOperand(reg), i);
			if (reg.isAddress()) {
				emit("  or.b64 %s, %s, 0x8000000000000000;",
						toPTXOperand(reg), toPTXOperand(reg));
			}
			i++;
		}		
		emit("");

		//		int irReg = blockIdx.x * blockDim.x + threadIdx.x;
		emit("  mov.u32 %s, %s;", pregBlockIDX, "%ctaid.x");
		emit("  mov.u32 %s, %s;", pregBlockDimX, "%ntid.x");
		emit("  mov.u32 %s, %s;", pregBlockIDX, "%tid.x");
		emit("  mad.lo.s32 %s, %s, %s, %s;", vmap.get(ivReg),
				pregBlockIDX, pregBlockDimX, pregBlockIDX);

		if (gi.loop.terminalIteratorValue.isRegister()) {
			Register termReg = gi.loop.terminalIteratorValue.asRegister().getRegister();
			emit("  setp.ge.s32 %s, %s, %s;", 
					predRegTermCheck, 
					vmap.get(ivReg), 
					vmap.get(termReg));
			emit("  @%s bra BBEND;", predRegTermCheck);
			emit("  bra BB%d;", gi.loop.header.getNumber());
			emit("");
		} else {
			_assert(false, "TBI");
		}

		for(BasicBlock bb : bbList) {
			for(Enumeration<Instruction> ei = bb.forwardInstrEnumerator();
					ei.hasMoreElements();) {
				Instruction inst = ei.nextElement();
				if (inst == gi.loop.iteratorInstr 
						|| inst == gi.loop.initInstr
						|| inst == gi.loop.cmpInstr) {
					continue;
				}
				generate(inst);
			}
		}
		emit("BBEND: ret;");
		emit("}");
		pout.close();
		gpucode = baOut.toByteArray();
	}

	private void generate(Instruction i) {

		comment("%s", i.toString());
		switch(i.getOpcode()) {
		case Operators.LABEL_opcode:
			emit("BB%d:", Label.getBlock(i).block.getNumber());
			break;
		case Operators.BBEND_opcode:
			emit("");
			break;
		case Operators.INT_IFCMP_opcode: {
			ConditionOperand cond = IfCmp.getCond(i);
			String predReg =i2predreg.get(i);
			emit("  setp.%s.s32 %s, %s, %s;", 
					getPTXCond(cond), predReg, 
					toPTXOperand(IfCmp.getVal1(i)),
					toPTXOperand(IfCmp.getVal2(i)));
			BasicBlock tbb = IfCmp.getTarget(i).target.getBasicBlock();
			if (gi.loop.contains(tbb) && tbb != gi.loop.header) {
				emit("  @%s bra BB%d;", predReg, tbb.getNumber());
			} else {
				emit("  @%s bra BBEND;", predReg);
			}
			break;
		}
		case Operators.GOTO_opcode: {
			BasicBlock tbb = Goto.getTarget(i).target.getBasicBlock();
			if (gi.loop.contains(tbb) && tbb != gi.loop.header) {
				emit("  bra BB%d;", tbb.getNumber());
			} else {
				emit("  bra BBEND;");
			}
			break;
		}
		case Operators.NULL_CHECK_opcode: {
			break;
		}
		case Operators.INT_MOVE_opcode: {
			String dst = toPTXOperand(Move.getResult(i));
			String src = toPTXOperand(Move.getVal(i));
			emit("  mov.s32 %s, %s;", dst, src);
			break;
		}
		case Operators.INT_ZERO_CHECK_opcode: {
			break;
		}
		case Operators.INT_DIV_opcode: { 
			String def = toPTXOperand(Binary.getResult(i));
			String r1 = toPTXOperand(Binary.getVal1(i));
			String r2 = toPTXOperand(Binary.getVal2(i));
			emit("  div.s32 %s, %s, %s;", def, r1, r2);
			break;
		}
		case Operators.INT_REM_opcode: {
			String def = toPTXOperand(Binary.getResult(i));
			String r1 = toPTXOperand(Binary.getVal1(i));
			String r2 = toPTXOperand(Binary.getVal2(i));
			emit("  rem.s32 %s, %s, %s;", def, r1, r2);
			break;
		}
		case Operators.INT_MUL_opcode: {
			String def = toPTXOperand(Binary.getResult(i));
			String r1 = toPTXOperand(Binary.getVal1(i));
			String r2 = toPTXOperand(Binary.getVal2(i));
			emit("  mul.lo.s32 %s, %s, %s;", def, r1, r2);
			break;
		}
		case Operators.INT_ADD_opcode: {
			String def = toPTXOperand(Binary.getResult(i));
			String r1 = toPTXOperand(Binary.getVal1(i));
			String r2 = toPTXOperand(Binary.getVal2(i));
			emit("  add.s32 %s, %s, %s;", def, r1, r2);
			break;
		}
		case Operators.BOUNDS_CHECK_opcode: {
			break;
		}
		case Operators.GUARD_COMBINE_opcode: {
			break;
		}
		case Operators.GETFIELD_opcode: {
			FieldReference fref = GetField.getLocation(i).getFieldRef();
			_assert(fref.isResolved());
			RVMField field = fref.peekResolvedField();
			TypeReference tref = fref.getFieldContentsType();
			String defPreg = toPTXOperand(GetField.getResult(i));
			String refPreg = toPTXOperand(GetField.getRef(i));
			int foffset = field.getOffset().toInt();

			if (tref.isIntType()) {
				emit("  add.u64 %s, %s, %d;", pregS64, refPreg, foffset);	
				emit("  call (%s), %s, (%s);", pregS64, "cpuptr_link", pregS64);
				emit("  ld.global.s32 %s, [%s];", defPreg, pregS64);
				emit("  call (%s), %s, (%s);", pregS64, "cpuptr_unlink", pregS64);
			} else if (tref.isReferenceType()) {
				emit("  add.u64 %s, %s, %d;", pregS64, refPreg, foffset);	
				emit("  call (%s), %s, (%s);", pregS64, "cpuptr_link", pregS64);
				emit("  ld.global.u64 %s, [%s];", defPreg, pregS64);
				emit("  call (%s), %s, (%s);", pregS64, "cpuptr_unlink", pregS64);
			} else {
				_assert(false, "TBI: %s", tref);
			}
			break;
		}
		case Operators.INT_ALOAD_opcode: {
			String def = toPTXOperand(ALoad.getResult(i));
			String ref = toPTXOperand(ALoad.getArray(i));
			String idx = toPTXOperand(ALoad.getIndex(i));
			TypeReference tref = ALoad.getLocation(i).getElementType();			
			if (tref.isIntType()) {
				emit("  cvt.u64.s32 %s, %s;", pregS64, idx);
				emit("  shl.b64 %s, %s, %d;", pregS64, pregS64, 2);
				emit("  add.u64 %s, %s, %s;", pregS64, pregS64, ref);
				emit("  or.b64 %s, %s, 0x8000000000000000;", pregS64, pregS64);
				emit("  call (%s), %s, (%s);", pregS64, "cpuptr_link", pregS64);
				emit("  ld.global.s32 %s, [%s];", def, pregS64);
				emit("  call (%s), %s, (%s);", pregS64, "cpuptr_unlink", pregS64);
			} else {
				_assert(false, "TBI: %s", tref);
			}
			break;
		}
		case Operators.INT_ASTORE_opcode: {
			String ref = toPTXOperand(AStore.getArray(i));
			String idx = toPTXOperand(AStore.getIndex(i));
			String val = toPTXOperand(AStore.getValue(i));
			TypeReference tref = ALoad.getLocation(i).getElementType();
			if (tref.isIntType()) {
				emit("  cvt.u64.s32 %s, %s;", pregS64, idx);
				emit("  shl.b64 %s, %s, %d;", pregS64, pregS64, 2);
				emit("  add.u64 %s, %s, %s;", pregS64, pregS64, ref);
				emit("  or.b64 %s, %s, 0x8000000000000000;", pregS64, pregS64);
				emit("  call (%s), %s, (%s);", pregS64, "cpuptr_link", pregS64);
				emit("  call (%s), %s, (%s, %s);", 
						pregS64, "cpuptr_write_int", pregS64, val);
				emit("  call (%s), %s, (%s);", pregS64, "cpuptr_unlink", pregS64);				
			} else {
				_assert(false, "TBI: %s", tref);
			}
			break;
		}
		default:
			_assert(false, "TBI: BB%d: %s\n", i.getBasicBlock().getNumber(), i);
			break;
		}
	}

	private void comment(String fmt, Object...args) {
		if (DEBUG) {
			pout.printf("// " + fmt + "\n", args);
		}
	}

	private void emit(String fmt, Object...args) {
		pout.printf(fmt +"\n", args);
	}
	
	private static String getPTXCond(ConditionOperand cond) {
		if (cond.isGREATER_EQUAL()) {
			return "ge";
		} else {
			_assert(false, "TBI:%s", cond);
			return "";
		}
	}

	private String toPTXOperand(Operand o) {
		if (o.isRegister()) {
			Register reg = o.asRegister().getRegister();
			return vmap.get(reg);
		} else if (o.isIntConstant()) {
			return Integer.toString(o.asIntConstant().value);
		} else {
			_assert(false, "TBI:%s", o);
			return "";
		}
	}
	
	private String toPTXOperand(Register reg) {
		return vmap.get(reg);
	}
	
	private String toPTXType(Register reg) {
		if (reg.isAddress()) {
			return ".u64";
		} else if (reg.isInteger()) {
			return ".s32";
		} else {
			_assert(false, "TBI: %s", reg);
			return "";
		}
	}
}
