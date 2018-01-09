/*
 * JITPTX Project.
 * 
 * @author Byeongcheol Lee
 */
package org.ptx.dispatch;

import java.util.Enumeration;
import java.util.HashMap;
import java.util.Map;

import org.jikesrvm.compilers.opt.ir.ALoad;
import org.jikesrvm.compilers.opt.ir.AStore;
import org.jikesrvm.compilers.opt.ir.Binary;
import org.jikesrvm.compilers.opt.ir.IR;
import org.jikesrvm.compilers.opt.ir.IfCmp;
import org.jikesrvm.compilers.opt.ir.Label;
import org.jikesrvm.compilers.opt.ir.Move;
import org.jikesrvm.compilers.opt.ir.Register;
import org.jikesrvm.compilers.opt.ir.Instruction;
import org.jikesrvm.compilers.opt.ir.Operators;
import org.jikesrvm.compilers.opt.ir.Return;
import org.jikesrvm.compilers.opt.ir.operand.ConditionOperand;
import org.jikesrvm.compilers.opt.ir.operand.IntConstantOperand;
import org.jikesrvm.compilers.opt.ir.operand.Operand;
import org.jikesrvm.compilers.opt.ir.operand.RegisterOperand;

import org.ptx.ir.PIR;
import org.ptx.ir.PInstruction;
import org.ptx.ir.PTXIntConstantOperand;
import org.ptx.ir.PTXOperand;
import org.ptx.ir.PTXRegister;
import org.ptx.ir.PTXRegisterOperand;
import org.ptx.ir.PTX.PTXComOP;
import static org.ptx.Util.*;
import static org.ptx.ir.PTX.*;

public final class LIR2PIR {

	static PTXRegType getPTXRegisterType(Register r) {
		if (r.isInteger()) 
			return PTXRegType.s32;
		else if (r.isLong())
			return PTXRegType.s64;
		else if (r.isAddress())
			return PTXRegType.b64;
		else if (r.isDouble())
			return PTXRegType.f64;
		else {
			_assert(false, "%s", r);
			return null;
		}	
	}

	final IR lir;
	final PIR pir;

	final Map<Register,PTXRegister> regmap = new HashMap<Register, PTXRegister>();
	private int nextPTXRegisterID = 0;
	
	LIR2PIR(IR lir) {
		this.lir = lir;
		this.pir = new PIR(lir.getMethod());
	}

	private PTXRegister createPTXRegister(PTXRegType t) {
		int id = nextPTXRegisterID++;
		PTXRegister r = new PTXRegister(t, id);
		pir.registers.add(r);
		return r;
	}

	public PIR execute() {
		for(Enumeration<Instruction> ei=lir.forwardInstrEnumerator();
		ei.hasMoreElements();) {
			Instruction i = ei.nextElement();
			for(Enumeration<Operand> eo =i.getOperands();eo.hasMoreElements();) {
				Operand o = eo.nextElement();
				if (o instanceof RegisterOperand == false)
					continue;
				RegisterOperand ro = (RegisterOperand)o;
				Register reg = ro.getRegister();
				if (reg.isValidation())
					continue;
				if (regmap.containsKey(reg))
					continue;
				PTXRegType ptxtype = getPTXRegisterType(reg);
				PTXRegister preg = createPTXRegister(ptxtype);
				regmap.put(reg, preg);
				//emit("  .reg .%s %s;", preg.type, preg);
			}
		}

		for(Enumeration<Instruction> ei=lir.forwardInstrEnumerator();
		ei.hasMoreElements();) {
			Instruction i = ei.nextElement();
			if (i ==lir.firstInstructionInCodeOrder())
				continue;
			//emit("//%5d %s", i.getBytecodeIndex(), i);
			switch(i.operator().getOpcode())
			{
			case Operators.IR_PROLOGUE_opcode:
				code_IR_PROLOGUE(i);
				break;
			case Operators.LABEL_opcode:
				code_LABEL(i);
				break;
			case Operators.INT_MOVE_opcode:
				code_INT_MOVE(i);
				break;
			case Operators.INT_IFCMP_opcode:
				code_INT_IFCMP(i);
				break;
			case Operators.ARRAYLENGTH_opcode:
				code_ARRAYLENGTH(i);
				break;
			case Operators.DOUBLE_ALOAD_opcode:
				code_DOUBLE_ALOAD(i);
				break;
			case Operators.DOUBLE_ADD_opcode:
				code_DOUBLE_ADD(i);
				break;
			case Operators.DOUBLE_ASTORE_opcode:
				code_DOUBLE_ASTORE(i);
				break;
			case Operators.INT_ADD_opcode:
				code_INT_ADD(i);
				break;
			case Operators.RETURN_opcode:
				code_RETURN(i);
				break;
			default:
				break;
			}
		}
		return pir;
	}
	private void code_RETURN(Instruction i) {
		//Return.getVal(i);
		emit("ret;");
	}

	private void code_INT_ADD(Instruction i) {
		emit("add.%s %s, %s, %s;", PTXRegType.s32,
				getPTXOperand(Binary.getResult(i)),
				getPTXOperand(Binary.getVal1(i)),
				getPTXOperand(Binary.getVal2(i)));
	}

	private void code_DOUBLE_ADD(Instruction i) {
		emit("add.%s %s, %s, %s;", PTXRegType.f64,
				getPTXOperand(Binary.getResult(i)),
				getPTXOperand(Binary.getVal1(i)),
				getPTXOperand(Binary.getVal2(i)));
	}

	private void code_DOUBLE_ALOAD(Instruction i) {
		PTXRegisterOperand d = (PTXRegisterOperand)getPTXOperand(ALoad.getResult(i));
		PTXOperand idx = getPTXOperand(ALoad.getIndex(i));
		PTXRegisterOperand arrayOpr = (PTXRegisterOperand)getPTXOperand(ALoad.getArray(i));
		PTXRegister raddr = createPTXRegister(PTXRegType.b64);
		PTXRegister offsetReg = createPTXRegister(PTXRegType.b64);
//		emit("mad.wide.u32 %s, %s, %s, %s;", 
//				new PTXRegisterOperand(raddr), idx, new PTXIntConstantOperand(8), rarray);
//		emit("ld.global.%s %s, [%s];", 
//				PTXRegType.f64, d, new PTXRegisterOperand(raddr));
		
		emit("cvta.to.global.%s %s, %s;", 
				PTXRegType.u64, new PTXRegisterOperand(raddr), arrayOpr);
		emit("mul.wide.s32 %s, %s, %s;", 
				new PTXRegisterOperand(offsetReg),
				idx, new PTXIntConstantOperand(8));
		emit("add.%s %s, %s, %s;", PTXRegType.s64, 
				new PTXRegisterOperand(raddr), 
				new PTXRegisterOperand(raddr),
				new PTXRegisterOperand(offsetReg));
		emit("ld.global.%s %s, [%s];", 
				PTXRegType.f64, d, new PTXRegisterOperand(raddr));
	}

	private void code_DOUBLE_ASTORE(Instruction i) {
		PTXRegisterOperand v = (PTXRegisterOperand)getPTXOperand(AStore.getValue(i));
		PTXOperand idx = getPTXOperand(AStore.getIndex(i));
		PTXRegisterOperand arrayOpr = (PTXRegisterOperand)getPTXOperand(AStore.getArray(i));
		PTXRegister raddr= createPTXRegister(PTXRegType.b64);
		PTXRegister offsetReg= createPTXRegister(PTXRegType.b64);

//		emit("mad.wide.u32 %s, %s, %s, %s;", 
//				new PTXRegisterOperand(raddr), idx, new PTXIntConstantOperand(8), rarray);
//		emit("st.global.%s [%s], %s;", PTXRegType.f64,
//				new PTXRegisterOperand(raddr), v);
		emit("cvta.to.global.%s %s, %s;", PTXRegType.u64,
				new PTXRegisterOperand(raddr), arrayOpr);
		emit("mul.wide.s32 %s, %s, %s;", 
				new PTXRegisterOperand(offsetReg),
				idx, new PTXIntConstantOperand(8));
		emit("add.%s %s, %s, %s;", PTXRegType.s64, 
				new PTXRegisterOperand(raddr), 
				new PTXRegisterOperand(raddr), 
				new PTXRegisterOperand(offsetReg));
		emit("st.global.%s [%s], %s;", 
				PTXRegType.f64,
				new PTXRegisterOperand(raddr),
				v);
	}


	private void code_ARRAYLENGTH(Instruction i) {}

	private final Map<Integer,PTXComOP> lircond2ptxcomop = new HashMap<Integer,PTXComOP>();
	{
		lircond2ptxcomop.put(ConditionOperand.LESS_EQUAL, PTXComOP.le);
		lircond2ptxcomop.put(ConditionOperand.LESS, PTXComOP.lt);
	}

	static boolean isRegisterOrIntContant(Operand o) {
		return o.isRegister() || o.isIntConstant();
	}

	private PTXOperand getPTXOperand(Operand o) {
		if (o.isRegister()) {
			RegisterOperand ro = o.asRegister();
			PTXRegister pr = regmap.get(ro.getRegister());
			return new PTXRegisterOperand(pr);
		} else if (o.isIntConstant()) {
			IntConstantOperand io = o.asIntConstant();
			return new PTXIntConstantOperand(io.value);
		} else {
			_assert(false, "%s", o);
			return null;
		}
	}

	private void code_INT_IFCMP(Instruction i) {
		_assert(isRegisterOrIntContant(IfCmp.getVal1(i)));
		_assert(isRegisterOrIntContant(IfCmp.getVal2(i)));
		ConditionOperand cop = IfCmp.getCond(i);
		PTXComOP pcond = lircond2ptxcomop.get(cop.value);
		PTXOperand po1 = getPTXOperand(IfCmp.getVal1(i));
		PTXOperand po2 = getPTXOperand(IfCmp.getVal2(i));
		int bbno = Label.getBlock(IfCmp.getTarget(i).target).block.getNumber();
		PTXRegister predReg = createPTXRegister(PTXRegType.pred);

		emit("setp.%s.%s %s, %s, %s;",
				pcond, PTXRegType.s32, predReg, 
				po1, 
				po2);//new PTXIntConstantOperand(100));
		emit("@%s bra BB%d;", predReg, bbno);
	}

	private void code_INT_MOVE(Instruction i) {
		Operand d = Move.getResult(i);
		Operand s = Move.getVal(i);
		_assert(d instanceof RegisterOperand);
		_assert(s instanceof RegisterOperand || s instanceof IntConstantOperand);

		if (s instanceof RegisterOperand) {
			PTXRegister pd = regmap.get(d.asRegister().getRegister());
			PTXRegister ps = regmap.get(s.asRegister().getRegister());
			emit("mov.%s %s, %s;", pd.type, pd, ps);
		} else if (s instanceof IntConstantOperand) {
			PTXRegister pd = regmap.get(d.asRegister().getRegister());
			int imm = s.asIntConstant().value;
			emit("mov.%s %s, %d;", pd.type, pd, imm);
		} else {
			_assert(false);
		}
	}

	private void code_LABEL(Instruction i) {
		int bbno = Label.getBlock(i).block.getNumber();
		emit("BB%d:", bbno);
	}

	void code_IR_PROLOGUE(Instruction i) {
		int paramOrdinal = 0;
		for(Enumeration<Operand> oe = i.getDefs();
		oe.hasMoreElements();) {
			RegisterOperand param = (RegisterOperand)oe.nextElement();
			PTXRegister preg = regmap.get(param.getRegister());
			emit("ld.param.%s %s, [param%d];", 
					preg.type, preg, paramOrdinal);
			paramOrdinal++;
		}
	}
	public void emit(String fmt, Object...args) {
		PInstruction pi = new PInstruction(String.format(fmt, args));
		this.pir.instructions.add(pi);
	}
}
