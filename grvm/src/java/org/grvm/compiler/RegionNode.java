package org.grvm.compiler;

import static org.jikesrvm.compilers.opt.ir.Operators.INT_ADD_opcode;
import static org.jikesrvm.compilers.opt.ir.Operators.INT_IFCMP_opcode;
import static org.jikesrvm.compilers.opt.ir.Operators.INT_SUB_opcode;

import java.util.ArrayList;
import java.util.Enumeration;

import org.jikesrvm.compilers.opt.DefUse;
import org.jikesrvm.compilers.opt.OptimizingCompilerException;
import org.jikesrvm.compilers.opt.controlflow.LSTNode;
import org.jikesrvm.compilers.opt.ir.BasicBlock;
import org.jikesrvm.compilers.opt.ir.Binary;
import org.jikesrvm.compilers.opt.ir.Goto;
import org.jikesrvm.compilers.opt.ir.IR;
import org.jikesrvm.compilers.opt.ir.IfCmp;
import org.jikesrvm.compilers.opt.ir.Instruction;
import org.jikesrvm.compilers.opt.ir.Label;
import org.jikesrvm.compilers.opt.ir.Move;
import org.jikesrvm.compilers.opt.ir.operand.ConditionOperand;
import org.jikesrvm.compilers.opt.ir.operand.ConstantOperand;
import org.jikesrvm.compilers.opt.ir.operand.IntConstantOperand;
import org.jikesrvm.compilers.opt.ir.operand.Operand;
import org.jikesrvm.compilers.opt.ir.operand.RegisterOperand;
import org.jikesrvm.compilers.opt.util.GraphNode;
import org.jikesrvm.util.BitVector;

/**
 * A "RegionNode" represent a canonical single-entry and single-exit loop
 * where the single exit node connects to the single-entry header node.
 * It contains an induction variable that iterates over an integer range
 * from an initial value inclusively and a terminal value exclusively. 
 */
public final class RegionNode extends LSTNode {

	public BasicBlock predecessor;
	public BasicBlock exit;
	public BasicBlock successor;

	public Instruction initInstr;
	public Instruction cmpInstr;
	public Instruction iteratorInstr;
	public Operand carriedLoopIterator;
	public Operand terminalIteratorValue;
	public Operand initialIteratorValue;
	public ConditionOperand condition;
	public Operand strideValue;

	public RegionNode(IR ir, LSTNode node) {
		super(node);
		Enumeration<GraphNode> innerLoops = node.outNodes();
		while (innerLoops.hasMoreElements()) {
			RegionNode nestedLoop = new RegionNode(ir, (LSTNode) innerLoops.nextElement());
			insertOut(nestedLoop);
		}
		perform();
	}

	public boolean isIntegerRangeLoop() {
		boolean ok = true;
		ok = ok && (initialIteratorValue != null) && isConstant(initialIteratorValue);
		ok = ok && (terminalIteratorValue != null) && isConstant(terminalIteratorValue);
		ok = ok && (strideValue != null) && isConstant(strideValue);
		ok = ok && (iteratorInstr != null);
		ok = ok && ((iteratorInstr.getOpcode() == INT_ADD_opcode) || (iteratorInstr.getOpcode() == INT_SUB_opcode));
		ok = ok && getMonotonicStrideValue() == 1;
		ok = ok && condition.isLESS();
		return ok;
	}

	public boolean isInvariant(Operand op) {
		return isLoopInvariant(op, loop, header);
	}

	public boolean isCarriedLoopIterator(Operand op) {
		return op.similar(carriedLoopIterator);
	}

	public boolean isMonotonic() {
		return isConstant(strideValue);
	}

	public int getMonotonicStrideValue() {
		if (iteratorInstr.getOpcode() == INT_SUB_opcode) {
			return -((IntConstantOperand) strideValue).value;
		} else if (iteratorInstr.getOpcode() == INT_ADD_opcode) {
			return ((IntConstantOperand) strideValue).value;
		} else {
			throw new Error("Error reading stride value");
		}
	}

	public boolean contains(BasicBlock block) {
		return block.getNumber() < loop.length() 
				&& loop.get(block.getNumber());
	}

	@Override
	public String toString() {
		return  "head:   {" + header + "}:\n"
					+ "blocks: " + loop + "\n"
				  + "pred:    " + predecessor + "\n"
					+ "exit:   {" + exit + "}:\n"
					+ "succ:   " + successor + "\n"
				  + "iter:   " + carriedLoopIterator + "\n"
					+ "begin:  " + initialIteratorValue + "\n"
					+ "end:    " + terminalIteratorValue + "\n"
					+ "init:   " + initInstr + "\n"
					+ "inc:    " + iteratorInstr + "\n"
					+ "cond:   " + cmpInstr + "\n";
	}

	private static boolean isConstant(Operand op) {
		return op instanceof IntConstantOperand;
	}

	private static boolean isLoopInvariant(Operand op, BitVector loop, BasicBlock header) {
		if (op.isConstant()) {
			return true;
		} else if (op.isRegister()) {
			Enumeration<RegisterOperand> defs = DefUse.defs(((RegisterOperand) op).getRegister());
			while (defs.hasMoreElements()) {
				RegisterOperand rop = defs.nextElement();
				Instruction dinst = rop.instruction;
				if (!loop.get(dinst.getBasicBlock().getNumber())) {
					continue;
				} else {
					return false;
				}
			}
			return true;
		} else {
			return true;
		}
	}

	public boolean isInLoop(BasicBlock block) {
		return loop.get(block.getNumber());
	}

	private void checkOutEdgesAreInLoop(BasicBlock block) throws NonCannonicalRegionException {
		for(Enumeration<BasicBlock> eout = block.getOut();
				eout.hasMoreElements();) {
			BasicBlock succ = eout.nextElement();
			if ((!isInLoop(succ)) && (block != exit)) {
				fail();
			}
		}
	}

	private void checkInEdgesAreInLoop(BasicBlock block) throws NonCannonicalRegionException {
		Enumeration<BasicBlock> block_inEdges = block.getIn();
		while (block_inEdges.hasMoreElements()) {
			BasicBlock curEdgeBB = block_inEdges.nextElement();
			if ((!isInLoop(curEdgeBB)) && (block != header)) {
				fail();
			}
		}
	}

	private void perform() throws OptimizingCompilerException {
		if (loop == null) {
			return;
		}
		try {
			processHeader();
			Enumeration<BasicBlock> loopBlocks = getBasicBlocks();
			while (loopBlocks.hasMoreElements()) {
				BasicBlock curLoopBB = loopBlocks.nextElement();
				if (curLoopBB == header) {
				} else {
					processLoopBlock(curLoopBB);
				}
			}
		} catch (NonCannonicalRegionException e) {
			e.printStackTrace();
			initialIteratorValue = null;
		}
	}

	private void processHeader() throws NonCannonicalRegionException {
		Enumeration<BasicBlock> ein = header.getIn();
		while (ein.hasMoreElements()) {
			BasicBlock predBB = ein.nextElement();
			if (isInLoop(predBB)) {
				if (exit != null) {
					fail();
				}
				exit = predBB;
				if (header != exit) {
					checkInEdgesAreInLoop(exit);
				}
				Enumeration<BasicBlock> exitBlock_outEdges = exit.getOut();
				boolean exits = false;
				while (exitBlock_outEdges.hasMoreElements()) {
					BasicBlock curExitBlockOutEdgeBB = exitBlock_outEdges.nextElement();
					if (!isInLoop(curExitBlockOutEdgeBB)) {
						exits = true;
						successor = curExitBlockOutEdgeBB;
						if (successor == header) {
							fail();
						}
					}
				}
				if (!exits) {
					fail();
				}
				processExit();
			} else {
				if (predecessor != null) {
					fail();
				}
				predecessor = predBB;
			}
		}
		if (header != exit) {
			checkOutEdgesAreInLoop(header);
		}
	}

	private void processExit() throws NonCannonicalRegionException {
		cmpInstr = exit.firstBranchInstruction();
		if (cmpInstr == null) {
			fail();
		} else if (cmpInstr.getOpcode() != INT_IFCMP_opcode) {
			fail();
		}
		carriedLoopIterator = IfCmp.getVal1(cmpInstr);
		terminalIteratorValue = IfCmp.getVal2(cmpInstr);
		condition = (ConditionOperand) IfCmp.getCond(cmpInstr).copy();
		if (isLoopInvariant(carriedLoopIterator, loop, header)) {
			if (isLoopInvariant(terminalIteratorValue, loop, header)) {
				fail();
			} else {
				Operand temp = terminalIteratorValue;
				terminalIteratorValue = carriedLoopIterator;
				carriedLoopIterator = temp;
			}
		} else {
			if (isLoopInvariant(terminalIteratorValue, loop, header)) {
			} else {
				fail();
			}
		}
		if (Label.getBlock(IfCmp.getTarget(cmpInstr).target).block != header) {
			Instruction ninst = cmpInstr.nextInstructionInCodeOrder();
			if (Goto.conforms(ninst)) {
				if (Label.getBlock(Goto.getTarget(ninst).target).block == header) {
					condition.flipCode();
				} else {
					fail();
				}
			}
		}

		Enumeration<RegisterOperand> iteratorDefs = 
				DefUse.defs(((RegisterOperand) carriedLoopIterator).getRegister());
		while (iteratorDefs.hasMoreElements()) {
			Operand curDef = iteratorDefs.nextElement();
			if (isInLoop(curDef.instruction.getBasicBlock())) {
				if ((iteratorInstr == null) || (iteratorInstr == curDef.instruction)) {
					iteratorInstr = curDef.instruction;
				} else {
					fail();
				}
			} else {
				if (initInstr == null) {
					if (Move.conforms(curDef.instruction)) {
						initInstr = curDef.instruction;
						initialIteratorValue = Move.getVal(initInstr);
					} else {
						fail();
					}
				} else {
					fail();
				}
			}
		}
		if (iteratorInstr == null) {
			fail();
		}
		if ((iteratorInstr.getOpcode() != INT_ADD_opcode) 
				&& (iteratorInstr.getOpcode() != INT_SUB_opcode)) {
			fail();
		}
		if (!Binary.getVal1(iteratorInstr).similar(carriedLoopIterator)) {
			fail();
		}
		if (!isLoopInvariant(initialIteratorValue, loop, header)) {
			fail();
		}
		strideValue = Binary.getVal2(iteratorInstr);
		if (!(strideValue instanceof ConstantOperand)) {
			fail();
		}
	}

	private void processLoopBlock(BasicBlock block) throws NonCannonicalRegionException {
		checkInEdgesAreInLoop(block);
		checkOutEdgesAreInLoop(block);
	}

	public Operand getCarriedLoopIterator() {
		return carriedLoopIterator;
	}

	private static class NonCannonicalRegionException extends Exception {
		NonCannonicalRegionException() {
			super("Fail to find a cannonical loop region");
		}
	}

	private void fail() throws NonCannonicalRegionException {
		throw new NonCannonicalRegionException();
	}
	
	private BBEnum getBasicBlocks(BasicBlock block, BBEnum bbs, BitVector blocksLeftToVisit) {
		if (block != exit) {
			bbs.add(block);
		}
		blocksLeftToVisit.clear(block.getNumber());
		Enumeration<BasicBlock> successors = block.getNormalOut();
		while (successors.hasMoreElements()) {
			block = successors.nextElement();
			if (blocksLeftToVisit.get(block.getNumber())) {
				getBasicBlocks(block, bbs, blocksLeftToVisit);
			}
		}
		return bbs;
	}

	public Enumeration<BasicBlock> getBasicBlocks() {
		BitVector blocksLeftToVisit = new BitVector(loop);
		BBEnum bbs = getBasicBlocks(header, new BBEnum(), blocksLeftToVisit);
		if (exit != null) {
			bbs.add(exit);
		}
		return bbs;
	}

	static final class BBEnum implements Enumeration<BasicBlock> {
		private final ArrayList<BasicBlock> blocks;

		private int currentBlock;

		BBEnum() {
			blocks = new ArrayList<BasicBlock>();
		}

		public void add(BasicBlock block) {
			blocks.add(block);
		}

		@Override
		public boolean hasMoreElements() {
			return currentBlock < blocks.size();
		}

		@Override
		public BasicBlock nextElement() {
			BasicBlock result = blocks.get(currentBlock);
			currentBlock++;
			return result;
		}

		@Override
		public String toString() {
			return blocks.toString();
		}
	}
}
