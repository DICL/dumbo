package org.grvm.compiler;

import java.util.LinkedList;

import org.jikesrvm.compilers.opt.ir.Register;

public class GPUKernelInfo {
	final RegionNode loop;
	final LinkedList<Register> params = new LinkedList<Register>();
	public GPUKernelInfo(RegionNode lstNode) {
		this.loop = lstNode;
	}
}
