package org.grvm.dispatch;

import static org.grvm.Util.*;

import org.vmmagic.pragma.Entrypoint;

public abstract class GPUKernelMethods {
	static int nextID = 0;
	static final GPUKernelMethod[] kernels = new GPUKernelMethod[128];

	public static final GPUKernelMethod emptyKernelMethod = allocate(
			"kmain",
			".visible .entry kmain(){ret;}".getBytes(),
			new byte[0]);

	public static GPUKernelMethod allocate(
			String entry, byte[] ptxcode, byte[] paramTypes) {
		int id = nextID;
		_assert( id < kernels.length);
		GPUKernelMethod gm = new GPUKernelMethod(id, entry, ptxcode, paramTypes);
		kernels[id] = gm;
		nextID++;
		return gm;
	}

	@Entrypoint
	public static GPUKernelMethod getMethod(int id) {
		return kernels[id];
	}
}
