package org.grvm.dispatch;

public final class GPUKernelMethod {

	public final int id;
	public final byte[] ptxcode;
	public final String entry;
	public final byte[] paramTypes;

	public GPUKernelMethod(int id, String entry, 
			byte[] ptxcode, byte[] paramTypes) {
		this.id = id;
		this.ptxcode = ptxcode;
		this.entry = entry;
		this.paramTypes = paramTypes;
	}
}
