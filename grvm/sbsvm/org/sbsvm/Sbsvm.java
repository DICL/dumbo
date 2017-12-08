package org.sbsvm;

import java.nio.ByteBuffer;

public class Sbsvm {
    private static Sbsvm instance;

    static {
	System.loadLibrary("sbsvm");
	instance = new Sbsvm();
    }

    private Sbsvm() {
	initialize();
    }
    
    public static Sbsvm getInstance() {
	return instance;
    }

    private native void initialize();
    @Override
    protected native void finalize() throws Throwable;
    public native void clear();
    public native void run();
    public native long loadModule(ByteBuffer image);
    public native long getFunction(long module, ByteBuffer name);
    public native long createStream();
    public native void launchKernel(long function, long gridDimX, long gridDimY, long gridDimZ, long blockDimX, long blockDimY, long blockDimZ, long sharedMemBytes, long stream);
    public native void test(); 
}
