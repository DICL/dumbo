package org.grvm.dispatch;

import org.jikesrvm.classloader.RVMMethod;
import org.jikesrvm.classloader.TypeReference;
import org.jikesrvm.runtime.EntrypointHelper;

public class GPUEntrypoints {

	public static final TypeReference typeRefGPUKernelMethod =
			TypeReference.findOrCreate(GPUKernelMethod.class);

	public static final TypeReference typeRefParamObject =
			TypeReference.findOrCreate(ParamObject.class);

	public static final TypeReference typeRefInteger =
			TypeReference.findOrCreate(Integer.class);

	public static final RVMMethod integerValueofmethod =
			EntrypointHelper.getMethod(
					"Ljava/lang/Integer;",
					"valueOf",
					"(I)Ljava/lang/Integer;");

	public static final RVMMethod paramObjectValueofMethod =
			EntrypointHelper.getMethod(
					"Lorg/grvm/dispatch/ParamObject;",
					"valueOf", 
					"(Ljava/lang/Object;)Lorg/grvm/dispatch/ParamObject;");

	public static final RVMMethod dispatcherInvokeMethod =
			EntrypointHelper.getMethod(
					"Lorg/grvm/dispatch/Dispatcher;",
					"invoke", 
					"(IILjava/util/function/IntConsumer;)V");

	public static final RVMMethod shouldOffloadMethod = 
			EntrypointHelper.getMethod(
					"Lorg/grvm/dispatch/Dispatcher;", 
					"shouldOffload", "()Z");

	public static final RVMMethod getGPUKernelMethod = 
			EntrypointHelper.getMethod(
					"Lorg/grvm/dispatch/GPUKernelMethods;",
					"getMethod", 
					"(I)Lorg/grvm/dispatch/GPUKernelMethod;");

	public static final RVMMethod launchGPUKernelMethod =
			EntrypointHelper.getMethod(
					"Lorg/grvm/dispatch/Dispatcher;",
					"launch", 
					"(Lorg/grvm/dispatch/GPUKernelMethod;II[Ljava/lang/Object;)V");

	static {
		ensurePrecompile(integerValueofmethod);
		ensurePrecompile(paramObjectValueofMethod);
		ensurePrecompile(shouldOffloadMethod);
		ensurePrecompile(getGPUKernelMethod);
		ensurePrecompile(launchGPUKernelMethod);
	}

	private static void ensurePrecompile(RVMMethod m) {
		if (m.isCompiled()) {
			return;
		}
		m.compile();
		m.getDeclaringClass().updateMethod(m);
	}
}
