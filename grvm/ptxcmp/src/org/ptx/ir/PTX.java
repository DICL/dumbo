/*
 * JITPTX Project.
 * 
 * @author Byeongcheol Lee 
 */
package org.ptx.ir;

public class PTX {
	public static enum PTXRegType {
		b8, b16, b32, b64,
		u8, u16, u32, u64,
		s8, s16, s32, s64,
		f16, f16x2, f32, f64,
		pred,
	}
	public static enum PTXComOP {
		eq, ne, lt, le, gt, ge, 
		lo, ls, hi, hs,
		equ, neu, ltu, leu, gtu, geu, num, nam
	}

}
