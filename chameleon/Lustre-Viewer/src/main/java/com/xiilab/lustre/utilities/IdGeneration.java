package com.xiilab.lustre.utilities;

import java.util.Random;

public class IdGeneration {
	private static final String CHARS
	="0123456789"
	+"abcdefghijkmnopqrstuvwxyz"
	+"ABCDEFGHIJKMNOPQRSTUVWXYZ";

	
	public static String generateId(int length) {
		char[] chars  = new char[length];
		
		Random random = new Random();
		
		for (int i = 0; i < length; i++) {
			chars[i] = CHARS.charAt(random.nextInt(CHARS.length()));
		}
		
		return new String(chars);
	}
}
