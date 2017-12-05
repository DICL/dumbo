/**
  * Copyright 2015 José Manuel Abuín Mosquera <josemanuel.abuin@usc.es>
  * 
  * This file is part of BigBWA.
  *
  * BigBWA is free software: you can redistribute it and/or modify
  * it under the terms of the GNU General Public License as published by
  * the Free Software Foundation, either version 3 of the License, or
  * (at your option) any later version.
  *
  * BigBWA is distributed in the hope that it will be useful,
  * but WITHOUT ANY WARRANTY; without even the implied warranty of
  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  * GNU General Public License for more details.
  * 
  * You should have received a copy of the GNU General Public License
  * along with BigBWA. If not, see <http://www.gnu.org/licenses/>.
  */
  
public class BwaJni {

	private native int bwa_jni(int argc, String[] argv, int[] lenStrings);
	
	public static void Bwa_Jni(String[] args) {
		
		int[] lenStrings = new int[args.length];
		
		int i=0;
		
		for(String argumento: args){
			lenStrings[i] = argumento.length();
		}
		
		new BwaJni().bwa_jni(args.length, args, lenStrings);
	}
	static {
		System.loadLibrary("bwa");
	}
}

