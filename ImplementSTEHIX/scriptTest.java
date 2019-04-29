package client;

import java.io.FileReader;

import javax.script.ScriptEngine;
import javax.script.ScriptEngineManager;
import javax.script.ScriptException;

public class scriptTest {
	public scriptTest(){
		ScriptEngineManager scriptEngineMgr = new ScriptEngineManager();
		ScriptEngine jsEngine = scriptEngineMgr.getEngineByName("JavaScript");
		
		if(jsEngine == null){
			System.exit(1);
		}
		
		try {
			Object mapApi = jsEngine.eval(new FileReader("https://openapi.map.naver.com/openapi/v3/maps.js?clientId=Tqi19MAoi6IOugovdau9"));
			Object map1 = jsEngine.eval("new naver.maps.Map('map', {center: new naver.maps.LatLng(37.6132884, 127.1394276), zoom:11});");
			Object map2 = jsEngine.eval("new naver.maps.Marker({position: new naver.maps.LatLng(37.6132884, 127.1394276), map: map});");
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
