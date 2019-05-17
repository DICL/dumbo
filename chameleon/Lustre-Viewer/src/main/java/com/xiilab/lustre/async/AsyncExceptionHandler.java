package com.xiilab.lustre.async;

import java.lang.reflect.Method;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.aop.interceptor.AsyncUncaughtExceptionHandler;

public class AsyncExceptionHandler implements AsyncUncaughtExceptionHandler {
	
	static final Logger LOGGER = LoggerFactory.getLogger(AsyncExceptionHandler.class);
	
	@Override
	public void handleUncaughtException(Throwable throwable, Method method, Object... obj) {
		
		LOGGER.error("==============>>>>>>>>>>>> Async ERROR");
		LOGGER.error("Exception Message :: {}",throwable.getMessage());
		LOGGER.error("Method Name :: {}", method.getName());
		for (Object param : obj) {
			LOGGER.error("Parameter Value :: {}" , param);
        }
        // JOB_LOG : 종료 입력
        // ...
		LOGGER.error("==============>>>>>>>>>>>> Async ERROR END");

	}

}
