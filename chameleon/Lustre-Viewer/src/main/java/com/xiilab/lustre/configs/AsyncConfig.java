package com.xiilab.lustre.configs;

import java.util.concurrent.Executor;

import javax.annotation.Resource;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.aop.interceptor.AsyncUncaughtExceptionHandler;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.scheduling.annotation.AsyncConfigurer;
import org.springframework.scheduling.annotation.EnableAsync;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;

import com.xiilab.lustre.async.AsyncExceptionHandler;

@Configuration
@EnableAsync
public class AsyncConfig implements AsyncConfigurer{
	
	static final Logger LOGGER = LoggerFactory.getLogger(AsyncConfig.class);
	
	/** 샘플 기본 Thread 수 */
    private static int TASK_SAMPLE_CORE_POOL_SIZE = 10;
    /** 샘플 최대 Thread 수 */
    private static int TASK_SAMPLE_MAX_POOL_SIZE = 10;
    /** 샘플 QUEUE 수 */
    private static int TASK_SAMPLE_QUEUE_CAPACITY = 0;
    /** 샘플 Thread Bean Name */
    private static String EXECUTOR_SAMPLE_BEAN_NAME = "executorLustreManager";
    /** 샘플 Thread */
    @Resource(name = "executorLustreManager")
    private ThreadPoolTaskExecutor executorLustreManager;
    



    /**
     * 샘플 Thread 생성
     *
     * @return
     */
    @Bean(name = "executorLustreManager")
	@Override
	public Executor getAsyncExecutor() {
    	ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
        executor.setCorePoolSize(TASK_SAMPLE_CORE_POOL_SIZE);
        executor.setMaxPoolSize(TASK_SAMPLE_MAX_POOL_SIZE);
        executor.setQueueCapacity(TASK_SAMPLE_QUEUE_CAPACITY);
        executor.setBeanName(EXECUTOR_SAMPLE_BEAN_NAME);
        executor.initialize();
        return executor;
    }
    

    
    
    /**
     * 샘플 Thread 등록 가능 여부
     *
     * @return 실행중인 task 개수가 최대 개수(max + queue)보다 크거나 같으면 false
     */
    public boolean isSampleTaskExecute() {
        boolean rtn = true;
 
        LOGGER.info("EXECUTOR_SAMPLE.getActiveCount ==> {}",executorLustreManager.getActiveCount());
        
        // 실행중인 task 개수가 최대 개수(max + queue)보다 크거나 같으면 false
        if (executorLustreManager.getActiveCount() >= (TASK_SAMPLE_MAX_POOL_SIZE + TASK_SAMPLE_QUEUE_CAPACITY)) {
            rtn = false;
        }
 
        return rtn;
    }
 
    /**
     * 샘플 Thread 등록 가능 여부
     *
     * @param createCnt : 생성 개수
     * @return 실행중인 task 개수 + 실행할 개수가 최대 개수(max + queue)보다 크면 false
     */
    public boolean isSampleTaskExecute(int createCnt) {
        boolean rtn = true;
 
        // 실행중인 task 개수 + 실행할 개수가 최대 개수(max + queue)보다 크거나 같으면 false
        if ((executorLustreManager.getActiveCount() + createCnt) > (TASK_SAMPLE_MAX_POOL_SIZE + TASK_SAMPLE_QUEUE_CAPACITY)) {
            rtn = false;
        }
 
        return rtn;
    }

	@Override
	public AsyncUncaughtExceptionHandler getAsyncUncaughtExceptionHandler() {
		return new AsyncExceptionHandler();
	}

}
