package com.xiilab.websocket;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Semaphore;

public class ThreadHelper {
	private static final Semaphore uiSemaphore = new Semaphore(1);
    private static final ExecutorService singleExecutorService = Executors.newSingleThreadExecutor();


    private static void releaseUiSemaphor() {
        singleExecutorService.submit(() -> {
            uiSemaphore.release();
        });
    }

    public static void start(Runnable runnable) {
        Thread thread = new Thread(runnable);
        thread.start();
    }

    public static void sleep(int millis) {
        try {
            Thread.sleep(millis);
        } catch (InterruptedException e) {
//            e.printStackTrace();
        }
    }
}
