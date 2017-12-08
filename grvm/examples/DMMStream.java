// import java.util.stream.IntStream;
import org.ptx.dispatch.Dispatcher;


public class DMMStream {
    
    public static void main(String[] args) {
	int N;
	if (args.length > 0) {
	    N = Integer.parseInt(args[0]);
	} else {
	    N = 512;
	}

	DMMStream dmm = new DMMStream(N);

	long ts = System.currentTimeMillis(); 

	/*
	IntStream.range(0, N * N).parallel().forEach(
	    idx -> {
	        int i = idx / N;
	        int j = idx % N;
	        for(int k = 0; k < N; k++) {
	            C[idx] += A[i * N + k] * B[ k * N + j];
	    }
	});
	*/

	Dispatcher.invoke(Dispatcher.getMethod(dmm.getClass(), "test"));

	ts = System.currentTimeMillis() - ts;
	System.out.printf("finished in %5d ms\n", ts);
    }

    int N;
    int[] A,B,C;

    public DMMStream(int n) {
	N = n;
	A = new int[N*N];
        B = new int[N*N];
	C = new int[N*N];
    }

    public void test() {
	for(int i = 0; i < N; i++) {
	    for(int j = 0; j < N; j++) {
		for(int k = 0; k < N; k++) {
		    C[i*N+j] += A[i * N + k] * B[ k * N + j];
		}
	    }
	}
    }
	
}
