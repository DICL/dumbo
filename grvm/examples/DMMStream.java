import java.util.stream.IntStream;

public class DMMStream {
	public static void main(String[] args) {
		int N;		
		if (args.length > 0) {
			N = Integer.parseInt(args[0]);
		} else {
			N = 512;
		}
		int[] A = new int[N*N];
		int[] B = new int[N*N];
		int[] C = new int[N*N];

		System.out.printf("==== begin DMMStream N = %d ====\n", N);
		long ts = System.currentTimeMillis(); 
		IntStream.range(0, N * N).parallel().forEach(
				idx -> {
					int i = idx / N;
					int j = idx % N;
					for(int k = 0; k < N; k++) {
						C[idx] += A[i * N + k] * B[ k * N + j];
					}
		});

		ts = System.currentTimeMillis() - ts;
		System.out.printf("==== finished DMMStream in %d ms ====\n", ts);
	}
}
