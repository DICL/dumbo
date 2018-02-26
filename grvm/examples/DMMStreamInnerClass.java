import java.util.function.IntConsumer;
import java.util.stream.IntStream;

public class DMMStreamInnerClass {
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

		long ts = System.currentTimeMillis(); 
		IntStream.range(0, N * N).parallel().forEach(new IntConsumer() {
			public void accept(int idx) {
				int i = idx / N;
				int j = idx % N;
				for (int k = 0, n = N; k < n; k++) {
					C[idx] += A[i * n + k] * B[k * n + j];
				}
			}
		});

		ts = System.currentTimeMillis() - ts;
		System.out.printf("finished in %5d ms\n", ts);
	}
}
