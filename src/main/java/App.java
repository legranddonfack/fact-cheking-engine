import java.io.IOException;

public class App {
	public static void main(String[] args) {
        /*if (args.length < 4) {
            System.out.println("Usage: java FactCheckRunner <data_dir> <train_file> <test_file> <output_file>");
            System.out.println("Example: java FactCheckRunner ./data ./data/train.nt ./data/test.nt ./results.ttl");
            return;
        }*/

		String dataDir = "./data"; //args[0];
		String trainFile = dataDir + "/fokg-sw-train-2024.nt"; // args[1];
		String testFile = dataDir + "/fokg-sw-test-2024.nt"; // args[2];
		String outputFile = "./results.ttl"; // args[3];

		try {
			FactChecker checker = new FactChecker(dataDir);

			// Train on labeled data
			System.out.println("Training on: " + trainFile);
			checker.train(trainFile);

			// Process test data
			System.out.println("Processing test data: " + testFile);
			checker.processTestFile(testFile, outputFile);

			System.out.println("Results written to: " + outputFile);

		} catch (IOException e) {
			System.err.println("Error: " + e.getMessage());
			e.printStackTrace();
		}
	}
}
