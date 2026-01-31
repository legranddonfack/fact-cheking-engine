package org.sw;

import java.io.IOException;

public class App {
	public static void main(String[] args) {
		System.out.println("args len " + args.length);
		System.out.println("args: ");
		for (String arg : args) {
			System.out.println(arg);
		}
        if (args.length < 4) {
            System.out.println("Usage: java -jar fact-checking.jar <data_dir> <train_file> <test_file> <output_file>");
            System.out.println("Example: java fact-checking-1.0-SNAPSHOT.jar ./data ./data/fokg-sw-train-2024.nt ./data/fokg-sw-test-2024.nt ./result.ttl");
            return;
        }

		String dataDir = args[0];
		String trainFile = args[1];
		String testFile = args[2];
		String outputFile = args[3];

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
