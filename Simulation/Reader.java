package sim;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

public class Reader {

	static List<Double> values;
	static List<Integer> points;
	static int sum;
	static int count;

	public static void readInput() throws FileNotFoundException, IOException {

		values = new ArrayList<>();
		points = new ArrayList<>();
		try (BufferedReader reader = new BufferedReader(new FileReader("target.txt"))) {

			String sCurrentLine;
			String[] splits; 

			while ((sCurrentLine = reader.readLine()) != null) {
				// remove trailing spaces
				sCurrentLine = sCurrentLine.trim();
				splits = sCurrentLine.split(",");
				int timeid = Integer.parseInt(splits[0]);
				double std = Double.parseDouble(splits[1]);

				values.add(std);
				points.add(timeid);
			}
			reader.close(); 
		}

	}

	public static Stream<String> reader() {
		Path filePath = Paths.get("/Users/laric/Documents", "data.txt");
		Stream<String> lines = null;
		// try-with-resources
		try {
			lines = Files.lines(filePath);
			lines.forEach(System.out::println);
		} catch (IOException e) {
			e.printStackTrace();
		}

		return lines;
	}

	public static void writer() {
		try {
			String content = "Moving up and down!!\n";
			Files.write(Paths.get("/Users/laric/Documents/output.txt"), 
					content.getBytes(), StandardOpenOption.APPEND);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public static void writer1() {

		for (int i = 0; i < 10; i++) {
			Double time = Math.random();
			String content = time.toString() + "\n";
			try {
				Files.write(Paths.get("/Users/laric/Documents/out.txt"), 
						content.getBytes(), StandardOpenOption.APPEND);
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}
	
	public static void delete() {
		try {
			new FileWriter("/Users/laric/Documents/out.txt", false).close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
