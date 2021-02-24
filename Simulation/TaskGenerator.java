package sim;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class TaskGenerator  {

	// static final double MEAN = 5;
	static final double VARIANCE = 1.2;
	static final long SEED = 2;
	static final Random RANDOM = new Random(SEED);
	static final Random RAND = new Random();
	static List<Task> taskList;
	static List<Task> localTaskList;
	static int task_index[] = {1, 1, 2, 1, 1, 2};
	//static int task_index[] = {1, 1, 2};

	public static double getGaussian(double aMean) {
		double time = aMean + RANDOM.nextGaussian() * VARIANCE;
		if (time <= 0)
			getGaussian(aMean);
		return time;
	}

	public static void log(Object aMsg) {
		System.out.println(String.valueOf(aMsg));
	}
	
	public static int getRandomNumberInRange(int min, int max) {
		if (min > max) {
			throw new IllegalArgumentException("max must be greater than min");
		}
		return RAND.nextInt((max - min) + 1) + min;
	}

	public static List<Task> generate() {
		Task t;
		taskList = new ArrayList<>();

		for (int i = 0; i < task_index.length; i++) {
			t = new Task(task_index[i], 5, 5.5, 1.2);
			// System.out.println(t);
			taskList.add(t);

		}
		//taskList.forEach(elem -> log(elem + " --> " + elem.getName() +
				//" --> " + elem.getEarlyMean() + " --> " + elem.getLateMean()));
		return taskList;

	}
	
	public static List<Task> generateLocal() {
		Task t;
		localTaskList = new ArrayList<>();
		
		for (int i = 0; i < getRandomNumberInRange(2, 3); i++) {
			t = new Task(task_index[i], 0.5, 0.8, 0.2);

			localTaskList.add(t);

		}
		return localTaskList;
		
	}
	
//	public static void main (String [] args) {
//		log(getRandomNumberInRange(1, 4));
//	}


}
