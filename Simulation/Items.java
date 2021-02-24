package sim;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

//class to hold simulation parameters
public class Items {
	
	//from clustering experiment all times in ms
//	static double SMALL_MEAN = 25.1033858768765;
//	static double SMALL_STD = 4.047807741846692;
//	static double MEDIUM_MEAN = 43.5953908874562; 
//	static double MEDIUM_STD = 3.3151579918223866;
//	static double LARGE_MEAN = 80.39536447075561;
//	static double LARGE_STD = 4.753120185957348;
	static double SMALL_MEAN = 25.1033858768765;
//	static double SMALL_STD = 1.047807741846692;
	static double SMALL_VARIANCE;
	static double SMALL_STD = 1.75;
	static double MEDIUM_MEAN = 43.5953908874562; 
	static double MEDIUM_STD = 1.3151579918223866;
	static double LARGE_MEAN = 80.39536447075561;
	static double LARGE_STD = 1.753120185957348;
	static double LOCAL_COMMUNICATION; // ms var 0.3
	static double WORKER_CONTROLLER_COMMUNICATION; // ms var 2
	static final double CLUSTER_RECONFIGURATION = 20; //ms
	static final double CONTROLLER_COMPUTATION = 10; //ms
	static final double OFFSET = 5; //ms
	
	static int CLUSTER_REPS = 0; // 0 for default,  any other number for fixed iteration.
	static int GENERATE_TAG = 0; // 0 for short, 1 for medium and 2 for long.
	
	//tags for identifying tasks
	public static final int TAG_C_ASYNC = 1;
	public static final int TAG_C_SYNC = 2;
	public static final int TAG_C_LOCAL = 3;
	
	//parameters
	//static final double VARIANCE = 1.0; // task runtime variance
	static final double LOCAL_VARIANCE = 0.15;
	//static long SEED = 10; 
	
	
	// lists
	public static List<Worker> initMachines; //initialized devices at startup
	public static List<Worker> readyMachines; //devices ready to run a task
	public static List<Task> taskQueue; // queue for tasks
	public static List<Task> localTaskQueue; // local task queue
	public static List<Double> clusterDeviation; // hold values to cluster
	
	//static final Random RANDOM = new Random(SEED);
	
	static final Random RANDOM = new Random();
	static double VARIANCE;
	static double SMALL_TASK = 5;
	static double BIG_TASK = 5.5;
	static final long SEED = 2;
	//static final Random RANDOM = new Random(SEED);
	static final Random RAND = new Random();
	static List<Task> taskList;
	static List<Task> localTaskList;
	////static int task_index[] = {1, 1, 2, 1, 1, 2};
	static int task_index[] = {1, 1, 2};
	// helper methods

	// returns a random number between minimum and max
	public static int getRandomNumberInRange(int min, int max) {
		if (min > max) {
			throw new IllegalArgumentException("max must be greater than min");
		}
		return RANDOM.nextInt((max - min) + 1) + min;
	}
	
	public static void log(Object aMsg){
		System.out.println(String.valueOf(aMsg));
	}
	
	public static double getTaskGaussian(double aMean, double variance) {
		double time = aMean + RANDOM.nextGaussian() * variance;
		if (time <= 0)
			getTaskGaussian(aMean, variance);
		return time;
	}
	
	public static double getDev(double variance) {
		double time = RANDOM.nextGaussian() * variance;
		if (time <= 0)
			getDev(variance);
		return time;
	}
	
	public static double getGaussian(double aMean) {
		double time = aMean + RANDOM.nextGaussian() * VARIANCE;
		if (time <= 0)
			getGaussian(aMean);
		return time;
	}
	
	public static double getSyncGaussian(double aMean) {
		double time = aMean + RANDOM.nextGaussian() * 0.2;
		if (time <= 0)
			getSyncGaussian(aMean);
		return time;
	}
	
	public static double getLocalGaussian(double aMean) {
		double time = aMean + RANDOM.nextGaussian() * LOCAL_VARIANCE;
		if (time <= 0)
			getLocalGaussian(aMean);
		return time;
	}
	
    public static double sum (List<Double> a){
        if (a.size() > 0) {
            double adder = 0;
 
            for (Double i : a) {
                adder += i;
            }
            return adder;
        }
        return 0;
    }
    public static double getMean (List<Double> a){
        double sum = sum(a);
        double mean = 0;
        mean = sum / (a.size() * 1.0);
        return mean;
    }
    public static double getMedian (List<Double> a){
        int middle = a.size()/2;
 
        if (a.size() % 2 == 1) {
            return a.get(middle);
        } else {
           return (a.get(middle-1) + a.get(middle)) / 2.0;
        }
    }
    public static double getStd (List<Double> a){
        double sum = 0;
        double mean = getMean(a);
 
        for (Double i : a)
            sum += Math.pow((i - mean), 2);
        return Math.sqrt( sum / ( a.size() - 1 ) ); // sample
    }
    
    public static double getMin (List<Double> a){
        double min = 100000000000.0;
 
        for (Double i : a)
            if (i < min) min = i;
        return min; 
    }
    
    public static double getMax (List<Double> a){
        double max = 0;
        
        for (Double i : a)
            if (i > max) max = i;
        return max;
    }
    
    public static List<Double> getCurrentTimes(List<Worker> w){
    	List<Double> lst = new ArrayList<Double>();
    	for (Worker work: w) {
    		lst.add(work.getTime());
    	}  	
    	return lst;
    	
    }
    
	public static void writer(String content, String filePath) {
		content = content + "\n";
		try {
			Files.write(Paths.get(filePath), 
					content.getBytes(), StandardOpenOption.APPEND);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public static List<Worker> returnListDifference(List<Worker> one, List<Worker> two) {
		List<Worker> tempDevices = new ArrayList<Worker>();
		Boolean bool = true;
		for (int i = 0; i < one.size(); i++) {
			for (int j = 0; j < two.size(); j++) {
				if (one.get(i).name == two.get(j).name)
					bool = false;
			}
			if (bool)
				tempDevices.add(one.get(i));
			bool = true;
		}
		return tempDevices;
	}
	
	// generate tasks with two distributions - early and late 
	public static List<Task> generate() {
		Task t;
		double early_dist = 0;
		double late_dist = 0;
		double variance = 0;
		taskList = new ArrayList<>();
		
		if (GENERATE_TAG == 0) {
			early_dist = SMALL_MEAN;
			late_dist = early_dist + 1;
			variance = SMALL_STD;
		}else if (GENERATE_TAG == 1) {
			early_dist = MEDIUM_MEAN;
			late_dist = early_dist + 1;
			variance = MEDIUM_STD;
		}else if(GENERATE_TAG == 2) {
			early_dist = LARGE_MEAN;
			late_dist = early_dist + 1;
			variance = LARGE_STD;
		}

		for (int i = 0; i < task_index.length; i++) {
			t = new Task(task_index[i], early_dist, late_dist, variance);
			// System.out.println(t);
			taskList.add(t);

		}
		//taskList.forEach(elem -> log(elem + " --> " + elem.getName() +
				//" --> " + elem.getEarlyMean() + " --> " + elem.getLateMean()));
		return taskList;

	}
	
	// generate local tasks
	public static List<Task> generateLocal() {
		Task t;
		localTaskList = new ArrayList<>();
		
		for (int i = 0; i < 3; i++) {
			t = new Task(0, 3, 5, 1);

			localTaskList.add(t);

		}
		return localTaskList;
		
	}
	
	
	//.............................................................//
}

