package sim;

import java.io.BufferedReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

 
public class FlexibleFastSyncExecutor extends Items {
	
	static DecimalFormat df = new DecimalFormat("#.000000");
	static Cluster C1; // early cluster
	static Cluster C2; // late cluster	
	static int task_counter = 0;
	static List<Double> prev = new ArrayList<Double>();
	static List<Worker> toRun;
	static double max = 0;
	static int sync_success1 = 0;
	static int sync_success2 = 0;
	static int sync_success3 = 0;
	static int sync_fail = 0;
	static double sync_now = 0;
	static double sync_next = 0;
	static double run_prev_ = 0;
	static double run_now_ = 0;
	static double quorum_ = 0;


	//==================TERMINAL PARAMS===================
	public static int INIT_DEVICES; //devices initialized at startup
	static int SIMULATION_ROUNDS; // simulation rounds
	static String OUTFILE1; // inter-sync time
	static String OUTFILE2; // success/failure
    static String OUTFILE3; // quorum
	
	//================ECLIPSE PARAMS
//	public static int INIT_DEVICES = 20; //devices initialized at startup
//	static int SIMULATION_ROUNDS = 2; // simulation rounds
//	static String OUTFILE1 = "output.txt"; // inter-sync time
//	static String OUTFILE2 = "clus.txt"; // success/failure
//	static String OUTFILE3 = "quorum.txt"; // participation
	

	// create devices
	public static void createInitDevices() {
		
		System.out.println("Initializing machines");
		for (int i = 1; i <= INIT_DEVICES; i++) {
			String name = "D" + (i);
			double var_s_ = 0;
			double var_a_ = 0;
			if (GENERATE_TAG == 0) {
				var_s_ = getDev(SMALL_STD);
				var_a_ = getDev(SMALL_STD);
			}else if (GENERATE_TAG == 1) {
				var_s_ = getDev(MEDIUM_STD);
				var_a_ = getDev(MEDIUM_STD);
			}else if(GENERATE_TAG == 2) {
				var_s_ = getDev(LARGE_STD);
				var_a_ = getDev(LARGE_STD);
			}
			
			Worker d = new Worker(name, i, var_s_, var_a_); // worker name starting from 1
			initMachines.add(d); // add all workers to list
			// unusedDevices.forEach(d -> d.addCluster(NONE));
			clusterDeviation.add(0.0);
			prev.add(0.0);
		}	
		
		//initMachines.forEach(d->System.out.println(d.timing));
		readyMachines = new ArrayList<>(initMachines);
		
		fast_scheduler_();
	}
	
	// loop through tasks
	public static void fast_scheduler_() {
		
		// generate tasks
		taskQueue = generate();
		// generate local tasks
		
		log(taskQueue.size() + " tasks.");
		int z = 0;
		// loop through each round
		while (z < SIMULATION_ROUNDS) {
			// generate local task queue for current simulation run
			localTaskQueue = generateLocal();
			// get machine time at beginning of iteration
			run_prev_ = getMax(getCurrentTimes(readyMachines));
			log("Simulation round " + (z + 1));
			// loop through tasks
			for (int k = 0; k < taskQueue.size(); k++) {
				Task t = taskQueue.get(k);
				log("====================================== ");
				// check task type
				if (t.type == TAG_C_ASYNC) {
					scheduleAsyncTask(t);
					// report task progress during async operations
					if (k + 1 < taskQueue.size() - 1) {
						if (taskQueue.get(k + 1).type != 2)
							reportProgress();
					}
					
					if (z == 0 & k == 1) {
						getClustering();
					}
				} else if (t.type == TAG_C_SYNC) {
					scheduleSyncTask(t);

				}
				task_counter++;
			}
			
			for (Task local: localTaskQueue) {
				readyMachines.sort(Comparator.comparing(d -> d.getTime()));
				scheduleLocalTask(readyMachines, local);
			}
			if (z == 0) {
				readyMachines.forEach(d->d.increaseTime(CLUSTER_RECONFIGURATION));
			}
			
			run_now_ = getMax(getCurrentTimes(readyMachines));
			z++;
			double runtime = (run_now_ - run_prev_)/2;
			log("Runtime per sync point is:" + runtime);
			writer(df.format(runtime), OUTFILE1);
			
		}
	} 
	
	// local task scheduler
	public static void scheduleLocalTask(List<Worker> w, Task t) {
		log("Running local task");
		for (Worker worker: w) {
			double time = 0;
				if (RANDOM.nextDouble() >= 0.5) time = t.early_mean;
				else time = t.late_mean;
				runTask(worker, t, getLocalGaussian(time), 0);
		}		
	}
	
	// asynchronous task scheduler
	public static void scheduleAsyncTask(Task t) {
		log ("Running async task");
		for (Worker worker: readyMachines) {
			double time = 0;
			if (worker.timing == true) time = t.early_mean;
			else time = t.late_mean;
			runTask(worker, t, (time+worker.async_deviation), 2);
		}		
	}
	
	//======================FAST SYNCHRONIZTION ALGORITHM===========================
	
	//sync task operation
	public static void scheduleSyncTask(Task t) {
		getSyncOptions(t);
	}	
	
	public static void runTask(Worker worker, Task t, double runtime, int type) {
		
		if (type == 2) {
			log("Async task was run on " + worker.name + " at: "
		+ df.format(worker.time) + " for: " + df.format(runtime) + " and ended: "
					+ df.format(worker.time + runtime));
			worker.increaseTime(runtime);	
		}else if (type == 1) {
			log(t.getName() + " was run at sync point one on " + worker.name + " at: "
		+ df.format(worker.time) + " for: " + df.format(runtime) + " and ended: "
					+ df.format(worker.time + runtime));
			worker.increaseTime(runtime);
		}else if (type == 0) {
			log("Local task was run on " + worker.name + " at: "
		+ df.format(worker.time) + " for: " + df.format(runtime) + " and ended: "
					+ df.format(worker.time + runtime));
			worker.increaseTime(runtime);
		}
	}
	
	public static void reportProgress() {
		
		List<Double> devs = new ArrayList<Double>();
		List<Double> new_devs = new ArrayList<Double>();
		List<Double> norm = new ArrayList<Double>();		
		List<Double> temp_addition = new ArrayList<Double>();
		
		// sort devices by id
		readyMachines.sort(Comparator.comparing(d -> d.getId()));
		
		// get times from machines after running a-sync task
		readyMachines.forEach(w -> devs.add(w.getTime()));
		
		// time to be used for predicting sync options
		for (int k = 0; k < readyMachines.size(); k++) {
			readyMachines.get(k).setClusterTime(devs.get(k));
		}
		
		// get runtime in current period
		new_devs = listOperator(devs, prev, "-");
		
		// pivot worker value
		double normalize = new_devs.get(0);
				
		// normalize deviations using first worker 
		new_devs.forEach(d -> norm.add(d - normalize));
		
		temp_addition = listOperator(norm, clusterDeviation, "+");
		
		// update cluster deviation
		clusterDeviation = temp_addition;
		
		// set previous time to compare with new time
		prev = devs;
						
	}
	
	// sync step 1
	public static void getClustering() {
		
		// clear cluster details file
		try {
			new FileWriter("/Users/laric/Documents/eclipse-workspace/FastSynchronization/clus.txt",
					false).close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		for (Double time: clusterDeviation) {
			// get average deviation per task
			String content = df.format(time/task_counter) + "\n";
			//System.out.print(content);
			try {
				Files.write(Paths.get(
						"/Users/laric/Documents/eclipse-workspace/FastSynchronization/clus.txt"), 
						content.getBytes(), StandardOpenOption.APPEND);
			} catch (IOException e) {
				e.printStackTrace();
			}
		}	
		
		runPythonClustering();
		 
	}
	
	// sync step 2
	public static void runPythonClustering() {
		String s = null;
		List<String> stdIn = new ArrayList<String>();

        try {
        	
        	String command = "/usr/local/opt/python/libexec/bin/python java.py";
            
            Process p = Runtime.getRuntime().exec(command);
            
            BufferedReader stdInput = new BufferedReader(new 
                 InputStreamReader(p.getInputStream()));

            BufferedReader stdError = new BufferedReader(new 
                 InputStreamReader(p.getErrorStream()));

            // read the output from the command
            log("Here is the clustering output:\n");
            while ((s = stdInput.readLine()) != null) {
                log(s);
                stdIn.add(s);
            }
            
            // read any errors from the attempted command
            log("Here is the clustering error (if any):\n");
            while ((s = stdError.readLine()) != null) {
                log(s);
            }
        }
        catch (IOException e) {
            log("exception happened - here's what I know: ");
            e.printStackTrace();
            System.exit(-1);
        }
        
        int num_clusters_ = Integer.valueOf(stdIn.get(0).split(",")[1]);
        int noise_ = Integer.valueOf(stdIn.get(1).split(",")[1]);
        
        String [] clus1_  = stdIn.get(2).split(",");
        String [] clus2_  = stdIn.get(3).split(",");
        
        int c1 = clus1_.length;
        int c2 = clus2_.length;
        
        double quorum_ = (double)(c1 + c2) / readyMachines.size();
        
        log("Num clusters: " + num_clusters_ + " Noise: " + noise_  + " Clustered: " + quorum_);
        
        List<Worker> cluster_a = new ArrayList<Worker>();
        List<Worker> cluster_b = new ArrayList<Worker>();
        
        List<Double> d_a = new ArrayList<Double>();
        List<Double> d_b = new ArrayList<Double>();
        
        
        // add workers to cluster 1
        for (String st1 : clus1_) {
        	for (Worker wk1 : readyMachines) {
        		if (wk1.getId() == Integer.parseInt(st1.strip())) {
        			//wk1.setCurrentCluster(C1);
        			//C1.addDevice(wk1);
        			cluster_a.add(wk1);
        			d_a.add(wk1.getTime());
        		} 
        	}
        }
        
        // add clusters to cluster 2
        for (String st2 : clus2_) {
        	//log(st2.strip());
        	for (Worker wk2 : readyMachines) {
        		//log(wk2.getName());
        		if (wk2.getId()  == Integer.parseInt(st2.strip())) {
        			//log("true");
        			//wk2.setCurrentCluster(C2);
        			//C2.addDevice(wk2);
        			cluster_b.add(wk2);
        			d_b.add(wk2.getTime());
        		}
        	}
        }
        // clear existing device lists
        C1.clearDeviceList();
        C2.clearDeviceList();
        
        // make sure C1 gets early cluster
        if (getMean(d_a) < getMean(d_b)) {
        	cluster_a.forEach(d ->{
        		C1.addDevice(d);
        		d.setCurrentCluster(C1);
        	});
        	
        	cluster_b.forEach(d ->{
        		C2.addDevice(d);
        		d.setCurrentCluster(C2);
        	});
        }else {
        	cluster_a.forEach(d ->{
        		C2.addDevice(d);
        		d.setCurrentCluster(C2);
        	});
        	
        	cluster_b.forEach(d ->{
        		C1.addDevice(d);
        		d.setCurrentCluster(C1);
        	});
        }
        //checkClusters();        
	}
	
	// sync step 3a
	public static void getSyncOptions(Task t) {
		
		// minimum time - used to know sync lag time
		sync_now = getMin(getCurrentTimes(readyMachines));
		
		// fixing first sync point 
		List<Double> tempList2_ = new ArrayList<>();
		// get cluster 2 machine times 
		for (Worker work: C2.getDevices()) {
			tempList2_.add(work.getClusterTime());
			//log(work.getName());
		}
		// mean and std of late cluster	for first sync point	
		double ts1_mean_late_clus_ = getMean(tempList2_); 
		log("Mean is: " + ts1_mean_late_clus_);
		double ts1_std_late_clus_ = getStd(tempList2_);
		log("Standard deviation is: " + ts1_std_late_clus_);
		
		// ts1
		double ts1 = ts1_mean_late_clus_ +  ts1_std_late_clus_ + t.early_mean;
		log("ts1 is: " + ts1);
		
		if (quorumMet(ts1)) {
			sync_success1++;
			runSyncOptionOne(t, toRun, ts1);
			//writer(df.format(ts1 - early_avail),out50 );
			log(df.format(ts1) + " vs " +  sync_now);
		}else {
			
			// add communication cost for 3 late notifications
			C2.getDevices().forEach(d->d.increaseTime(LOCAL_COMMUNICATION)); 
			// quorum failed for ts1......proceed to ts2
			syncOptionTwo(t);
		}
	}
	
	// sync step 4
	public static boolean quorumMet(double q_time) {	
		toRun = new ArrayList<>();
		int q_counter_ = 0;
		// check workers that are available before ts1
		for (Worker w: readyMachines) {
			if (w.getTime() <= q_time) {
				q_counter_++;
				toRun.add(w);
			}			
		}
		quorum_ = (double) (q_counter_ * 1.0) / (double) (readyMachines.size() * 1.0);
		log("Quorum is " + quorum_);
		
		if (quorum_ > 0.7) {
			writer(df.format(quorum_), OUTFILE3);
			return true;
		}
		else return false;
	}
	
	// sync step 4 --- run actual sync task
	public static void runSyncOptionOne(Task t, List<Worker> availableWorkers, double ts1) {
		//update available machine times
		availableWorkers.forEach(d -> d.setTime(ts1)); 
		// run sync task
		log("Running sync task at: " + ts1);
		// run sync tasks at ts1
		availableWorkers.forEach(d ->{
			double time = 0;
			if (d.timing == true) time = t.early_mean;
			else time = t.late_mean;
			runTask(d, t, (time+(d.sync_deviation)/4), 1);
		});
		// update remaining worker times		
		List<Worker> rem = new ArrayList<Worker>();
		rem = returnListDifference(readyMachines, toRun);

		rem.forEach(d -> d.setTime(getMax(getCurrentTimes(toRun))));		
	}
	
	// step 3b
	public static void syncOptionTwo(Task t) {
		
		// check if to run local task
		log("Attempting second sync option");
		List<Double> tempList1_ = new ArrayList<>();
		List<Double> tempList2_ = new ArrayList<>();
		
		for (Worker work: C2.getDevices()) {
			tempList2_.add(work.getTime());
		}
		for (Worker work: C1.getDevices()) {
			tempList1_.add(work.getTime());
		}
				
		// mean of early cluster
		double mean_clus_one = getMean(tempList1_);
		double std_clus_one = getStd(tempList1_);
		double potential_local_task_finish_time = mean_clus_one +  std_clus_one + 3 + 1;
		
		// late cluster
		double mean_clus_two = getMean(tempList2_);
		double std_clus_two = getStd(tempList2_);
		double potential_finish_time_clus_two = mean_clus_two + std_clus_two;
		
		//double finish_time_clus_two = getMax(tempList2_);
		log("Potential finish time of local task is " + potential_local_task_finish_time);
		log("Cluster two finish time is " + potential_finish_time_clus_two);
		
		// check if local task can be scheduled on fast cluster
		if(potential_local_task_finish_time <= potential_finish_time_clus_two) {
			Task task = localTaskQueue.get(0);
			scheduleLocalTask(C1.getDevices(), task);	
			List<Double> temp_ = new ArrayList<>();
			for (Worker work: C1.getDevices()) {
				temp_.add(work.getTime());
			}
			// remove task from queue
			localTaskQueue.remove(task);

			
			//double expected_local_finish = one_local_mean + one_local_std;
			double time = Math.max(potential_finish_time_clus_two, potential_local_task_finish_time);
			// check if quorum met at ts2
			if (quorumMet(time)) {
				sync_success2++;
				log(df.format(time) + " vs " +  sync_now);
				//writer(df.format(time), OUTFILE2);
				runOptionTwoOrThree(t, toRun, time);
			}else {
				C1.getDevices().forEach(d->d.increaseTime(LOCAL_COMMUNICATION));
				syncOptionThree(t);
			}
			
			//readyMachines.forEach(d -> d.setClusterTime(Math.max(finish_time_clus_two, getMax(temp_))));			
		}else {// local task not run
			toRun = new ArrayList<>();
			readyMachines.forEach(d ->{
				if (d.getTime() <= potential_finish_time_clus_two) {
					d.setTime(potential_finish_time_clus_two);
					toRun.add(d);
				}
			});	
			sync_success2++;
			log(df.format(potential_finish_time_clus_two) + " vs " +  sync_now);
			//writer(df.format(finish_time_clus_two), OUTFILE1);
			// run option 2
			runOptionTwoOrThree(t, toRun, potential_finish_time_clus_two);		
		}
	}
	
	// run sync task at ts2 or ts3
	public static void runOptionTwoOrThree(Task t, List<Worker> availableWorkers, double ts2) {
		//update available machine times
		availableWorkers.forEach(d -> d.setTime(ts2));
		// run sync task
		log("Running sync task at: " + ts2);
		availableWorkers.forEach(d ->{
			double time = 0;
			if (d.timing == true) time = t.early_mean;
			else time = t.late_mean;
			runTask(d, t, (time+(d.sync_deviation)/4), 1);
		});
		
		//update remaining machine time
		
		List<Worker> rem = new ArrayList<Worker>();
		rem = returnListDifference(readyMachines, toRun);
		
		rem.forEach(d -> d.setTime(getMax(getCurrentTimes(toRun))));
		
	}
	 
	
	// third sync option
	public static void syncOptionThree(Task t) {
		//scheduleLocalTask(C2.getDevices(), localTaskQueue.get(1));
		List<Double> temp_1 = new ArrayList<>();
		List<Double> temp_2 = new ArrayList<>();
		
		for (Worker work: C1.getDevices()) {
			temp_1.add(work.getTime());
		}
		
		for (Worker work: C2.getDevices()) {
			temp_2.add(work.getTime());
		}
		
		double one_local_mean = getMean(temp_1);
		double one_local_std = getStd(temp_1);
				
		double two_local_mean = getMean(temp_2);
		double two_local_std = getStd(temp_2);
		
		double expected_local_one_finish = one_local_mean + one_local_std;

		
		double expected_local_two_finish = two_local_mean + two_local_std + 3 + 1;
		
		if (expected_local_two_finish <= expected_local_one_finish) {
			scheduleLocalTask(C2.getDevices(), localTaskQueue.get(1));
		}else {
			if (quorumMet(expected_local_one_finish)) {
				sync_success3++;
				//writer(df.format(expected_local_finish), OUTFILE1);
				log(df.format(expected_local_one_finish) + " vs " +  sync_now);
				runOptionTwoOrThree(t, toRun, expected_local_one_finish);
			}
			else {
				sync_fail++;
				log("Sync failed");
			}
		}
		
	}
	
	// tester
	public static void checkClusters() {
		
		List<Worker> initW1 = new ArrayList<Worker>();
		List<Worker> initW2 = new ArrayList<Worker>();
		for (Worker w : readyMachines) {
			if (w.getCurrentCluster() == C1) {
				initW1.add(w);
			}else if (w.getCurrentCluster() == C2) {
				initW2.add(w);
			}
			
			//w.setCurrentCluster(C2);
			
			//log(w.name + ": " + w.getCurrentCluster().c_id);
		}
		System.out.print("Cluster 1: ");
		for (Worker w : initW1) {
			System.out.print(w.name + ", ");
		}
		System.out.print("\n");
		System.out.print("Cluster 2: ");
		for (Worker w : initW2) {
			System.out.print(w.name + ", ");
		}
	}
	
	//tester
	public static void toTest() {
		
		readyMachines.forEach(m -> m.increaseTime(Math.random()));
		//readyMachines.sort(Comparator.comparing(d -> d.getTime()));
		//readyMachines.forEach(m -> log(m.name));
		//readyMachines.sort(Comparator.comparing(d -> d.getId()));
		//readyMachines.forEach(m -> log(m.name));		
	}
	
	// add two lists together or diff two lists
	public static List<Double> listOperator(List<Double> a, List<Double> b, String sign) {

		List<Double> sum = new ArrayList<Double>();
		
		if (sign == "-") {
			for (int i = 0; i < a.size(); i++) {
				sum.add(a.get(i) - b.get(i));
				//log( a.get(i) + " - " + b.get(i));
			}
		}		
		else if (sign == "+") {
			for (int i = 0; i < a.size(); i++) {
				sum.add(a.get(i) + b.get(i));
				//log("Adding " + a.get(i) + " to " + b.get(i));
			}
		}		
		return sum;
	}
	
	public static void generateReport() {
		String content = sync_success1 + " " + sync_success2 + " "  +
				sync_success3 + " "  + sync_fail;
		log(content);
		writer(content, OUTFILE2);
		
	}
	
	public static void main(String[] args) {
		
		try {
			INIT_DEVICES = Integer.parseInt(args[0]);
			SIMULATION_ROUNDS = Integer.parseInt(args[1]);
			OUTFILE1 = args[2];
			OUTFILE2 = args[3];
			OUTFILE3 = args[4];
			GENERATE_TAG = Integer.parseInt(args[5]);
			
		} catch (Exception e) {
			System.out.println("Invalid input(s)");
		}
		
		// initializations
		initMachines = new ArrayList<>();
		clusterDeviation = new ArrayList<>();
		C1 = new Cluster(1); // early cluster
		C2 = new Cluster(2); // late cluster
		
		// method calls
		createInitDevices();
		//toTest();
		//reportProgress();
		//getClustering();
		//checkClusters();
		//scheduler_();
		generateReport();
		log("Simulation ended");
	}

}
