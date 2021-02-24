package sim;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

 
public class BSPSyncExecutor extends Items {
	
	static DecimalFormat df = new DecimalFormat("#.000000");
	static Cluster C1; // early cluster
	static Cluster C2; // late cluster	
	static int sync_task_counter = 0;
	static List<Worker> toRun;
	static double max = 0;
	static double run_prev_ = 0;
	static double run_now_ = 0;
	static double quorum_ = 0;
	static double COMMUNICATION_COST = 0;

	//==================TERMINAL PARAMS===================
	public static int INIT_DEVICES; //devices initialized at startup
	static int SIMULATION_ROUNDS; // simulation rounds
	static String OUTFILE1; // inter-sync time

	
	//================ECLIPSE PARAMS
//	public static int INIT_DEVICES = 20; //devices initialized at startup
//	static int SIMULATION_ROUNDS = 3; // simulation rounds
//	static String OUTFILE1 = "output.txt"; // inter-sync time
	

	// create devices
	public static void createInitDevices() {
		
		System.out.println("Initializing machines");
		for (int i = 1; i <= INIT_DEVICES; i++) {
			String name = "D" + (i);
			double var_s_ = 0;
			double var_a_ = 0;
			if (GENERATE_TAG == 0) {
				var_s_ = getDev(SMALL_VARIANCE);
				var_a_ = getDev(SMALL_VARIANCE);
				
			}else if (GENERATE_TAG == 1) {
				var_s_ = getDev(MEDIUM_STD);
				var_a_ = getDev(MEDIUM_STD);
			}else if(GENERATE_TAG == 2) {
				var_s_ = getDev(LARGE_STD);
				var_a_ = getDev(LARGE_STD);
			}
			
			Worker d = new Worker(name, i, var_s_, var_a_); // worker name starting from 1
			initMachines.add(d); // add all workers to list
		}		
		readyMachines = new ArrayList<>(initMachines);
		
		bsp_scheduler_();
	}
	
	// loop through tasks
	public static void bsp_scheduler_() {

		taskQueue = generate();
		localTaskQueue = generateLocal();
		log(taskQueue.size() + " tasks.");
		int z = 0;
		while (z < SIMULATION_ROUNDS) { 
			run_prev_ = getMax(getCurrentTimes(readyMachines));
			log("Simulation round " + (z + 1));
			sync_task_counter = 0;
			for (Task t: taskQueue) {
				readyMachines.sort(Comparator.comparing(d -> d.getTime()));
				log("====================================== ");
				if (t.type == TAG_C_ASYNC) {
					scheduleAsyncTask(t);
				} else if (t.type == TAG_C_SYNC) {						
					COMMUNICATION_COST = (getTaskGaussian(WORKER_CONTROLLER_COMMUNICATION, 2) * 2) +
								((INIT_DEVICES -1) * LOCAL_COMMUNICATION);
				
					
					scheduleSyncTask(t);
					sync_task_counter++;
				}				
			}
			
			for (Task local: localTaskQueue) {
				readyMachines.sort(Comparator.comparing(d -> d.getTime()));
				scheduleLocalTask(readyMachines, local);
			}
			
			run_now_ = getMax(getCurrentTimes(readyMachines));
			z++;
			double runtime = (run_now_ - run_prev_);
			log("Runtime per sync point is:" + runtime + " Comm cost is: " + COMMUNICATION_COST);
			writer(df.format(runtime) + " " + df.format(COMMUNICATION_COST), OUTFILE1);
			
			readyMachines.forEach(d ->{
				d.async_deviation = getDev(SMALL_VARIANCE);
				d.sync_deviation = getDev(SMALL_VARIANCE);
			});
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
		log ("Running sync task");
		readyMachines.forEach(d -> d.setTime(getMax(getCurrentTimes(readyMachines))));
		for (Worker worker: readyMachines) {
			double time = 0;
			if (worker.timing == true) time = t.early_mean;
			else time = t.late_mean;
			runTask(worker, t, (time+(worker.sync_deviation)/4), 1); 
		}
		
		readyMachines.forEach(d -> d.increaseTime(2));
		
	}	
	
	public static void runTask(Worker worker, Task t, double runtime, int type) {
		
		if (type == 2) {
			log(t.getName() + " was run on " + worker.name + " at: "
		+ df.format(worker.time) + " for: " + df.format(runtime) + " and ended: "
					+ df.format(worker.time + runtime));
			worker.increaseTime(runtime);	
		}else if (type == 1) {
			log(t.getName() + " was run on " + worker.name + " at: "
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
	
	public static void main(String[] args) {
		
		try {
			INIT_DEVICES = Integer.parseInt(args[0]);
			SIMULATION_ROUNDS = Integer.parseInt(args[1]);
			OUTFILE1 = args[2];
			SMALL_VARIANCE = Double.parseDouble(args[3]);
			LOCAL_COMMUNICATION = Double.parseDouble(args[4]);
			WORKER_CONTROLLER_COMMUNICATION = Double.parseDouble(args[5]);
			
		} catch (Exception e) {
			System.out.println("Invalid input(s)");
		}
		
		// initializations
		initMachines = new ArrayList<>();
		
		// method calls
		createInitDevices();

		log("Simulation ended");
	}

}
