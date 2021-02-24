package sim;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

 
public class DSSPSyncExecutor extends Items {
	
	static DecimalFormat df = new DecimalFormat("#.000000");
	static Cluster C1; // early cluster
	static Cluster C2; // late cluster	
	static Worker slowest;
	static Worker fastest;
	static int sync_task_counter = 0;
	static List<Worker> toRun;
	static double max = 0;
	static double run_prev_ = 0;
	static double run_now_ = 0;
	static double quorum_ = 0;
	static boolean staleness_check = false;
	static double COMMUNICATION_COST = 0;

	// holds last two runtimes for a single worker
	//static Utility.SizedStack<Double> stack_worker = new Utility.SizedStack<Double>(2);
	// A_list for holding previous two runtimes for all workers
	static Utility.FullStack full_stack = new Utility.FullStack();

	//==================TERMINAL PARAMS===================
	public static int INIT_DEVICES; //devices initialized at startup
	static int SIMULATION_ROUNDS; // simulation rounds
	static String OUTFILE1; // inter-sync time
	
	
//	static double STALE_VALUE; //staleness value
//	static double MIN_STALE; // minimum staleness
//	static double MAX_STALE; // maximum staleness

	
	//================ECLIPSE PARAMS
//	public static int INIT_DEVICES = 20; //devices initialized at startup
//	static int SIMULATION_ROUNDS = 3; // simulation rounds
//	static String OUTFILE1 = "output.txt"; // inter-sync time
	
	
//	static double STALE_VALUE; //staleness value
	static final double MIN_STALE = 3; // minimum staleness
	static final double MAX_STALE = 7; // maximum staleness
	static final double STALENESS = 10; // staleness timing
	

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
		
		dssp_scheduler_();
	}
	
	// loop through tasks
	public static void dssp_scheduler_() {
		
		taskQueue = generate();
		localTaskQueue = generateLocal();
		log(taskQueue.size() + " tasks.");
		int z = 0;
		while (z < SIMULATION_ROUNDS) {
			run_prev_ = getMax(getCurrentTimes(readyMachines));
			log("Simulation round " + (z + 1));
			sync_task_counter = 0;
			COMMUNICATION_COST = 0;
			for (int k = 0; k < taskQueue.size(); k++) {
				Task t = taskQueue.get(k);
				readyMachines.sort(Comparator.comparing(d -> d.getId()));
				// update stack
//				readyMachines.forEach(d ->{
//					d.populateStack(d.getTime());
//				});
				staleness_check = false;
				log("====================================== ");
				if (t.type == TAG_C_ASYNC) {
					scheduleAsyncTask(t);
				} else if (t.type == TAG_C_SYNC) {
					
					readyMachines.forEach(d -> {
						if (d.allowable_runs > 0) {
							d.allowable_runs -= 1;
							//d.increaseTime(1);
							scheduleSyncTask(d, t);
						}else {
							checkBounds(d, t);
						}
						
					});
					if (staleness_check) {
						readyMachines.forEach(d -> {						
							COMMUNICATION_COST = Math.max(COMMUNICATION_COST,
									2*getTaskGaussian(WORKER_CONTROLLER_COMMUNICATION, 2));
						});
					}
					sync_task_counter++;
				}
				readyMachines.sort(Comparator.comparing(d -> d.getId()));
				// update stack
				readyMachines.forEach(d ->{
					d.populateStack(d.getTime());
				});
				full_stack.clearStack();
				readyMachines.forEach(d ->{
					full_stack.populateFullStack(d.stack_worker, d.getId());
				});				
			}
			
			for (Task local: localTaskQueue) {
				readyMachines.sort(Comparator.comparing(d -> d.getId()));
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
	
	// async task scheduler
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
	public static void scheduleSyncTask(Worker worker, Task t) {
		
		double time = 0;
		if (worker.timing == true) time = t.early_mean;
		else time = t.late_mean;
		runTask(worker, t, (time+(worker.sync_deviation)/4), 1); 		
		
	}	
	
	public static void checkBounds(Worker p, Task t) {
		
		System.out.println("Checking bounds");
		int slowest_id = full_stack.getSlowestWorkerId();	
		int fastest_id = full_stack.getFastestWorkerId();
				
		// identify slowest worker
		readyMachines.forEach(d ->{
			if (d.getId() == slowest_id) {
				slowest = d;
			}
			if (d.getId() == fastest_id) {
				fastest = d;
			}
		});

		
		if (slowest.getTime() - p.getTime() <= STALENESS) {
			p.increaseTime(1);
			scheduleSyncTask(p, t);
		}else {
			if (fastest == p) {
				checkStaleness(p, t);
				COMMUNICATION_COST += LOCAL_COMMUNICATION;
				if (p.allowable_runs > 0) {
					//p.increaseTime(1);
					scheduleSyncTask(p, t);
				}
			}else {
				waitForSlowest(slowest, t);
				//p.increaseTime(1);
				scheduleSyncTask(p, t);
			}
			
		}		
	}
	
	public static void checkStaleness(Worker p, Task t) {
		
		System.out.println("Updating optimal maximum bound r");
		staleness_check = true;
		
		Utility.SizedStack<Double> stack_p = new Utility.SizedStack<Double>((int) MAX_STALE);
		Utility.SizedStack<Double> stack_slowest = new Utility.SizedStack<Double>((int) MAX_STALE);
		
//		for (int k = 1; k < full_stack.getSize()+1; k++) {
//			System.out.println(full_stack.getStackForWorker(k).get(0)+ "," +
//					full_stack.getStackForWorker(k).get(1));
//		}
//				
		
		System.out.println("Slowest worker is " + slowest.name);
		System.out.println("Fastest worker is " + fastest.name);
		System.out.println("p worker is " + p.name);
		// I_p
		double p_A0 = full_stack.getStackForWorker(p.getId()).get(1);
		double p_A1 = full_stack.getStackForWorker(p.getId()).get(0);
		
		double p_length = p_A0 - p_A1;
		
		double slowest_A0 = full_stack.getStackForWorker(slowest.getId()).get(1);
		double slowest_A1 = full_stack.getStackForWorker(slowest.getId()).get(0);
		
//		System.out.println("Slowest A0 " + slowest_A0 + "," + "Slowest A1 " + slowest_A1 +
//				"P A0 " + p_A0 + "P A1 " + p_A1);
		
		// I_slowest
		double slowest_length = slowest_A0 - slowest_A1; 
		
		// p stack Sim_p
		stack_p.add(full_stack.getStackForWorker(p.getId()).get(1));		
		for (int i = 1; i < (int) MAX_STALE; i++) {
			stack_p.add(full_stack.getStackForWorker(p.getId()).get(1) + (i * p_length));
		}
		//System.out.println("Size of p stack is " + stack_p.size());
		// slowest stack Sim_slowest
		stack_slowest.add(full_stack.getStackForWorker(slowest.getId()).get(0) + slowest_length);
		for (int i = 2; i < (int) MAX_STALE + 1; i++) {
			stack_slowest.add(full_stack.getStackForWorker(slowest.getId()).get(0) + (i * slowest_length));
		}
		//System.out.println("Size of slowest stack is " + stack_slowest.size());
//		
		double minimum_simulated_gap = 100000000000000.0;
		
		for (int j = 0; j < stack_slowest.size(); j++) {
			for (int m = 0; m < stack_p.size(); m++) {
				if ( stack_slowest.get(j) - stack_p.get(m) < minimum_simulated_gap) {
					minimum_simulated_gap = stack_slowest.get(j) - stack_p.get(m);
					p.allowable_runs = m;
				}
			}
		}		
		
	}
	
	public static void waitForSlowest(Worker slowest, Task t) {
		System.out.println("Waiting for slowest");
		readyMachines.forEach(d ->{
			double slowest_time = full_stack.getStackForWorker(slowest.getId()).get(1);
			double d_time = full_stack.getStackForWorker(d.getId()).get(1);
			if (slowest_time - d_time > STALENESS) {
				d.setTime(slowest_time - (STALENESS - 7));
			}
		});
		
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
