package sim;

import java.util.ArrayList;
import java.util.List;

public class Worker implements Runnable {
	String name;
	int d_id;
	double time;
	double cluster_time;
	Boolean hasJoined = false;
	Boolean timing = false;
	Cluster c = null;
	List<Task> localTaskList;
	static Thread t;
	double sync_deviation;
	double async_deviation;
	double prev_time1; // earlier
	double prev_time2; // latest
	int allowable_runs;
	Utility.SizedStack<Double> stack_worker;
	
	
	//constructor with name
	Worker(String tName, int id, double s_var, double a_var){
		localTaskList = new ArrayList<Task>();
		stack_worker = new Utility.SizedStack<Double>(2);
		name = tName;
		d_id = id;
		t = new Thread(this, name); 
		t.start();
		time = 0;
		sync_deviation = s_var;
		async_deviation = a_var;
		
		if (Math.random() > 0.5) {
			timing = true;
		}
	}
	@Override
	public void run() {
		//System.out.println("I am here" + t.getName());
	}
	
	public int getId() {
		return d_id;
	}
	
	public String getName() {
		return name;
	}
	
	public void populateStack(double value) {
		stack_worker.push_(value);
	}
	
	public void setTime(double time){
		this.time = time;
	}
	
	public Double getPrevTime1() {
		return prev_time1;
	}
	
	public void setPrevTime1(double time){
		this.prev_time1 = time;
	}
	
	public Double getPrevTime2() {
		return prev_time2;
	}
	
	public void setPrevTime2(double time){
		this.prev_time2 = time;
	}
	
	public int getRunGap() {
		return allowable_runs;
	}
	
	public void setRunGap(int gap){
		this.allowable_runs = gap;
	}
	
	public void increaseTime(double i){
		time = time + i;
	}
	
	//return current machine time
	public Double getTime(){
		return time;
	}
	
	public Double getSyncVariance(){
		return sync_deviation;
	}
	
	public Double getAsyncVariance(){
		return async_deviation;
	}
	
	public void setClusterTime(double time){
		this.cluster_time = time;
	}
	
	public Double getClusterTime(){
		return cluster_time;
	}
	
	public void setCurrentCluster(Cluster clus) {
		this.c = clus;
	}
	
	public Cluster getCurrentCluster() {
		Cluster none = new Cluster(0);
		if (c != null) {
			return c;
		}else {
			return none;
		}
	}
	
	public void addLocalTask(Worker this, Task t) {
		localTaskList.add(t);
	}
	
	public boolean hasLocalTask() {
		if (localTaskList.size() == 0) {return false;}		
		else {return true;} 
	}
	
	public void removeLocalTask(Task t) {
		localTaskList.remove(t);
	}
	
	public void setJoined(boolean bool) {
		hasJoined = bool;
	}

	public boolean isJoined() {
		return hasJoined;
	}
}
