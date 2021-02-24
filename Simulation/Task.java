package sim;


public class Task{
	int type; // task type
	String name; // task name
	double early_mean; // early average runtime
	double late_mean; // late average runtime
	double variance;

	// constructor with name
	public Task(int type_, double mean1, double mean2, double var) {
		type = type_;
		early_mean = mean1;
		late_mean = mean2;
		variance = var;
	}
	
	public int getType() {
		return type;
	}
	
	public String getName() { 
		String s = null;
		switch (type){
			case 1:
				s = "Async task";
				break;
			case 2:
				s= "Sync task";
				break;
			default:
				break;
		
		}			
		return s;	
	}
	
	public double getEarlyMean() {
		return early_mean;
	}
	
	public double getLateMean() {
		return late_mean;
	}
	
	public double getVariance() {
		return variance;
	}
	
}
