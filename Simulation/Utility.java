package sim;


import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Stack;

public class Utility {	

	public static class SizedStack<T> extends Stack<T> {
	    /**
		 * 
		 */
		private static final long serialVersionUID = 1L;
		private int maxSize;

	    public SizedStack(int size) {
	        super();
	        this.maxSize = size;
	    }

	    public T push_(T object) {
	        //If the stack is too big, remove elements until it's the right size.
	        while (this.size() >= maxSize) {
	            this.remove(0);
	        }
	        return super.push(object);
	    }
	}
	
	public static class FullStack {
		private List<SizedStack<Double>> stack_;
		private List<Integer> ids;
		
		public FullStack() {
			stack_ = new ArrayList<SizedStack<Double>>();
			ids = new ArrayList<Integer>();
		}
		
		public void populateFullStack(SizedStack<Double> stack1, int id) {
			stack_.add(stack1);
			ids.add(id);
		}
		
		public void populateFullIds(List<Integer> id_list) {
			ids = id_list;
		}
		
		public SizedStack<Double> getStackForWorker(int id){
			return stack_.get(ids.indexOf(id));
		}
		
		public int getIdFromIndex(int index) {
			return ids.get(index);
		}
		
		public void clearStack() {
			stack_.clear();
		}
		
		public int getSize() {
			return stack_.size();
		}
		
		public int getSlowestWorkerId() {
			int slowest_id = 0;
			double slowest_runtime = 0.0;
			
			for (int i = 0; i < stack_.size(); i++) {
				if (stack_.get(i).get(1) > slowest_runtime) {
					slowest_id = ids.get(i);
					slowest_runtime = stack_.get(i).get(1);
				}
			}			
			return slowest_id;				
		}
		
		public int getFastestWorkerId() {
			int fastest_id = 0;
			double fastest_runtime = 100000000000000.0;
			
			for (int i = 0; i < stack_.size(); i++) {
				if (stack_.get(i).get(1) < fastest_runtime) {
					fastest_id = ids.get(i);
					fastest_runtime = stack_.get(i).get(1);
				}
			}			
			return fastest_id;				
		}
		
		
	}

	public static class KeyValueMap {

		private List<Integer> keys;
		private List<Double> values;

		double a, b;

		public KeyValueMap() {
			keys = new ArrayList<Integer>();
			values = new ArrayList<Double>();
		}

		
		public void insert(int key, double v1) {
			keys.add(key);
			values.add(v1);
		}

		public int getKey(int index) {
			if (index >= 0)
				return (int) keys.get(index);
			else
				return 0;
		}

		// get value based on index
		public double getValueFromIndex(int index) {
			return (double) values.get(index);
		}
		
		//get value based on key
		public double getValueFromKey(int key) {
			return (double) values.get(keys.indexOf(key));
		}

		public int size() {
			return keys.size();
		}
		
		public void clear() {
			keys.clear();
			values.clear();
		}
		
		public void removeFirst() {
			keys.remove(0);
			values.remove(0);
		}
	}

	public static class WorkerMap<W> {
		// private Machine machine;
		private List<Worker> machineList;
		//private List<Machine> sortedList;

		public WorkerMap() {
			machineList = new ArrayList<Worker>();
		}

		public void putMachine(Worker m) {
			machineList.add(m);
		}

		public Worker getMachine(int index) {
			return machineList.get(index);
		}

		public void removeMachine(Worker m) {
			machineList.remove(m);
		}
		
		public void removeAllMachines() {
			machineList.clear();
		}

		public String getName(int index) {
			Worker m = getMachine(index);
			return m.name.toString();
		}

		public int getSize() {
			return machineList.size();
		}

		public void sortByTime() {
			Collections.sort(machineList, new Comparator<Worker>() {

				public int compare(Worker m1, Worker m2) {
					return m1.getTime().compareTo(m2.getTime());
				}
			});
		}
	}

	public static class WorkerTimes {
		private List<Double> times;
		private List<Worker> workers;

		public WorkerTimes() {
			times = new ArrayList<>();
			workers = new ArrayList<>();
		}

		public void addInput(Worker w, Double b) {
			workers.add(w);
			times.add(b);
		}

		public Worker getWorker(int index) {
			return workers.get(index);
		}

		public double getTime(int index) {
			return times.get(index);
		}
	}
	
	public static class methods {
	    public double sum (List<Double> a){
	        if (a.size() > 0) {
	            int sum = 0;
	 
	            for (Double i : a) {
	                sum += i;
	            }
	            return sum;
	        }
	        return 0;
	    }
	    public double mean (List<Double> a){
	        double sum = sum(a);
	        double mean = 0;
	        mean = sum / (a.size() * 1.0);
	        return mean;
	    }
	    public double median (List<Double> a){
	        int middle = a.size()/2;
	 
	        if (a.size() % 2 == 1) {
	            return a.get(middle);
	        } else {
	           return (a.get(middle-1) + a.get(middle)) / 2.0;
	        }
	    }
	    public double sd (List<Double> a){
	        double sum = 0;
	        double mean = mean(a);
	 
	        for (Double i : a)
	            sum += Math.pow((i - mean), 2);
	        return Math.sqrt( sum / ( a.size() - 1 ) ); // sample
	    }
	}
}
