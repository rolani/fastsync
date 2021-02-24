package sim;

import java.util.ArrayList;
import java.util.List;

//represents the fog class.
public class Cluster {
	int c_id = 0;
	Double time;
	Boolean assigned;
	List<Worker> deviceList; // contains devices in a cluster
	
	// constructor with id
	Cluster(int id){
		deviceList = new ArrayList<Worker>();
		c_id = id;
	}

	// register a device to a cluster
	public void addDevice(Worker d) {  
		deviceList.add(d); 
	}

	// remove a device from a cluster
	public void removeDevice(Worker d) {
		deviceList.remove(d);
	}
	
	public void clearDeviceList() {
		deviceList.clear();
	}

	// check if a device is in a cluster
	public boolean findDevice(Worker d) {
		if (deviceList.contains(d)) {
			return true;
		} else
			return false;
	}

	public List<Worker> getDevices() {
		return deviceList;
	}

	// get the number of devices in cluster
	public int numOfDevices() {
		return deviceList.size();
	}

}
