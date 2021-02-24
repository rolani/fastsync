package init;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import javax.script.ScriptContext;
import javax.script.ScriptEngine;
import javax.script.ScriptEngineManager;
import javax.script.ScriptException;
import javax.script.SimpleScriptContext;


public class Logging {
	static final Random RANDOM = new Random();
	static void try1() throws FileNotFoundException, ScriptException {
		StringWriter writer = new StringWriter(); //output will be stored here
		 
	    ScriptEngineManager manager = new ScriptEngineManager();
	    ScriptContext context = new SimpleScriptContext();
	 
	    context.setWriter(writer); //configures output redirection
	    System.out.println("About to configure");
	    ScriptEngine engine = manager.getEngineByName("python");
	    System.out.println("Run python file");
	    engine.eval(new FileReader("java.py"), context);
	    System.out.println("Done running py file");
	    System.out.println(writer.toString()); 
	}
	
	static double gauss(double variance) {
		double time = RANDOM.nextGaussian() * variance;
		if (time <= 0)
			gauss(variance);
		return time;
	}
	
	static void try2() {
		String s = null; 

        try {
            
            Process p = Runtime.getRuntime().exec("python java.py");
            
            BufferedReader stdInput = new BufferedReader(new 
                 InputStreamReader(p.getInputStream()));

            BufferedReader stdError = new BufferedReader(new 
                 InputStreamReader(p.getErrorStream()));

            // read the output from the command
            System.out.println("Here is the standard output of the command:\n");
            while ((s = stdInput.readLine()) != null) {
                System.out.println(s);
            }
            
            // read any errors from the attempted command
            System.out.println("Here is the standard error of the command (if any):\n");
            while ((s = stdError.readLine()) != null) {
                System.out.println(s);
            }
            
            System.exit(0);
        }
        catch (IOException e) {
            System.out.println("exception happened - here's what I know: ");
            e.printStackTrace();
            System.exit(-1);
        } 
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
        
	}
	
	public static void log(Object aMsg){
		System.out.println(String.valueOf(aMsg));
	}
	
//	public static void main(String[] args) throws ScriptException, IOException {
//		
////		 for (int i = 0; i < 14; i++) {
////			 System.out.println(gauss(4.5));
////		 }
//		runPythonClustering();
//	    
//	}

}
