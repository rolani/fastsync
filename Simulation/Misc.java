package sim;

import java.util.Random;

public class Misc {
	static Utility.SizedStack<Double> stack1;
	static final Random RANDOM = new Random();
	
	public static void elems() {
		stack1 = new Utility.SizedStack<Double>(2);
		for (int i = 0; i < 10; i ++) {
			stack1.push_(i*5.5);
			
		}
//		
//		for (Double d: stack1) {
//			System.out.print(d + " - ");
//		}
//		System.out.println();
	}
	
	public static void main(String[] args) {
//		elems();
//		System.out.println(stack1.get(1));
//		System.out.println(stack1.get(0));
//		System.out.println((int) 123.345);
		
		
		for (int i = 0; i < 20; i++) {
		
			System.out.println(RANDOM.nextGaussian());
		}
	}

}
