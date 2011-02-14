/*  Copyright (c) 2009 Steven Brandt
 *
 *  Distributed under the Boost Software License, Version 1.0. (See accompanying 
 *  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

import java.util.concurrent.*;

public class Fib2 implements Callable<Integer> {
    final int n;

    public Fib2(int n) {
        this.n = n;
    }

    public Integer call() throws Exception {
        if(n < 2)
            return n;
        FutureTask<Integer> f1 = new FutureTask<Integer>(new Fib2(n-1));
        FutureTask<Integer> f2 = new FutureTask<Integer>(new Fib2(n-2));
        exec(f1);
        exec(f2);
        return f1.get()+f2.get();
    }
    
    static public void exec(Runnable run) {
        new Thread(run).start();
    }

    public static void main(String[] args) throws Exception {
        double sum = 0, cnt = 0;
	int val = 0;
	if (args.length != 1) {
	    System.err.println("Error: program requires one integer argument");
	    System.exit(2);
	}
	try {val = Integer.parseInt(args[0]);}
	catch (NumberFormatException exc) {
	    System.err.println("Error: program argument is not an integer!");
	    System.exit(2);
	}
        for(int i=0;i<10;i++) {
            long t1 = System.nanoTime();
            FutureTask<Integer> f = new FutureTask<Integer>(new Fib2(val));
            exec(f);
            int res = f.get();
            long t2 = System.nanoTime();
            sum += 1e-9*(t2-t1);
            cnt += 1.0;
            System.out.println(res+" "+(sum/cnt));
            System.gc();
            System.gc();
        }
        System.exit(0);
    }
}
