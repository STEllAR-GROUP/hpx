/*  Copyright (c) 2009 Steven Brandt
 *
 *  Distributed under the Boost Software License, Version 1.0. (See accompanying 
 *  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

use Time;
var timer: Timer;
config var v: int = 30;

def fib(n: int): int {
    if n < 2 {
        return n;
    }
    var n1, n2: int;
    cobegin {
        n1 = fib(n-1);
        n2 = fib(n-2);
    }
    return n1+n2;
}

def main() {
    var f: int;
    timer.start();
    f = fib(v);
    timer.stop();
    writeln("time: ", timer.elapsed(), " fib(", v, "): ", f);
}
