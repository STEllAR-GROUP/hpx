//  Copyright (c) 2006, Giovanni P. Deretta
//
//  This code may be used under either of the following two licences:
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy 
//  of this software and associated documentation files (the "Software"), to deal 
//  in the Software without restriction, including without limitation the rights 
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
//  copies of the Software, and to permit persons to whom the Software is 
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in 
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN 
//  THE SOFTWARE. OF SUCH DAMAGE.
//
//  Or:
//
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <iomanip>
#include <sys/time.h>
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/coroutine/coroutine.hpp>
using namespace std;

namespace coroutines = boost::coroutines;
using coroutines::coroutine;

inline double gettimestamp() {
    timeval timestamp;
    gettimeofday(&timestamp,NULL);
    return double(timestamp.tv_usec) + double(timestamp.tv_sec)*1000000.0;
}

struct timer {
  double counter;
  timer() : counter (gettimestamp()) {
  }

  double stop() {
    return gettimestamp() - counter;
  }
};


void sink(int&);
void ol_foo(int*);

typedef void signature(int*);
typedef boost::function<signature> function_type;
typedef coroutine<signature> coroutine_type;

void foo(int * x) {
  *x ^= 0xAAAA; 
  return;
}

void foo_coro(coroutine_type::self& self, int * x) {
  while(true) {
    foo(x);
    self.yield();
  }
}

void ol_foo_coro(coroutine_type::self& self, int * x) {
  while(true) {
    ol_foo(x);
    self.yield();
  }
}

struct foo_struct {
  void operator()(int *x) {
    *x ^= 0xAAAA;
  };
};

struct foo_struct_coro {
  typedef void result_type;
  void operator()(coroutine_type::self& self, int *x) {
    while(true){
      *x ^= 0xAAAA;
      self.yield();
    }
  };
};


template<typename F>
double test(F f, int n) {
  int accum = 5;
  timer t;
  while(n--)
    f(&accum);
  sink(accum);
  return t.stop();
}

int main() {

  int iterations = 1000*10*10*10*10;
#if 0
  function_type function_foo(foo);
  function_type function_foo_struct = foo_struct();
  function_type function_ol_foo(ol_foo);
  coroutine_type coro_foo(foo_coro);
  foo_struct_coro t;
  coroutine_type coro_foo_struct  (t);
  coroutine_type coro_ol_foo(ol_foo_coro);
  cout.setf(ios_base::floatfield);
  cout.unsetf(ios_base::scientific);
  cout << setw(50) << "Call to function: " 
       << setw(16) << right << test(foo, iterations) << endl;
  cout << setw(50) << "Call to out-of-line function: " 
       << setw(16) << right << test(ol_foo, iterations) << endl;
  cout << setw(50) << "Call to function object: " 
       << setw(16) << right << test(foo_struct(), iterations) << endl;
  cout << setw(50) << "Call to boost::function of function: " 
       << setw(16) << right << test(function_foo, iterations) << endl;
  cout << setw(50) << "Call to boost::function of function object: " 
       << setw(16) << right << test(function_foo_struct, iterations) << endl;
  cout << setw(50) << "Call to boost::function of out-of-line function: " 
       << setw(16) << right << test(function_ol_foo, iterations) << endl;
  cout << setw(50) << "Call to coroutine of function: " 
       << setw(16) << right << noshowpoint << test(coro_foo, iterations) << endl;
  cout << setw(50) << "Call to coroutine of function object: " 
       << setw(16) << right  << noshowpoint << test(coro_foo_struct, iterations) << endl;
  cout << setw(50) << "Call to coroutine of out-of-line function: " 
       << setw(16) << right << noshowpoint << test(coro_ol_foo, iterations) << endl;
#else
  foo_struct_coro t;
  coroutine_type coro_foo_struct  (t);
  cout << setw(50) << "Call to coroutine of function object: " 
    << setw(16) << right  << noshowpoint << test(coro_foo_struct, iterations) << endl;
#endif
}
