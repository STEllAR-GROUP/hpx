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
#include <string>
#include <boost/coroutine/shared_coroutine.hpp>
#include <boost/bind.hpp>
#include<queue>

namespace coro = boost::coroutines;
using coro::shared_coroutine;

typedef shared_coroutine<void()> job_type;

class scheduler {
public:
  void add(job_type job) {
    m_queue.push(job);
  }
  
  job_type& current() {
    return m_queue.front();
  }

  void run () {
    while(!m_queue.empty()) {
      current()(std::nothrow);
      if(current()) 
	add(current());
      m_queue.pop();
    }
  }
private:

  std::queue<job_type> m_queue;
};

scheduler global_scheduler;

void printer(job_type::self& self, std::string name, int iterations) {
  while(iterations --) {
    std::cout<<name <<" is running, "<<iterations<<" iterations left\n";
    self.yield();
  }
  self.exit();
}

int main() {
  global_scheduler.add(boost::bind(printer, _1, "first", 10));
  global_scheduler.add(boost::bind(printer, _1, "second", 5));
  global_scheduler.add(boost::bind(printer, _1, "third", 3));
  global_scheduler.run();
}
