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
#include <boost/lexical_cast.hpp>
#include <queue>
#include <list>

namespace coro = boost::coroutines;
using coro::shared_coroutine;

typedef shared_coroutine<void()> job_type;

class scheduler {
public:
  void add(job_type job) {
    BOOST_ASSERT(job);
    m_queue.push(job);
  }
  
  void reschedule(job_type::self& self) {
    BOOST_ASSERT(current());
    add(current());
    self.yield();
  }

  job_type& current() {
    BOOST_ASSERT(m_current);
    return m_current;
  }

  void run () {
    while(!m_queue.empty()) {
      pop();    
      current()(std::nothrow);	
    }

  }
private:
  void pop() {
    m_current = m_queue.front();
    m_queue.pop();
  }

  std::queue<job_type> m_queue;
  job_type m_current;
};

class message_queue {
public:
  std::string pop(job_type::self& self) {
    while(m_queue.empty()) {
      m_waiters.push(m_scheduler.current());
      self.yield();      
    }
    BOOST_ASSERT(!m_queue.empty());
    std::string res = m_queue.front();
    m_queue.pop();
    return res;
  }

  void push(const std::string& val) {
    m_queue.push(val);
    while(!m_waiters.empty()) {
      m_scheduler.add(m_waiters.front());
      m_waiters.pop();
    }
  }

  message_queue(scheduler& s) :
    m_scheduler(s) {}

private:
  std::queue<std::string> m_queue;
  std::queue<job_type> m_waiters;
  scheduler & m_scheduler;
};

scheduler global_scheduler;
message_queue mqueue(global_scheduler);

void producer(job_type::self& self, int id, int count) {
  while(--count) {
    std::cout << "In producer: "<<id<<", left: "<<count <<"\n";
    mqueue.push("message from " + boost::lexical_cast<std::string>(id));
    std::cout << "\tmessage sent\n";
    global_scheduler.reschedule(self);
  } 
}

void consumer(job_type::self& self, int id) {
  while(true) {
    std::string result = mqueue.pop(self);
    std::cout <<"In consumer: "<<id<<"\n";
    std::cout <<"\tReceived: "<<result<<"\n";
  }
}

int main() {
  global_scheduler.add(boost::bind(producer, _1, 0, 4));
  global_scheduler.add(boost::bind(producer, _1, 1, 3));
  global_scheduler.add(boost::bind(producer, _1, 2, 2));
  global_scheduler.add(boost::bind(consumer, _1, 3));
  global_scheduler.add(boost::bind(consumer, _1, 4));
  global_scheduler.run();
}
