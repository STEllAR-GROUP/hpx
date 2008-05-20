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

#include <cstdlib>
#include <iostream>
#include <string>
#include <algorithm>
#include <boost/bind.hpp>
#include <boost/coroutine/coroutine.hpp>

// peers version of the producer/consumer pattern.
namespace coroutines = boost::coroutines;
using coroutines::coroutine;

typedef coroutine<void(const std::string&)> consumer_type;
typedef coroutine<void()> producer_type;

void producer_body(producer_type::self& self, 
		   std::string base, 
		   consumer_type& consumer) {
  std::sort(base.begin(), base.end());
  do {
    self.yield_to(consumer, base);
  } while (std::next_permutation(base.begin(), base.end()));
}

void consumer_body(consumer_type::self& self, 
		   const std::string& value,
		   producer_type& producer) {
  std::cout << value << "\n";
  while(true) {
    std::cout << self.yield_to(producer) << "\n";
  } 
}

int main() {
  {
    std::string test("test");
    std::cout << test;

    producer_type producer;
    consumer_type consumer;
    
    producer = producer_type
      (boost::bind
       (producer_body, 
	_1, 
	"hello", 
	boost::ref(consumer)));

    consumer = consumer_type
      (boost::bind
       (consumer_body, 
	_1,
	_2,
	boost::ref(producer)));
       
    producer();
  }

  {
    producer_type producer;
    consumer_type consumer;
    
    consumer = consumer_type
      (boost::bind
       (consumer_body, 
	_1,
	_2,
	boost::ref(producer)));
       
    producer = producer_type
      (boost::bind
       (producer_body, 
	_1, 
	"hello", 
	boost::ref(consumer)));
       
    consumer(std::string());
  }  
}
