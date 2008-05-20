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
#include <algorithm>
#include <string>
#include <boost/bind.hpp>
#include <boost/coroutine/generator.hpp>

// Consumer driven producer/consumer pattern.
namespace coroutines = boost::coroutines;
using coroutines::generator;

typedef generator<const std::string&> generator_type;

const std::string& producer(generator_type::self& self, std::string base) {
  std::sort(base.begin(), base.end());
  do {
    self.yield(base);
  } while (std::next_permutation(base.begin(), base.end()));
  self.exit();
}

template<typename Producer>
void consumer(Producer producer) {
  do {
    std::cout <<*producer << "\n";
  } while(++producer);
}

struct filter {
  typedef const std::string& result_type;

  template<typename Producer>
  const std::string& operator()
    (generator_type::self& self, Producer producer) {
    do {
      self.yield(*producer + " world");
    } while(++producer);

    self.exit();
    // gcc complains here while it doesn't for `producer`... werid.
    // the abort quiets it.
    abort();
    // it doesn't quiet VCPP8 though!
  }
};
int main() {
  consumer
    (generator_type
     (boost::bind
      (producer, _1, std::string("hello"))));

  
  consumer
    (generator_type
     (boost::bind
      (filter(),
       _1,
       generator_type
       (boost::bind
	(producer, _1, std::string("hello"))))));
  
}
