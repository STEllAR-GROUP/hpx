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

#include <vector>
#include <iterator>
#include <algorithm>
#include <boost/bind.hpp>
#include <boost/coroutine/generator.hpp>
#include <boost/test/unit_test.hpp>

namespace coroutines = boost::coroutines;
using coroutines::generator;

typedef generator<int> generator_type;

int count(generator_type::self& self, int rmin, int rmax) {
  while(rmin < rmax -1) {
    self.yield(rmin++);
  }
  return (rmin);
}

int count2(generator_type::self& self, int rmin, int rmax) {
  while(rmin < rmax) {
    self.yield(rmin++);
  }
  self.exit();
}


void test_generator() {
  {
    generator_type gen(boost::bind(count, _1, 10, 20));
    std::vector<int> vec;
    std::copy(gen, generator_type(), std::back_inserter(vec));
    BOOST_CHECK(vec.size() == 10);
  }
  {
    generator_type gen(boost::bind(count2, _1, 10, 20));
    std::vector<int> vec;
    std::copy(gen, generator_type(), std::back_inserter(vec));
    BOOST_CHECK(vec.size() == 10);
  }
  {
    generator_type gen(boost::bind(count2, _1, 10, 20));
    std::vector<int> vec;
    std::copy(gen, generator_type(), std::back_inserter(vec));
    BOOST_CHECK(vec.size() == 10);
  }

}

boost::unit_test::test_suite* init_unit_test_suite( int argc, char* argv[] )
{
    boost::unit_test::test_suite *test = BOOST_TEST_SUITE("generator test");

    test->add(BOOST_TEST_CASE(&test_generator));

    return test;
}
