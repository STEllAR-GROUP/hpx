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
#include <boost/assert.hpp>
#include <boost/any.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/coroutine/coroutine.hpp>

namespace coroutines = boost::coroutines;
using coroutines::coroutine;

typedef coroutine<boost::any()> coroutine_type;

struct first_type{};
struct second_type{};
struct third_type{};
struct fourth_type{};
struct fifth_type{};

boost::any coro_body(coroutine_type::self& self)  {
  self.yield(first_type());
  self.yield(second_type());
  self.yield(third_type());
  self.yield(fourth_type());
  return fifth_type();  
}

void test_any() {
  coroutine_type coro(coro_body);
  BOOST_CHECK(coro().type() == typeid(first_type));
  BOOST_CHECK(coro().type() == typeid(second_type));
  BOOST_CHECK(coro().type() == typeid(third_type));
  BOOST_CHECK(coro().type() == typeid(fourth_type));
  BOOST_CHECK(coro().type() == typeid(fifth_type));
}

boost::unit_test::test_suite* init_unit_test_suite( int argc, char* argv[] )
{
    boost::unit_test::test_suite *test = BOOST_TEST_SUITE("any coroutine test");

    test->add(BOOST_TEST_CASE(&test_any));

    return test;
}
