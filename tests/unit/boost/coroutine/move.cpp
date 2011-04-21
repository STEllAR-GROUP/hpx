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
#include <boost/coroutine/coroutine.hpp>
#include <boost/coroutine/move.hpp>
#include <boost/test/unit_test.hpp>

namespace coroutines = boost::coroutines;
using coroutines::coroutine;

typedef coroutine<void(void)> coroutine_type;

void coroutine_body(coroutine_type::self&) {}

void sink(coroutine_type) {}

coroutine_type source() {
  return coroutine_type(coroutine_body);
}

void sink_ref(boost::coroutines::move_from<coroutine_type>){}

void test_move() {
  std::cout << "test 1\n";
  coroutine_type coro (source());
  std::cout << "test 2\n";
  coroutine_type coro2 = coroutine_type(coroutine_body);
  std::cout << "test 3\n";
  coroutine_type coro3;
  std::cout << "test 4\n";
  coro3 = coroutine_type(coroutine_body);
  std::cout << "test 5\n";
  coroutine_type coro4 = source();
  std::cout << "test 6\n";
  coroutine_type coro5 (source());
  std::cout << "test 7\n";
  sink(coroutine_type(coroutine_body));
  std::cout << "test 8\n";
  sink(move(coro5));
  std::cout << "test 9\n";
  coro3 = move(coro4);
  std::cout << "test 10\n";
  sink(source());

}

boost::unit_test::test_suite* init_unit_test_suite( int argc, char* argv[] )
{
    boost::unit_test::test_suite *test = BOOST_TEST_SUITE("move coroutine test");

    test->add(BOOST_TEST_CASE(&test_move));

    return test;
}
