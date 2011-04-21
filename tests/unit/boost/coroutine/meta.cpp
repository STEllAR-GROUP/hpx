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
#include <boost/mpl/assert.hpp>
#include <boost/coroutine/detail/is_callable.hpp>

namespace coroutines = boost::coroutines;

struct test_is_function_pointer {
  template<typename T>
  test_is_function_pointer(T) {
    BOOST_MPL_ASSERT((coroutines::detail::is_function_pointer<T>));
  }
};

struct test_is_functor {
  template<typename T>
  test_is_functor(T) {
    BOOST_MPL_ASSERT((coroutines::detail::is_functor<T>));
  }
};

struct test_is_callable {
  template<typename T>
  test_is_callable(T) {
    BOOST_MPL_ASSERT((coroutines::detail::is_callable<T>));
  }
};

void foo() {}

struct bar {
  typedef void result_type;
} a_bar;

int main() {
  test_is_function_pointer t1 (foo);
  test_is_functor t2 (a_bar);
  test_is_callable t3 (foo);
  test_is_callable t4 (a_bar);
}

