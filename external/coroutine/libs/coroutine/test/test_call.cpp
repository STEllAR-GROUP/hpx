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

#include <boost/test/unit_test.hpp>
#include <boost/ref.hpp>
#include <boost/tuple/tuple_comparison.hpp>
#include <boost/none.hpp>
#include <boost/function.hpp>
#include <boost/coroutine/coroutine.hpp>
#include <boost/coroutine/future.hpp>


namespace coroutines = boost::coroutines;
using coroutines::coroutine;

typedef coroutine<void()> coroutine_type;

class a_pipe {
public:
  
  void send(int x) {
    m_callback (x);
  }

  template<typename Callback>
  void listen(Callback c) {
    m_callback = c;
  }
private:
  boost::function<void(int)> m_callback;
};

struct coro_body {

  coro_body(bool& flag) :
    m_flag(flag) {
    m_flag = true;
  }

  ~coro_body() {
    m_flag = false;
  }
  bool &m_flag;

  typedef void result_type;

  void operator() (a_pipe& my_pipe, coroutine_type::self& self)  {
    typedef coroutines::future<int> future_type;
    future_type future(self);

    my_pipe.listen(coroutines::make_callback(future));
    coroutines::wait(future);
    BOOST_CHECK(future);

    BOOST_CHECK(*future == 1);
    future = boost::none;
    my_pipe.listen(coroutines::make_callback(future));
    coroutines::wait(future);
    BOOST_CHECK(*future == 2);
    future = boost::none;
    my_pipe.listen(coroutines::make_callback(future));
    coroutines::wait(future);
    BOOST_CHECK(*future == 3);
    future = boost::none;
    return;  
  }
};

void test_call() {
  bool run_flag = true;
  {
    a_pipe my_pipe;
    coroutine_type coro(boost::bind(coro_body(run_flag), boost::ref(my_pipe), _1));
    coro(std::nothrow);
    my_pipe.send(1);
    my_pipe.send(2);
    my_pipe.send(3);
  }

  //check for leaks
  BOOST_CHECK(run_flag == false);
}

boost::unit_test::test_suite* init_unit_test_suite( int argc, char* argv[] )
{
    boost::unit_test::test_suite *test = BOOST_TEST_SUITE("call coroutine test");

    test->add(BOOST_TEST_CASE(&test_call));

    return test;
}
