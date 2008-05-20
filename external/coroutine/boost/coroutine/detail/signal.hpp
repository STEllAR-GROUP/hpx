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

#ifndef BOOST_COROUTINE_DETAIL_SIGNAL_HPP_20060728
#define BOOST_COROUTINE_DETAIL_SIGNAL_HPP_20060728
#include <boost/coroutine/detail/future_impl.hpp>
namespace boost { namespace coroutines { namespace detail {
  /*
   * Private interface for coroutine signaling/waiting.
   * These class is friend of future_impl and
   * its static members are invoked by asynchronous callback functions.
   */
  struct wait_gateway {
    template<typename Future>
    static void wait(Future& f, int n) {
      f.wait(n);
    }

    template<typename Future>
    static 
    bool waited(const Future& f) {
      return f.waited();
    }

    template<typename Future>
    static 
    void mark_wait(Future& f, bool how) {
      f.mark_wait(how);
    }

    template<typename Future>
    static
    BOOST_DEDUCED_TYPENAME 
    Future::impl_pointer
    get_impl(Future& f) {
      return f.m_ptr;
    }
  };


} } }
#endif
