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

#ifndef BOOST_COROUTINE_DETAIL_FUTURE_IMPL_HPP_20060809
#define BOOST_COROUTINE_DETAIL_FUTURE_IMPL_HPP_20060809
#include <boost/optional.hpp>
#include <boost/assert.hpp>
#include <boost/optional.hpp>
#include <boost/noncopyable.hpp>
#include <boost/coroutine/detail/coroutine_accessor.hpp>
#include <boost/coroutine/detail/context_base.hpp>

namespace boost { namespace coroutines { namespace detail {


  template<typename ValueType, typename ContextImpl>
  class future_impl : boost::noncopyable  {
  public:
    typedef ValueType value_type;
    typedef 
    context_base<ContextImpl> *      
    context_weak_pointer;
    
    typedef 
    BOOST_DEDUCED_TYPENAME
    context_base<ContextImpl>::pointer
    context_pointer;

    typedef boost::optional<value_type> pointer;

    template<typename CoroutineSelf>
    future_impl(CoroutineSelf& self) :
      m_coro_impl_weak(coroutine_accessor::get_impl(self)),
      m_coro_impl(0),
      m_waited(false)
    {}

    value_type& 
    value() {
      return *m_optional;
    }

    value_type const& 
    value() const{
      return *m_optional;
    }

    pointer& 
    get() {
      return m_optional;
    }

    pointer const& 
    get() const{
      return m_optional;
    }

    bool pending() {
      return 0 != m_coro_impl.get();
    }
    
    template<typename T>
    void assign(const T& val) {
      BOOST_ASSERT(pending());
      context_pointer p = m_coro_impl;
      m_coro_impl = 0;
      m_optional = val;
      p->count_down();
      if(waited() && p->signal())
	p->wake_up();
    }

    void mark_pending() {
      m_coro_impl = m_coro_impl_weak;
      m_coro_impl ->count_up();
    }

    void mark_wait(bool how) {
      m_waited = how;
    }

    bool waited() const {
      return m_waited;
    }

    context_pointer context() {
      BOOST_ASSERT(pending());
      return m_coro_impl;
    }
        
    void wait(int n) {
      m_coro_impl_weak->wait(n);
    }

    void wait() {
      if(!pending()) return;
      mark_wait(true);
      try {
	m_coro_impl->wait(1);
	BOOST_ASSERT(!pending());
      } catch (...) {
	mark_wait(false);
	throw;
      }
      mark_wait(false);
    }

  private:
    context_weak_pointer m_coro_impl_weak;
    context_pointer m_coro_impl;
    bool m_waited;
    boost::optional<value_type> m_optional;
  };
} } }
#endif
