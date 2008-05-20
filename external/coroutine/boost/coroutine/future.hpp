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

#ifndef BOOST_COROUTINE_FUTURE_HPP_20060728
#define BOOST_COROUTINE_FUTURE_HPP_20060728

// Max number of futures that can be waited for.
#ifndef BOOST_COROUTINE_WAIT_MAX
#define BOOST_COROUTINE_WAIT_MAX 10
#endif
#include <boost/none.hpp>
#include <boost/config.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/mpl/eval_if.hpp>
#include <boost/mpl/identity.hpp>
#include <boost/preprocessor/repetition.hpp>
#include <boost/preprocessor/facilities/intercept.hpp>

#include <boost/coroutine/move.hpp>
#include <boost/coroutine/tuple_traits.hpp>
#include <boost/coroutine/detail/signal.hpp>
#include <boost/coroutine/detail/arg_max.hpp>
#include <boost/coroutine/detail/call_impl.hpp>
#include <boost/coroutine/detail/wait_impl.hpp>
#include <boost/coroutine/detail/future_impl.hpp>
#include <boost/coroutine/detail/default_context_impl.hpp>

namespace boost { namespace coroutines {

  template<
    BOOST_PP_ENUM_BINARY_PARAMS
  (BOOST_COROUTINE_ARG_MAX,
   typename T, 
   = boost::tuples::null_type BOOST_PP_INTERCEPT),
    typename ContextImpl = detail::default_context_impl 
    >
  class future : 
    public movable
  <future<BOOST_PP_ENUM_PARAMS(BOOST_COROUTINE_ARG_MAX, T)> > 
  {

    friend struct detail::wait_gateway;
    typedef void (future::*safe_bool)();           
    void safe_bool_true() {}                       

  public:
    typedef ContextImpl context_impl;

    typedef tuple_traits<
      BOOST_PP_ENUM_PARAMS
      (BOOST_COROUTINE_ARG_MAX, T)
      > tuple_traits_type;

    typedef boost::mpl::bool_<tuple_traits_type::length == 1>
      is_singular;
   
    typedef BOOST_DEDUCED_TYPENAME tuple_traits_type::as_tuple tuple_type;
    typedef BOOST_DEDUCED_TYPENAME boost::mpl::eval_if<
      boost::mpl::not_<is_singular>,
      boost::mpl::identity<tuple_type>,
      BOOST_DEDUCED_TYPENAME tuple_traits_type::template at<0>
      >::type  value_type;
   
    typedef detail::future_impl<tuple_type, context_impl> future_impl;
    typedef future_impl * impl_pointer;
    template<typename CoroutineSelf>
      future(CoroutineSelf& self) :
      m_ptr(new future_impl(self)) {}
   
    future(move_from<future> rhs) :
      m_ptr(rhs->pilfer()) {}

   
    value_type& operator *() {
      BOOST_ASSERT(m_ptr);
      wait();
      return remove_tuple(m_ptr->value(),  boost::mpl::not_<is_singular>());
    }

    const value_type& operator *() const{
      BOOST_ASSERT(m_ptr);
      BOOST_ASSERT(m_ptr->get());
      wait();
      return remove_tuple(m_ptr->value(), boost::mpl::not_<is_singular>());
    }
   
    future& operator=(move_from<future> rhs) {
      future(rhs).swap(*this);
      return *this;
    }

    future& operator=(const value_type& rhs) {
      BOOST_ASSERT(!pending());
      m_ptr->get() = tuple_type(rhs);
      return *this;
    }

    future& operator=(none_t) {
      BOOST_ASSERT(!pending());
      m_ptr->get() = none;
      return *this;
    }

    operator safe_bool() const {    
      BOOST_ASSERT(m_ptr);
      return m_ptr->get()?                     
	&future::safe_bool_true: 0;                 
    }       

    BOOST_DEDUCED_TYPENAME
      future_impl::pointer & 
      operator ->() {
      BOOST_ASSERT(m_ptr);
      wait();
      return m_ptr->get();
    }

    BOOST_DEDUCED_TYPENAME
      future_impl::pointer const 
      operator ->() const {
      BOOST_ASSERT(m_ptr);
      wait();
      return m_ptr->get();
    }

    friend void swap(future& lhs, future& rhs) {
      std::swap(lhs.m_ptr, rhs.m_ptr);
    }

    // On destruction, if the future is 
    // pending it will be destroyed.
    ~future() {
      if(m_ptr) {
	wait();
      	delete m_ptr;
      }
    }

    // Return true if an async call has
    // been scheduled for this future.
    bool pending() const {
      BOOST_ASSERT(m_ptr);
      return m_ptr->pending();
    }

  private:
    void wait(int n) {
      m_ptr->wait(n);
    }

    void wait() {
      m_ptr->wait();
    }

    template<typename T>
      value_type& remove_tuple(T& x, boost::mpl::false_) {
      return boost::get<0>(x);
    }

    template<typename T>
      value_type& remove_tuple(T& x, boost::mpl::true_) {
      return x;
    }

    void mark_wait(bool how) {
      BOOST_ASSERT(m_ptr);
      m_ptr->mark_wait(how);
    }

    bool waited() const {
      BOOST_ASSERT(m_ptr);
      return m_ptr->waited();
    }

    impl_pointer pilfer() {
      impl_pointer ptr = m_ptr;
      m_ptr = 0;
      return ptr;
    }

    impl_pointer m_ptr;    
  };

#define BOOST_COROUTINE_gen_call_overload(z, n, unused) \
  template<                                             \
    BOOST_PP_ENUM_PARAMS(n, typename T)                 \
    BOOST_PP_COMMA_IF(n)                                \
    typename Functor,                                   \
    typename Coroutine>                                 \
  future<BOOST_PP_ENUM_PARAMS(n, T)>                    \
  call(const Functor f, const Coroutine& coro) {        \
    return detail::call_impl<future<BOOST_PP_ENUM_PARAMS (n,T)> >(f, coro);  \
  }                                                     \
/**/

  /*
   * Generate overloads of call<...>(function, coroutine) for
   * an arbitrary argument numbers that will forward to 
   * detail::call_impl<future<...> >(function, coroutine)
   */
  BOOST_PP_REPEAT(BOOST_COROUTINE_ARG_MAX,
		  BOOST_COROUTINE_gen_call_overload,
		  ~);
    
#define BOOST_COROUTINE_empty(z, n, name) \
/**/

#define BOOST_COROUTINE_gen_reference(z, n, unused) \
    BOOST_PP_CAT(T, n) &                            \
/**/

#define BOOST_COROUTINE_gen_wait_non_zero(z, n, name)\
  template<BOOST_PP_ENUM_PARAMS(n, typename T)>      \
  void name (BOOST_PP_ENUM_BINARY_PARAMS(n, T, &arg)) {  \
    detail::BOOST_PP_CAT(name, _impl)                \
      (boost::tuple<BOOST_PP_ENUM(n, BOOST_COROUTINE_gen_reference, ~)> \
       (BOOST_PP_ENUM_PARAMS(n, arg)));              \
  }                                                  \
/**/

#define BOOST_COROUTINE_gen_wait(z, n, name)     \
  BOOST_PP_IF(n,                                 \
	      BOOST_COROUTINE_gen_wait_non_zero, \
	      BOOST_COROUTINE_empty)(z, n, name) \
/**/

  /*
   * Generate overloads of wait(coro, ...) for
   * an arbitrary argument number that will
   * forward to detail::wait_impl(coro, tuple<...>)
   */
  BOOST_PP_REPEAT(BOOST_COROUTINE_WAIT_MAX,
		  BOOST_COROUTINE_gen_wait,
		  wait);

  /*
   * Generate wait_all(coro, ...) for an arbitrary arguement
   * number that will forward to
   * detail::wait_all_impl(coro, tuple<...>)
   */
  BOOST_PP_REPEAT(BOOST_COROUTINE_WAIT_MAX,
		  BOOST_COROUTINE_gen_wait,
		  wait_all);

#undef BOOST_COROUTINE_gen_wait
#undef BOOST_COROUTINE_empty
#undef BOOST_COROUTINE_gen_reference
#undef BOOST_COROUTINE_gen_wait_non_zero
#undef BOOST_COROUTINE_gen_call_overload

  template<typename Future >
  struct make_callback_result {
    typedef detail::callback<Future> type;
  };

  /*
   * Returns a callback object that when invoked
   * will signal the associated coroutine::self object.
   * It will extend the lifetime of the object until
   * it is signaled. More than one callback object
   * can be pending at any time. The coroutine self
   * will last at least untill the last pending callback 
   * is fired.
   */
  template<typename Future>
  BOOST_DEDUCED_TYPENAME 
  make_callback_result<Future>
  ::type
  make_callback(Future& future) {
    return BOOST_DEDUCED_TYPENAME make_callback_result<Future>::type
      (future);
  }

} } 
#endif

