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

#ifndef BOOST_COROUTINE_DETAIL_CALL_IMPL_HPP_20060728
#define BOOST_COROUTINE_DETAIL_CALL_IMPL_HPP_20060728

#include <boost/preprocessor/repetition.hpp>
#include <boost/call_traits.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/bind.hpp>
#include <boost/coroutine/coroutine.hpp>
#include <boost/coroutine/detail/arg_max.hpp>
#include <boost/coroutine/detail/signal.hpp>
#include <boost/coroutine/detail/coroutine_accessor.hpp>
namespace boost { namespace coroutines { namespace detail {

#define BOOST_COROUTINE_tuple_param_n(z, n, tuple)\
  BOOST_DEDUCED_TYPENAME                          \
  boost::tuples::element<n, tuple>::type          \
  BOOST_PP_CAT(arg, n)                            \
 /**/

  template<typename Future>                       
  class callback {
  public:

    typedef void result_type;
    
    callback(Future& future) :
      m_future_pimpl(wait_gateway::get_impl(future)) {
      m_future_pimpl->mark_pending();
    }
    
    typedef BOOST_DEDUCED_TYPENAME                
    Future::tuple_type tuple_type;              

    typedef BOOST_DEDUCED_TYPENAME                
    Future::tuple_traits_type tuple_traits_type;              

    /*
     * By default a callback is one shot only.
     * By calling this method you can revive a 
     * callback for another shot.
     * You must guaranee that the future
     * is still alive.
     */
    void revive() {
      m_future_pimpl->mark_pending();
    }

#define BOOST_COROUTINE_gen_argn_type(z, n, unused) \
    typedef BOOST_DEDUCED_TYPENAME                  \
    tuple_traits_type::                             \
    template at<n>::type                            \
    BOOST_PP_CAT(BOOST_PP_CAT(arg, n), _type);      \
/**/

    BOOST_PP_REPEAT(BOOST_COROUTINE_ARG_MAX,
		    BOOST_COROUTINE_gen_argn_type,
		    ~);
      
#define BOOST_COROUTINE_param_with_default(z, n, type_prefix) \
    BOOST_DEDUCED_TYPENAME call_traits                   \
    <BOOST_PP_CAT(BOOST_PP_CAT(type_prefix, n), _type)>  \
    ::param_type                                         \
    BOOST_PP_CAT(arg, n) =                               \
    BOOST_PP_CAT(BOOST_PP_CAT(type_prefix, n), _type)()  \
/**/
    
    void operator() 
      (BOOST_PP_ENUM
       (BOOST_COROUTINE_ARG_MAX,
	BOOST_COROUTINE_param_with_default,
	arg)) {
      m_future_pimpl->assign(tuple_type
	(BOOST_PP_ENUM_PARAMS
	 (BOOST_COROUTINE_ARG_MAX, arg)));
    }

  private:
    BOOST_DEDUCED_TYPENAME
    Future::impl_pointer
    m_future_pimpl;
  };  
  
#undef BOOST_COROUTINE_gen_future_assigner
#undef BOOST_COROUTINE_tuple_param_n

  template<typename Future, typename Functor, typename CoroutineSelf>
  Future call_impl(Functor fun, const CoroutineSelf& coro_self) {
    Future future(coro_self);
    fun(callback<Future>(future));
    return future;
  }

} } }

#endif
