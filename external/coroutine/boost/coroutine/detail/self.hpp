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

#ifndef BOOST_COROUTINE_DETAIL_SELF_HPP_20060809
#define BOOST_COROUTINE_DETAIL_SELF_HPP_20060809
#include <boost/noncopyable.hpp>
#include <boost/coroutine/detail/fix_result.hpp>
#include <boost/coroutine/detail/coroutine_accessor.hpp>
#include <boost/coroutine/detail/signal.hpp>
#include <boost/detail/workaround.hpp>
#include <boost/coroutine/detail/noreturn.hpp>

namespace boost { namespace coroutines { namespace detail {

#if BOOST_WORKAROUND(BOOST_MSVC, BOOST_TESTED_AT(1400))
#define BOOST_COROUTINE_DEDUCED_TYPENAME_DEFAULT
#else
#define BOOST_COROUTINE_DEDUCED_TYPENAME_DEFAULT BOOST_DEDUCED_TYPENAME
#endif

#if BOOST_WORKAROUND(BOOST_MSVC, BOOST_TESTED_AT(1400))
#define BOOST_COROUTINE_VCPP80_WORKAROUND 
#else
//for now, define this unconditionally for testing.
#define BOOST_COROUTINE_VCPP80_WORKAROUND 
#endif

  template<typename Coroutine>
  class coroutine_self :boost::noncopyable {
  public:
    typedef Coroutine coroutine_type;
    typedef coroutine_self<coroutine_type> type;
    friend struct detail::coroutine_accessor;

    typedef BOOST_DEDUCED_TYPENAME coroutine_type
    ::impl_type impl_type;

    // Note, no reference counting here.
    typedef impl_type * impl_ptr;

    typedef BOOST_DEDUCED_TYPENAME coroutine_type
    ::result_type result_type;

    typedef BOOST_DEDUCED_TYPENAME coroutine_type
    ::result_slot_type result_slot_type;

    typedef BOOST_DEDUCED_TYPENAME coroutine_type
    ::yield_result_type yield_result_type;

    typedef BOOST_DEDUCED_TYPENAME coroutine_type
    ::result_slot_traits result_slot_traits;

    typedef BOOST_DEDUCED_TYPENAME coroutine_type
    ::arg_slot_type arg_slot_type;

    typedef BOOST_DEDUCED_TYPENAME coroutine_type
    ::arg_slot_traits arg_slot_traits;

    typedef BOOST_DEDUCED_TYPENAME coroutine_type
    ::yield_traits yield_traits;
#ifndef BOOST_COROUTINE_VCPP80_WORKAROUND 
#   define BOOST_COROUTINE_param_with_default(z, n, type_prefix)    \
    BOOST_DEDUCED_TYPENAME call_traits                              \
    <BOOST_PP_CAT(BOOST_PP_CAT(type_prefix, n), _type)>::param_type \
    BOOST_PP_CAT(arg, n) =                                          \
    BOOST_PP_CAT(BOOST_PP_CAT(type_prefix, n), _type)()             \
/**/

    yield_result_type yield
    (BOOST_PP_ENUM
       (BOOST_COROUTINE_ARG_MAX,
	BOOST_COROUTINE_param_with_default,
	BOOST_DEDUCED_TYPENAME yield_traits::arg))
    {
      return yield_impl
	(BOOST_DEDUCED_TYPENAME 
	 coroutine_type::result_slot_type
	 (BOOST_PP_ENUM_PARAMS
	  (BOOST_COROUTINE_ARG_MAX, 
	   arg)));
    }
    
    template<typename Target>
    yield_result_type yield_to
    (Target& target
     BOOST_PP_ENUM_TRAILING
     (BOOST_COROUTINE_ARG_MAX,
      BOOST_COROUTINE_param_with_default,
      typename Target::arg)) 
    {
      typedef BOOST_DEDUCED_TYPENAME Target::arg_slot_type slot_type;
      return yield_to_impl
	(target, slot_type(BOOST_PP_ENUM_PARAMS
	  (BOOST_COROUTINE_ARG_MAX, 
	   arg)));
    }
#else
        
    /* 
     * VC8.0 can't handle the call_traits meta-invocation inside
     * a function parameter list (except when it does, see operator()). 
     * Splitting it in separate typedefs
     * fixes the problem.
     */
#define BOOST_COROUTINE_param_typedef(z, n, prefix_tuple)  \
    typedef BOOST_DEDUCED_TYPENAME                   \
    call_traits<                                     \
    BOOST_PP_CAT                                     \
    (BOOST_PP_CAT                                    \
     (BOOST_PP_TUPLE_ELEM(2, 0, prefix_tuple), n),   \
     _type)                                          \
      >::param_type                                  \
    BOOST_PP_CAT                                     \
    (BOOST_PP_CAT                                    \
     (BOOST_PP_TUPLE_ELEM(2, 1, prefix_tuple), n),   \
     _type)                                          \
     /**/;
    
    /*
     * Generate lines like this:
     * 'typedef typename call_traits<typename coroutine_type::yield_traits::argN_type>::param_type yield_call_argN_type;'
     */
    BOOST_PP_REPEAT(BOOST_COROUTINE_ARG_MAX, 
		    BOOST_COROUTINE_param_typedef, 
		    (BOOST_DEDUCED_TYPENAME 
		     coroutine_type::yield_traits::arg, yield_call_arg));

    /*
     * Generate lines like this:
     * 'typedef typename call_traits<typename coroutine_type::argN_type>::param_type call_argN_type;'
     */
    BOOST_PP_REPEAT(BOOST_COROUTINE_ARG_MAX,
		    BOOST_COROUTINE_param_typedef,
		    (BOOST_DEDUCED_TYPENAME 
		     coroutine_type::arg, call_arg));

#undef BOOST_COROUTINE_param_typedef
#undef  BOOST_COROUTINE_param_with_default
#define BOOST_COROUTINE_param_with_default(z, n, prefix_tuple) \
    BOOST_PP_CAT(BOOST_PP_CAT                                  \
		 (BOOST_PP_TUPLE_ELEM(2, 0, prefix_tuple),     \
		  n), _type)                                   \
      BOOST_PP_CAT(arg, n) =                                   \
    BOOST_PP_CAT(BOOST_PP_CAT                                  \
		 (BOOST_PP_TUPLE_ELEM(2, 1, prefix_tuple),     \
		  n), _type)()                                 \
      /**/
     
    yield_result_type yield
    (BOOST_PP_ENUM(BOOST_COROUTINE_ARG_MAX,
		   BOOST_COROUTINE_param_with_default,
		   (yield_call_arg ,
		    BOOST_COROUTINE_DEDUCED_TYPENAME_DEFAULT 
		    coroutine_type::yield_traits::arg))) 
    {
      return yield_impl
      (result_slot_type
       (BOOST_PP_ENUM_PARAMS(BOOST_COROUTINE_ARG_MAX,arg)));
    }

    template<typename Target>
    yield_result_type yield_to
    (Target& target
     BOOST_PP_ENUM_TRAILING
     (BOOST_COROUTINE_ARG_MAX,
      BOOST_COROUTINE_param_with_default,
      (BOOST_DEDUCED_TYPENAME Target::self::call_arg, 
       BOOST_COROUTINE_DEDUCED_TYPENAME_DEFAULT Target::arg)))
    {
      typedef typename Target::arg_slot_type type;
      return yield_to_impl
	(target, type
	 (BOOST_PP_ENUM_PARAMS
	  (BOOST_COROUTINE_ARG_MAX, arg)));
    }
#endif

#undef  BOOST_COROUTINE_param_with_default

    BOOST_COROUTINE_NORETURN(void exit()) {
      m_pimpl -> exit_self();
      abort();
    }

    yield_result_type result() {
      return detail::fix_result<
	BOOST_DEDUCED_TYPENAME
	coroutine_type::arg_slot_traits>(*m_pimpl->args());
    }

    bool pending() const {
      BOOST_ASSERT(m_pimpl);
      return m_pimpl->pending();
    }
  private:
    coroutine_self(impl_type * pimpl, detail::init_from_impl_tag) :
      m_pimpl(pimpl) {}

    yield_result_type yield_impl(BOOST_DEDUCED_TYPENAME 
				 coroutine_type::result_slot_type result) {
      typedef BOOST_DEDUCED_TYPENAME
	coroutine_type::result_slot_type slot_type;

      BOOST_ASSERT(m_pimpl);

      this->m_pimpl->bind_result(&result);
      this->m_pimpl->yield();    
      return detail::fix_result<
	BOOST_DEDUCED_TYPENAME
	coroutine_type::arg_slot_traits>(*m_pimpl->args());
    }

    template<typename TargetCoroutine>
    yield_result_type yield_to_impl(TargetCoroutine& target, 
			   BOOST_DEDUCED_TYPENAME TargetCoroutine
			   ::arg_slot_type args) {
      BOOST_ASSERT(m_pimpl);

      coroutine_accessor::get_impl(target)->bind_args(&args);
      coroutine_accessor::get_impl(target)->bind_result_pointer(m_pimpl->result_pointer());    

      this->m_pimpl->yield_to(*coroutine_accessor::get_impl(target));

      return detail::fix_result<
	BOOST_DEDUCED_TYPENAME
	coroutine_type::arg_slot_traits>(*m_pimpl->args());
    }

    impl_ptr get_impl() {
      return m_pimpl;
    }
    impl_ptr m_pimpl;
  };
} } }

#endif
