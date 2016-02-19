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

#ifndef HPX_COROUTINE_DETAIL_COROUTINE_TRAITS_HPP_20060613
#define HPX_COROUTINE_DETAIL_COROUTINE_TRAITS_HPP_20060613

#include <boost/type_traits/function_traits.hpp>
#include <boost/type_traits/is_same.hpp>

#include <hpx/util/coroutine/tuple_traits.hpp>
#include <hpx/util/coroutine/detail/signature.hpp>
#include <hpx/util/coroutine/detail/yield_result_type.hpp>

namespace hpx { namespace util { namespace coroutines { namespace detail
{
  template <typename T>
  struct as_tuple {
    typedef typename T::as_tuple type;
  };

  // This trait class is used to compute
  // all nested typedefs of coroutines given
  // a signature in the form 'result_type(parm1, ... parmn)'.
  template <typename Signature>
  struct coroutine_traits {
  private:
    typedef typename boost::function_traits<Signature>::result_type
      signature_result_type;

  public:
    typedef typename boost::mpl::eval_if<
        is_tuple_traits<signature_result_type>,
        as_tuple<signature_result_type>,
        boost::mpl::identity<signature_result_type>
    >::type result_type;

    typedef typename boost::mpl::eval_if<
        is_tuple_traits<signature_result_type>,
        boost::mpl::identity<signature_result_type>,
        boost::mpl::if_<
            boost::is_same<signature_result_type, void>,
            tuple_traits<>,
            tuple_traits<signature_result_type>
        >
    >::type result_slot_traits;

    typedef typename result_slot_traits::as_tuple result_slot_type;

    typedef typename detail::make_tuple_traits<
        typename detail::signature<Signature>::type
    >::type arg_slot_traits;

    typedef typename arg_slot_traits::as_tuple arg_slot_type;

    typedef typename arg_slot_traits::as_result yield_result_type;
  };
}}}}

#endif
