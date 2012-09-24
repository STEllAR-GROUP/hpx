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

#ifndef HPX_COROUTINE_DETAIL_SIGNATURE_HPP_20060609
#define HPX_COROUTINE_DETAIL_SIGNATURE_HPP_20060609

#include <boost/preprocessor/repetition.hpp>
#include <hpx/util/coroutine/detail/arg_max.hpp>
namespace hpx { namespace util { namespace coroutines { namespace detail
{
  /*
   * Derived from an  mpl::vector describing
   * 'Function' arguments types.
   */
  template<typename Function>
  struct signature;

  /*
   * Generate specializations for the signature trait class.
   */
#define HPX_COROUTINE_SIGNATURE_GENERATOR(z, n, unused)        \
  template <class R BOOST_PP_ENUM_TRAILING_PARAMS(n, class A)>   \
  struct signature<R( BOOST_PP_ENUM_PARAMS(n,A) ) >              \
    : boost::mpl::BOOST_PP_CAT(vector,n)<                        \
    BOOST_PP_ENUM_PARAMS(n,A)                                    \
    > {};                                                        \
/**/

BOOST_PP_REPEAT(HPX_COROUTINE_ARG_MAX, HPX_COROUTINE_SIGNATURE_GENERATOR, ~)

#undef HPX_COROUTINE_SIGNATURE_GENERATOR

}}}}

#endif
