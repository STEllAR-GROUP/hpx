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

#ifndef BOOST_COROUTINE_ARGUMENT_PACKER_HPP_20060601
#define BOOST_COROUTINE_ARGUMENT_PACKER_HPP_20060601
#include <boost/preprocessor/control/if.hpp>
#include <boost/preprocessor/punctuation/comma_if.hpp>
#include <boost/preprocessor/repetition.hpp>
#include <boost/preprocessor/tuple/eat.hpp>

namespace boost { namespace coroutines { namespace detail {
#define BOOST_COROUTINE_DETAIL_TEMPLATE_PARAMETERS(n)  \
  template<BOOST_PP_ENUM_PARAMS(n, typename T)> \
/**/

#define BOOST_COROUTINE_ARGUMENT_PACKER(z, n, parm_tuple)        \
  BOOST_PP_IF(n,                                                 \
	      BOOST_COROUTINE_DETAIL_TEMPLATE_PARAMETERS ,              \
	      BOOST_PP_TUPLE_EAT(1) )(n)                         \
  BOOST_PP_TUPLE_ELEM(3, 0, parm_tuple)                          \
    (BOOST_PP_ENUM_BINARY_PARAMS(n, T, arg)) {                   \
    typedef BOOST_PP_TUPLE_ELEM(3, 2, parm_tuple) parm_type;     \
    return BOOST_PP_TUPLE_ELEM(3, 1, parm_tuple)                 \
                             (parm_type                          \
			      (BOOST_PP_ENUM_PARAMS(n, arg)));   \
  }                                                              \
/**/


#define BOOST_COROUTINE_ARGUMENT_PACKER_EX(z, n, parm_tuple)     \
  template<typename Arg                                          \
    BOOST_PP_COMMA_IF(n)                                         \
    BOOST_PP_ENUM_PARAMS(n, typename T)>                         \
  BOOST_PP_TUPLE_ELEM(3, 0, parm_tuple)                          \
  (Arg arg                                                       \
   BOOST_PP_COMMA_IF(n)                                          \
   BOOST_PP_ENUM_BINARY_PARAMS(n, T, arg)) {                     \
    typedef BOOST_PP_TUPLE_ELEM(3, 2, parm_tuple) parm_type;     \
    return BOOST_PP_TUPLE_ELEM(3, 1, parm_tuple)( arg, parm_type \
     (BOOST_PP_ENUM_PARAMS(n, arg)));                            \
  }                                                              \
/**/

} } }
#endif
