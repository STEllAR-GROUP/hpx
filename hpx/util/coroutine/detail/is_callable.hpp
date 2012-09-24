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

#ifndef HPX_COROUTINE_DETAIL_IS_CALLABLE_HPP_20060601
#define HPX_COROUTINE_DETAIL_IS_CALLABLE_HPP_20060601
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits.hpp>
#include <boost/mpl/logical.hpp>
#include <boost/mpl/has_xxx.hpp>
namespace hpx { namespace util { namespace coroutines { namespace detail 
{
  template<typename T>
  struct is_function_pointer :
    boost::mpl::and_<
    boost::is_pointer<T>,
    boost::is_function<typename boost::remove_pointer<T>::type > > {
    typedef is_function_pointer<T> type;
  };

  BOOST_MPL_HAS_XXX_TRAIT_DEF(result_type)
  BOOST_MPL_HAS_XXX_TRAIT_DEF(result)

  template<typename T>
  struct is_functor :
    boost::mpl::or_<typename has_result_type<T>::type,
            typename has_result<T>::type>
  {
    typedef is_functor<T> type;
  };

  template<typename T>
  struct is_callable : boost::mpl::or_<
    is_functor<T>,
    is_function_pointer<T> >::type {
    typedef is_callable<T> type;
  };
}}}}

#endif
