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


// Copyright David Abrahams 2004. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Slightly modified for inclusion on coroutines/detail library.
#ifndef HPX_COROUTINE_DETAIL_HAS_SWAP_HPP_20060709
#define HPX_COROUTINE_DETAIL_HAS_SWAP_HPP_20060709

#include <hpx/config.hpp>

#include <boost/mpl/bool.hpp>
#include <boost/detail/workaround.hpp>

namespace hpx { namespace util { namespace coroutines
{
namespace has_swap_
{
  // any soaks up implicit conversions and makes the following
  // operator++ less-preferred than any other such operator which
  // might be found via ADL.
  struct anything { template <class T> anything(T const&); };
  struct no_swap
  {
      char (& operator,(char) )[2];
  };
  no_swap swap(anything,anything);

#if defined(HPX_MSVC)
# pragma warning(push)
# pragma warning(disable: 4675)
// function found through argument dependent lookup -- duh!
#endif
  template <class T>
  struct has_swap_impl
  {
      static T& x;

      HPX_STATIC_CONSTEXPR bool value = sizeof(swap(x,x),'x') == 1;

      typedef boost::mpl::bool_<value> type;
  };
}
template <class T>
struct has_swap
  : has_swap_::has_swap_impl<T>::type
{};
#if defined(HPX_MSVC)
# pragma warning(pop)
#endif

}}}

#endif // include guard
