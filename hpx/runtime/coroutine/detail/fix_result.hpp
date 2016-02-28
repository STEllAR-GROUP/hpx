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

#ifndef HPX_RUNTIME_COROUTINE_DETAIL_FIX_RESULT_HPP
#define HPX_RUNTIME_COROUTINE_DETAIL_FIX_RESULT_HPP

#include <hpx/config.hpp>

#include <boost/tuple/tuple.hpp>

#include <type_traits>

namespace hpx { namespace coroutines { namespace detail
{
    template <typename Traits>
    inline void fix_result(const typename Traits::as_tuple&,
        typename std::enable_if<Traits::length == 0>::type * = 0)
    {}

    template <typename Traits>
    inline typename Traits::template at<0>::type
    fix_result(const typename Traits::as_tuple& x,
        typename std::enable_if<Traits::length == 1>::type * = 0)
    {
        using boost::get;
        return get<0>(x);
    }

    template <typename Traits>
    inline typename Traits::as_tuple
    fix_result(const typename Traits::as_tuple& x,
        typename std::enable_if< (Traits::length > 1) > ::type* = 0)
    {
        return x;
    }
}}}

#endif /*HPX_RUNTIME_COROUTINE_DETAIL_FIX_RESULT_HPP*/
