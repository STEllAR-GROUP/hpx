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

#ifndef HPX_RUNTIME_COROUTINE_DETAIL_MAKE_TUPLE_TRAITS_HPP
#define HPX_RUNTIME_COROUTINE_DETAIL_MAKE_TUPLE_TRAITS_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/coroutine/tuple_traits.hpp>

#include <boost/mpl/vector.hpp>

namespace hpx { namespace coroutines { namespace detail
{
    /*
     * Given a mpl::vector, returns a nullary metafunction
     * describing a tuple of all types in the vector.
     * NOTE this is just wrong because it should work for all mpl
     * sequences, not just vectors. But it is in detail, so leave it
     * as is. Eventually it will be replaced by Boost.Fusion.
     * @p type is a tuple of all types in TypeList.
     * TypeList is one of mpl::vector0, mpl::vector1, etc.
     */
    template <typename TypeList>
    struct make_tuple_traits;

    template <>
    struct make_tuple_traits<boost::mpl::vector0<> >
    {
        typedef tuple_traits<> type;
    };

    template <typename A0>
    struct make_tuple_traits<boost::mpl::vector1<A0> >
    {
        typedef tuple_traits<A0> type;
    };
}}}

#endif /*HPX_RUNTIME_COROUTINE_DETAIL_MAKE_TUPLE_TRAITS_HPP*/
