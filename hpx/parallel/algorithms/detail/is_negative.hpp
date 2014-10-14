//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_DETAIL_IS_NEGATIVE_JUL_2014_01_0148PM)
#define HPX_PARALLEL_DETAIL_IS_NEGATIVE_JUL_2014_01_0148PM

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_signed.hpp>
#include <boost/type_traits/is_unsigned.hpp>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1) { namespace detail
{
    // main template represents non-integral types (raises error)
    template <typename Size, typename Enable = void>
    struct is_negative;

    // signed integral values may be negative
    template <typename Size>
    struct is_negative<Size,
        typename boost::enable_if<boost::is_signed<Size> >::type>
    {
        static bool call(Size const& size) { return  size < 0; }
    };

    // unsigned integral values are never negative
    template <typename Size>
    struct is_negative<Size,
        typename boost::enable_if<boost::is_unsigned<Size> >::type>
    {
        static bool call(Size const&) { return false; }
    };
}}}}

#endif


