//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2014 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_DETAIL_GET_TABLE_HPP
#define HPX_UTIL_DETAIL_GET_TABLE_HPP

#include <hpx/config.hpp>
#include <boost/mpl/identity.hpp>

namespace hpx { namespace util { namespace detail
{
    template <typename VTable, typename T>
    static VTable const* get_table() HPX_NOEXCEPT
    {
        static VTable const vtable = boost::mpl::identity<T>();
        return &vtable;
    }
}}}

#endif
