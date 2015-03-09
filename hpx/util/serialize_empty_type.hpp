//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_SERIALIZE_EMPTY_TYPE_JAN_28_2013_0623AM)
#define HPX_UTIL_SERIALIZE_EMPTY_TYPE_JAN_28_2013_0623AM

#include <hpx/config.hpp>
#include <hpx/serialization/serialize.hpp>

#include <boost/type_traits/is_empty.hpp>
#include <boost/utility/enable_if.hpp>

namespace hpx { namespace serialization
{
    // This is a default implementation of serialization which kicks in for
    // empty types.
    template <typename T>
    BOOST_FORCEINLINE typename boost::enable_if<boost::is_empty<T> >::type
    serialize(output_archive&, T&, unsigned int const) {}

    template <typename T>
    BOOST_FORCEINLINE typename boost::enable_if<boost::is_empty<T> >::type
    serialize(input_archive&, T&, unsigned int const) {}
}}

#endif
