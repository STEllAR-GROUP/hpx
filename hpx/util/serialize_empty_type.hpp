//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_SERIALIZE_EMPTY_TYPE_JAN_28_2013_0623AM)
#define HPX_UTIL_SERIALIZE_EMPTY_TYPE_JAN_28_2013_0623AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/serialization.hpp>
#include <boost/type_traits/is_empty.hpp>
#include <boost/utility/enable_if.hpp>

namespace boost { namespace serialization
{
    // This is a default implementation of serialization which kicks in for
    // empty types.
    template <typename T>
    BOOST_FORCEINLINE typename boost::enable_if<boost::is_empty<T> >::type
    serialize(hpx::util::portable_binary_oarchive&, T&, unsigned int const) {}

    template <typename T>
    BOOST_FORCEINLINE typename boost::enable_if<boost::is_empty<T> >::type
    serialize(hpx::util::portable_binary_iarchive&, T&, unsigned int const) {}
}}

#endif
