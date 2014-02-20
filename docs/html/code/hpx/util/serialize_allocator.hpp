//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_SERIALIZE_ALLOCATOR_FEB_19_2014_0711PM)
#define HPX_UTIL_SERIALIZE_ALLOCATOR_FEB_19_2014_0711PM

#include <hpx/hpx_fwd.hpp>
#include <boost/serialization/serialization.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace boost { namespace serialization
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Archive, typename T>
    void save(Archive& ar, std::allocator<T> const&, unsigned int) {}

    ///////////////////////////////////////////////////////////////////////////
    template <typename Archive, typename T>
    void load(Archive& ar, std::allocator<T>&, unsigned int) {}
}}

#endif
