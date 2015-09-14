//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_SERIALIZATION_ALLOCATOR_FEB_19_2014_0711PM)
#define HPX_SERIALIZATION_ALLOCATOR_FEB_19_2014_0711PM

#include <hpx/config.hpp>
#include <hpx/runtime/serialization/serialize.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace serialization
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Archive, typename T>
    void save(Archive&, std::allocator<T> const&, unsigned int) {}

    ///////////////////////////////////////////////////////////////////////////
    template <typename Archive, typename T>
    void load(Archive&, std::allocator<T>&, unsigned int) {}
}}

#endif
