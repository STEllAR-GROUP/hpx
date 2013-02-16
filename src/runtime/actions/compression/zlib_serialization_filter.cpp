//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/actions/action_support.hpp>
#include <hpx/runtime/actions/compression/zlib_serialization_filter.hpp>
#include <hpx/runtime/actions/guid_initialization.hpp>
#include <hpx/util/void_cast.hpp>

///////////////////////////////////////////////////////////////////////////////
HPX_SERIALIZATION_REGISTER_TYPE_DEFINITION(hpx::actions::zlib_serialization_filter);
HPX_REGISTER_BASE_HELPER(hpx::actions::zlib_serialization_filter,
    zlib_serialization_filter);

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace actions
{
    zlib_serialization_filter::~zlib_serialization_filter()
    {
        detail::guid_initialization<zlib_serialization_filter>();
    }

    void zlib_serialization_filter::register_base()
    {
        util::void_cast_register_nonvirt<
            zlib_serialization_filter, util::binary_filter>();
    }

    std::size_t zlib_serialization_filter::load(void* address,
        void const* src, std::size_t count)
    {
        std::memcpy(address, src, count);
        return count;
    }

    std::size_t zlib_serialization_filter::save(void* dest,
        void const* address, std::size_t count)
    {
        std::memcpy(dest, address, count);
        return count;
    }
}}
