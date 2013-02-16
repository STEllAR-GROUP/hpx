//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/actions/action_support.hpp>
#include <hpx/runtime/actions/compression/zlib_serialization_filter.hpp>
#include <hpx/runtime/actions/guid_initialization.hpp>

///////////////////////////////////////////////////////////////////////////////
BOOST_CLASS_EXPORT(hpx::actions::zlib_serialization_filter);
HPX_REGISTER_BASE_HELPER(hpx::actions::zlib_serialization_filter,
    zlib_serialization_filter);

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace traits
{
    template <>
    struct needs_guid_initialization<hpx::actions::zlib_serialization_filter>
        : boost::mpl::false_
    {};
}}

namespace boost { namespace archive { namespace detail { namespace extra_detail
{
    template <>
    struct init_guid<hpx::actions::zlib_serialization_filter>;
}}}}

namespace hpx { namespace actions
{
    zlib_serialization_filter::~zlib_serialization_filter()
    {
        detail::guid_initialization<zlib_serialization_filter>();
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
