//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_SERVER_CREATE_COMPONENT_FWD_JUN_22_2015_0206PM)
#define HPX_COMPONENTS_SERVER_CREATE_COMPONENT_FWD_JUN_22_2015_0206PM

#include <hpx/exception.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/util/unique_function.hpp>

#include <cstdint>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    /// Create arrays of components using their default constructor
    template <typename Component>
    naming::gid_type create(std::size_t count);

    template <typename Component>
    naming::gid_type create(
        util::unique_function_nonser<void(void*)> const& ctor);

    template <typename Component>
    naming::gid_type create(naming::gid_type const& gid,
        util::unique_function_nonser<void(void*)> const& ctor);
}}}

#endif

