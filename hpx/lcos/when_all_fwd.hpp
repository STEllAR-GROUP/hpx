//  Copyright (c) 2018 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_WHEN_ALL_FWD_HPP)
#define HPX_LCOS_WHEN_ALL_FWD_HPP

#include <hpx/config.hpp>
#include <hpx/lcos_fwd.hpp>

#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos
{
    // special forwarding function to break #include dependencies
    HPX_API_EXPORT hpx::future<void> when_all_fwd(
        std::vector<hpx::future<void>>&);

    HPX_API_EXPORT hpx::future<void> when_all_fwd(
        std::vector<hpx::future<void>>&&);
}}

#endif

