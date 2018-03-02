//  Copyright (c) 2018 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/when_all.hpp>
#include <hpx/lcos/when_all_fwd.hpp>

#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos
{
    // special forwarding function to break #include dependencies
    hpx::future<void> when_all_fwd(std::vector<hpx::future<void>>& tasks)
    {
        return hpx::when_all(tasks);
    }

    hpx::future<void> when_all_fwd(std::vector<hpx::future<void>>&& tasks)
    {
        return hpx::when_all(std::move(tasks));
    }
}}
