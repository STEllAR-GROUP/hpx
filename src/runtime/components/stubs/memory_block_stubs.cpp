//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/async.hpp>
#include <hpx/apply.hpp>
#include <hpx/runtime/components/server/memory.hpp>
#include <hpx/runtime/components/stubs/memory_block.hpp>

namespace hpx { namespace components { namespace stubs
{

lcos::future<components::memory_block_data> memory_block::get_data_async(
    naming::id_type const& targetgid)
{
    // Create a future, execute the required action,
    // we simply return the initialized future, the caller needs
    // to call get() on the return value to obtain the result
    typedef server::detail::memory_block::get_action action_type;
    return hpx::async<action_type>(targetgid);
}

lcos::future<components::memory_block_data> memory_block::get_data_async(
    naming::id_type const& targetgid,
    components::memory_block_data const& cfg)
{
    // Create a future, execute the required action,
    // we simply return the initialized future, the caller needs
    // to call get() on the return value to obtain the result
    typedef server::detail::memory_block::get_config_action action_type;
    return hpx::async<action_type>(targetgid, cfg);
}

lcos::future<components::memory_block_data> memory_block::checkout_async(
    naming::id_type const& targetgid)
{
    // Create a future, execute the required action,
    // we simply return the initialized future, the caller needs
    // to call get() on the return value to obtain the result
    typedef server::detail::memory_block::checkout_action action_type;
    return hpx::async<action_type>(targetgid);
}

lcos::future<naming::id_type>
memory_block::clone_async(naming::id_type const& targetgid)
{
    // Create a future, execute the required action,
    // we simply return the initialized future, the caller needs
    // to call get() on the return value to obtain the result
    typedef server::detail::memory_block::clone_action action_type;
    return hpx::async<action_type>(targetgid);
}

}}}
