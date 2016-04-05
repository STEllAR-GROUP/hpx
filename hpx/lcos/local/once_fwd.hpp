//  Copyright (c) 2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_LCOS_LOCAL_ONCE_FWD_HPP
#define HPX_LCOS_LOCAL_ONCE_FWD_HPP

namespace hpx { namespace lcos { namespace local
{
    // call_once support
    struct once_flag;

    template <typename F, typename ...Args>
    void call_once(once_flag& flag, F&& f, Args&&... args);
}}}

#define HPX_ONCE_INIT hpx::lcos::local::once_flag()

#endif /*HPX_LCOS_LOCAL_ONCE_FWD_HPP*/
