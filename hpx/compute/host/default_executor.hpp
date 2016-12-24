//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPUTE_HOST_DEFAULT_EXECUTOR_HPP
#define HPX_COMPUTE_HOST_DEFAULT_EXECUTOR_HPP

#include <hpx/config.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/traits/is_executor_v1.hpp>

#include <hpx/compute/host/target.hpp>

#include <vector>

namespace hpx { namespace compute { namespace host
{
    struct default_executor : hpx::parallel::executor_tag
    {
        default_executor(host::target& target)
 //         : target_(target)
        {}

        template <typename F, typename ... Ts>
        void apply_execute(F && f, Ts &&... ts)
        {
        }

        template <typename F, typename ... Ts>
        hpx::future<void> async_execute(F && f, Ts &&... ts)
        {
            return hpx::future<void>();
        }

        template <typename F, typename ... Ts>
        static void execute(F && f, Ts &&... ts)
        {
        }

        template <typename F, typename Shape, typename ... Ts>
        static std::vector<hpx::future<void> >
        bulk_async_execute(F && f, Shape const& shape, Ts &&... ts)
        {
            return std::vector<hpx::future<void> >();
        }

        template <typename F, typename Shape, typename ... Ts>
        static void bulk_execute(F && f, Shape const& shape, Ts &&... ts)
        {
        }

    private:
 //       host::target& target_;
    };
}}}

#endif
