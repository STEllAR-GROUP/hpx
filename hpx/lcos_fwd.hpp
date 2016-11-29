//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_LCOS_FWD_HPP
#define HPX_LCOS_FWD_HPP

#include <hpx/config.hpp>
#include <hpx/traits/promise_local_result.hpp>
#include <hpx/traits/promise_remote_result.hpp>

namespace hpx
{
    /// \namespace lcos
    namespace lcos
    {
        namespace detail
        {
            struct future_data_refcnt_base;
        }

        class HPX_EXPORT base_lco;

        template <typename Result, typename RemoteResult =
            typename traits::promise_remote_result<Result>::type>
        class base_lco_with_value;

        template <>
        class base_lco_with_value<void, void>;

        template <typename Result, typename RemoteResult =
            typename traits::promise_remote_result<Result>::type>
        class promise;

        template <typename Action,
            typename Result = typename traits::promise_local_result<
                typename Action::remote_result_type>::type,
            bool DirectExecute = Action::direct_execution::value>
        class packaged_action;

        template <typename R>
        class future;

        template <typename R>
        class shared_future;

        template <typename ValueType>
        struct object_semaphore;

        namespace server
        {
            template <typename ValueType>
            struct object_semaphore;
        }

        namespace local
        {
            class barrier;

            template <typename R>
            class promise;
        }
    }

    using lcos::future;
    using lcos::shared_future;
    using lcos::promise;
}

#endif
