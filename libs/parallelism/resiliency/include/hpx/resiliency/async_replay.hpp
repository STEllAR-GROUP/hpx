//  Copyright (c) 2019 National Technology & Engineering Solutions of Sandia,
//                     LLC (NTESS).
//  Copyright (c) 2018-2020 Hartmut Kaiser
//  Copyright (c) 2018-2019 Adrian Serio
//  Copyright (c) 2019 Nikunj Gupta
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/resiliency/config.hpp>
#include <hpx/resiliency/resiliency_cpos.hpp>
#include <hpx/resiliency/util.hpp>

#include <hpx/functional/detail/invoke.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/modules/async_local.hpp>
#include <hpx/type_support/pack.hpp>

#include <cstddef>
#include <exception>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace resiliency { namespace experimental {

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        ///////////////////////////////////////////////////////////////////////
        template <typename Result, typename Pred, typename F, typename Tuple>
        struct async_replay_helper
          : std::enable_shared_from_this<
                async_replay_helper<Result, Pred, F, Tuple>>
        {
            template <typename Pred_, typename F_, typename Tuple_>
            async_replay_helper(Pred_&& pred, F_&& f, Tuple_&& tuple)
              : pred_(std::forward<Pred_>(pred))
              , f_(std::forward<F_>(f))
              , t_(std::forward<Tuple_>(tuple))
            {
            }

            template <std::size_t... Is>
            hpx::future<Result> invoke(hpx::util::index_pack<Is...>)
            {
                return hpx::async(f_, std::get<Is>(t_)...);
            }

            hpx::future<Result> call(std::size_t n)
            {
                // launch given function asynchronously
                hpx::future<Result> f = invoke(hpx::util::make_index_pack<
                    std::tuple_size<Tuple>::value>{});

                // attach a continuation that will relaunch the task, if
                // necessary
                auto this_ = this->shared_from_this();
                return f.then(hpx::launch::sync,
                    [this_ = std::move(this_), n](hpx::future<Result>&& f) {
                        if (f.has_exception())
                        {
                            // rethrow abort_replay_exception, if caught
                            auto ex = rethrow_on_abort_replay(f);

                            // execute the task again if an error occurred and
                            // this was not the last attempt
                            if (n != 0)
                            {
                                return this_->call(n - 1);
                            }

                            // rethrow exception if the number of replays has
                            // been exhausted
                            std::rethrow_exception(ex);
                        }

                        auto&& result = f.get();

                        if (!HPX_INVOKE(this_->pred_, result))
                        {
                            // execute the task again if an error occurred and
                            // this was not the last attempt
                            if (n != 0)
                            {
                                return this_->call(n - 1);
                            }

                            // throw aborting exception as attempts were
                            // exhausted
                            throw abort_replay_exception();
                        }

                        if (n != 0)
                        {
                            // return result
                            return hpx::make_ready_future(std::move(result));
                        }

                        // throw aborting exception as attempts were
                        // exhausted
                        throw abort_replay_exception();
                    });
            }

            Pred pred_;
            F f_;
            Tuple t_;
        };

        template <typename Result, typename Pred, typename F, typename... Ts>
        std::shared_ptr<async_replay_helper<Result,
            typename std::decay<Pred>::type, typename std::decay<F>::type,
            std::tuple<typename std::decay<Ts>::type...>>>
        make_async_replay_helper(Pred&& pred, F&& f, Ts&&... ts)
        {
            using return_type = async_replay_helper<Result,
                typename std::decay<Pred>::type, typename std::decay<F>::type,
                std::tuple<typename std::decay<Ts>::type...>>;

            return std::make_shared<return_type>(std::forward<Pred>(pred),
                std::forward<F>(f), std::make_tuple(std::forward<Ts>(ts)...));
        }
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    // Asynchronously launch given function \a f. Verify the result of
    // those invocations using the given predicate \a pred. Repeat launching
    // on error exactly \a n times (except if abort_replay_exception is thrown).
    template <typename Pred, typename F, typename... Ts>
    hpx::future<
        typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type>
    tag_invoke(
        async_replay_validate_t, std::size_t n, Pred&& pred, F&& f, Ts&&... ts)
    {
        using result_type =
            typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type;

        auto helper = detail::make_async_replay_helper<result_type>(
            std::forward<Pred>(pred), std::forward<F>(f),
            std::forward<Ts>(ts)...);

        return helper->call(n);
    }

    ///////////////////////////////////////////////////////////////////////////
    // Asynchronously launch given function \a f. Repeat launching
    // on error exactly \a n times (except if abort_replay_exception is thrown).
    template <typename F, typename... Ts>
    hpx::future<
        typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type>
    tag_invoke(async_replay_t, std::size_t n, F&& f, Ts&&... ts)
    {
        using result_type =
            typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type;

        auto helper = detail::make_async_replay_helper<result_type>(
            detail::replay_validator{}, std::forward<F>(f),
            std::forward<Ts>(ts)...);

        return helper->call(n);
    }
}}}    // namespace hpx::resiliency::experimental
