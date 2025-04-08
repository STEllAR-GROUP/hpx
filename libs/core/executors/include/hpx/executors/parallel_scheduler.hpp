// Copyright (c) 2025 Sai Charan Arvapally
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/errors/try_catch_exception_ptr.hpp>
#include <hpx/execution.hpp>
#include <hpx/execution/traits/executor_traits.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/modules/iterator_support.hpp>
#include <hpx/synchronization/stop_token.hpp>

#include <atomic>
#include <cstddef>
#include <exception>
#include <memory>
#include <optional>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx::execution::experimental {

    struct bulk_t;

    struct parallel_scheduler
    {
        using execution_category = parallel_execution_tag;

        parallel_scheduler() noexcept = default;

        parallel_scheduler(parallel_scheduler const&) noexcept = default;
        parallel_scheduler(parallel_scheduler&&) noexcept = default;
        parallel_scheduler& operator=(
            parallel_scheduler const&) noexcept = default;
        parallel_scheduler& operator=(parallel_scheduler&&) noexcept = default;

        friend bool operator==(
            parallel_scheduler const&, parallel_scheduler const&) noexcept
        {
            return true;
        }

        friend hpx::execution::experimental::forward_progress_guarantee
        tag_invoke(
            hpx::execution::experimental::get_forward_progress_guarantee_t,
            parallel_scheduler const&) noexcept
        {
            return hpx::execution::experimental::forward_progress_guarantee::
                parallel;
        }
    };

    namespace detail {
        namespace replaceability {

            struct storage
            {
                void* data;
                uint32_t size;
            };

            struct receiver
            {
                virtual ~receiver() = default;
                virtual void set_value() noexcept = 0;
                virtual void set_error(std::exception_ptr) noexcept = 0;
                virtual void set_stopped() noexcept = 0;
                virtual std::optional<hpx::stop_token>
                try_query_stop_token() noexcept
                {
                    return std::nullopt;
                }
            };

            struct bulk_item_receiver : receiver
            {
                virtual void start(uint32_t start, uint32_t end) noexcept = 0;
                virtual std::exception_ptr get_error() const noexcept
                {
                    return nullptr;
                }
            };

            struct parallel_scheduler
            {
                virtual ~parallel_scheduler() = default;
                virtual void schedule(receiver*, storage) noexcept = 0;
                virtual void schedule_bulk_chunked(
                    uint32_t n, bulk_item_receiver*, storage) noexcept = 0;
                virtual void schedule_bulk_unchunked(
                    uint32_t n, bulk_item_receiver*, storage) noexcept = 0;
            };

            template <typename Receiver>
            struct my_receiver : receiver
            {
                Receiver& r;
                my_receiver(Receiver& r_)
                  : r(r_)
                {
                }
                void set_value() noexcept override
                {
                    hpx::execution::experimental::set_value(r);
                }
                void set_error(std::exception_ptr ep) noexcept override
                {
                    hpx::execution::experimental::set_error(
                        HPX_MOVE(r), HPX_MOVE(ep));
                }
                void set_stopped() noexcept override
                {
                    hpx::execution::experimental::set_stopped(r);
                }
                std::optional<hpx::stop_token> try_query_stop_token() noexcept
                    override
                {
                    auto env = hpx::execution::experimental::get_env(r);
                    return hpx::execution::experimental::get_stop_token(env);
                }
            };

            template <typename Receiver, typename Shape, typename F>
            struct my_bulk_item_receiver : bulk_item_receiver
            {
                Receiver& r;
                F f;
                Shape shape;
                std::exception_ptr error_ptr = nullptr;

                my_bulk_item_receiver(Receiver& r_, F&& f_, Shape sh)
                  : r(r_)
                  , f(HPX_FORWARD(F, f_))
                  , shape(sh)
                {
                }

                void start(uint32_t start, uint32_t end) noexcept override
                {
                    try
                    {
                        if constexpr (std::is_integral_v<Shape>)
                        {
                            for (uint32_t i = start; i < end; ++i)
                            {
                                HPX_INVOKE(f, static_cast<Shape>(i));
                            }
                        }
                        else
                        {
                            auto it = std::next(hpx::util::begin(shape), start);
                            for (uint32_t i = start; i < end; ++i, ++it)
                            {
                                HPX_INVOKE(f, *it);
                            }
                        }
                    }
                    catch (...)
                    {
                        error_ptr = std::current_exception();
                    }
                }

                std::exception_ptr get_error() const noexcept override
                {
                    return error_ptr;
                }

                void set_value() noexcept override
                {
                    hpx::execution::experimental::set_value(r);
                }

                void set_error(std::exception_ptr ep) noexcept override
                {
                    hpx::execution::experimental::set_error(
                        HPX_MOVE(r), HPX_MOVE(ep));
                }

                void set_stopped() noexcept override
                {
                    hpx::execution::experimental::set_stopped(r);
                }
            };

            class default_parallel_scheduler : public parallel_scheduler
            {
            public:
                void schedule(receiver* r, storage s) noexcept override
                {
                    (void) s;
                    if (!r)
                    {
                        return;
                    }
                    auto stop_token_opt = r->try_query_stop_token();
                    hpx::async([r, stop_token_opt]() {
                        if (stop_token_opt && stop_token_opt->stop_requested())
                        {
                            r->set_stopped();
                        }
                        else
                        {
                            r->set_value();
                        }
                    });
                }

                void schedule_bulk_chunked(uint32_t n, bulk_item_receiver* r,
                    storage s) noexcept override
                {
                    (void) s;
                    if (!r)
                    {
                        return;
                    }
                    if (n == 0)
                    {
                        r->set_value();
                        return;
                    }
                    if (auto stop_token_opt = r->try_query_stop_token())
                    {
                        if (stop_token_opt->stop_requested())
                        {
                            r->set_stopped();
                            return;
                        }
                    }
                    std::size_t num_threads =
                        std::thread::hardware_concurrency();
                    std::size_t chunk_size =
                        (n + num_threads - 1) / num_threads;
                    std::vector<hpx::future<void>> futures;

                    for (std::size_t t = 0; t < num_threads; ++t)
                    {
                        uint32_t start = static_cast<uint32_t>(t * chunk_size);
                        uint32_t end = static_cast<uint32_t>(std::min(
                            static_cast<std::size_t>(start) + chunk_size,
                            static_cast<std::size_t>(n)));
                        if (start >= n)
                            break;

                        futures.push_back(hpx::async(
                            [r, start, end]() { r->start(start, end); }));
                    }

                    hpx::when_all(futures).then(
                        [r](hpx::future<std::vector<hpx::future<void>>>&&
                                f) noexcept {
                            try
                            {
                                f.get();
                                auto error_ep = r->get_error();
                                if (error_ep != nullptr)
                                {
                                    r->set_error(error_ep);
                                }
                                else
                                {
                                    r->set_value();
                                }
                            }
                            catch (...)
                            {
                                r->set_error(std::current_exception());
                            }
                        });
                }

                void schedule_bulk_unchunked(uint32_t n, bulk_item_receiver* r,
                    storage s) noexcept override
                {
                    (void) s;
                    if (!r)
                    {
                        return;
                    }
                    if (n == 0)
                    {
                        r->set_value();
                        return;
                    }
                    if (auto stop_token_opt = r->try_query_stop_token())
                    {
                        if (stop_token_opt->stop_requested())
                        {
                            r->set_stopped();
                            return;
                        }
                    }
                    std::vector<hpx::future<void>> futures;

                    for (uint32_t i = 0; i < n; ++i)
                    {
                        futures.push_back(
                            hpx::async([r, i]() { r->start(i, i + 1); }));
                    }

                    hpx::when_all(futures).then(
                        [r](hpx::future<std::vector<hpx::future<void>>>&&
                                f) noexcept {
                            try
                            {
                                f.get();
                                auto error_ep = r->get_error();
                                if (error_ep != nullptr)
                                {
                                    r->set_error(error_ep);
                                }
                                else
                                {
                                    r->set_value();
                                }
                            }
                            catch (...)
                            {
                                r->set_error(std::current_exception());
                            }
                        });
                }
            };

            std::shared_ptr<parallel_scheduler>
            query_parallel_scheduler_backend();

        }    // namespace replaceability

        template <typename Scheduler, typename Receiver>
        struct parallel_operation_state
        {
            Scheduler scheduler;
            Receiver& receiver;
            std::shared_ptr<replaceability::my_receiver<Receiver>> my_r;

            template <typename S, typename R>
            parallel_operation_state(S&& s, R& r)
              : scheduler(HPX_FORWARD(S, s))
              , receiver(r)
              , my_r(std::make_shared<replaceability::my_receiver<Receiver>>(r))
            {
            }

            friend void tag_invoke(hpx::execution::experimental::start_t,
                parallel_operation_state& os) noexcept
            {
                hpx::detail::try_catch_exception_ptr(
                    [&]() {
                        auto backend =
                            replaceability::query_parallel_scheduler_backend();
                        auto env =
                            hpx::execution::experimental::get_env(os.receiver);
                        auto stop_token =
                            hpx::execution::experimental::get_stop_token(env);
                        if (stop_token.stop_requested())
                        {
                            hpx::execution::experimental::set_stopped(
                                os.receiver);
                            return;
                        }
                        replaceability::storage s{nullptr, 0};
                        backend->schedule(os.my_r.get(), s);
                    },
                    [&](std::exception_ptr ep) {
                        hpx::execution::experimental::set_error(
                            HPX_MOVE(os.receiver), HPX_MOVE(ep));
                    });
            }
        };

        template <typename S>
        std::enable_if_t<std::is_integral_v<S>, std::size_t> get_shape_size(
            S const& sh)
        {
            return static_cast<std::size_t>(sh);
        }

        template <typename S>
        std::enable_if_t<!std::is_integral_v<S>, std::size_t> get_shape_size(
            S const& sh)
        {
            return std::distance(hpx::util::begin(sh), hpx::util::end(sh));
        }

        template <typename Sender, typename Shape, typename F>
        struct parallel_bulk_sender
        {
            parallel_scheduler scheduler;
            Shape shape;
            F f;

            parallel_bulk_sender(parallel_scheduler&& sched, Shape sh, F&& func)
              : scheduler(HPX_MOVE(sched))
              , shape(sh)
              , f(HPX_FORWARD(F, func))
            {
            }

#if defined(HPX_HAVE_STDEXEC)
            using sender_concept = hpx::execution::experimental::sender_t;
#endif
            using completion_signatures =
                hpx::execution::experimental::completion_signatures<
                    hpx::execution::experimental::set_value_t(),
                    hpx::execution::experimental::set_error_t(
                        std::exception_ptr),
                    hpx::execution::experimental::set_stopped_t()>;

            template <typename Env>
            friend constexpr auto tag_invoke(
                hpx::execution::experimental::get_completion_signatures_t,
                parallel_bulk_sender const&, Env) noexcept
                -> completion_signatures
            {
                return {};
            }
        };

        template <typename Receiver, typename Shape, typename F>
        struct parallel_bulk_operation_state
        {
            parallel_scheduler scheduler;
            Receiver& receiver;
            Shape shape;
            F f;
            std::size_t size;
            std::shared_ptr<
                replaceability::my_bulk_item_receiver<Receiver, Shape, F>>
                my_r;

            template <typename R, typename S>
            parallel_bulk_operation_state(
                parallel_scheduler&& sched, R& r, S sh, F&& func)
              : scheduler(HPX_MOVE(sched))
              , receiver(r)
              , shape(sh)
              , f(HPX_FORWARD(F, func))
              , my_r(std::make_shared<
                    replaceability::my_bulk_item_receiver<Receiver, Shape, F>>(
                    r, std::move(f), sh))
            {
                size = get_shape_size(sh);
            }

            friend void tag_invoke(hpx::execution::experimental::start_t,
                parallel_bulk_operation_state& os) noexcept
            {
                hpx::detail::try_catch_exception_ptr(
                    [&]() {
                        auto backend =
                            replaceability::query_parallel_scheduler_backend();
                        auto env =
                            hpx::execution::experimental::get_env(os.receiver);
                        auto stop_token =
                            hpx::execution::experimental::get_stop_token(env);
                        if (stop_token.stop_requested())
                        {
                            hpx::execution::experimental::set_stopped(
                                os.receiver);
                            return;
                        }
                        replaceability::storage s{nullptr, 0};
                        backend->schedule_bulk_chunked(
                            static_cast<uint32_t>(os.size), os.my_r.get(), s);
                    },
                    [&](std::exception_ptr ep) {
                        hpx::execution::experimental::set_error(
                            HPX_MOVE(os.receiver), HPX_MOVE(ep));
                    });
            }
        };

    }    // namespace detail

    struct parallel_sender
    {
        parallel_scheduler scheduler;

#if defined(HPX_HAVE_STDEXEC)
        using sender_concept = hpx::execution::experimental::sender_t;
#endif
        using completion_signatures =
            hpx::execution::experimental::completion_signatures<
                hpx::execution::experimental::set_value_t(),
                hpx::execution::experimental::set_error_t(std::exception_ptr),
                hpx::execution::experimental::set_stopped_t()>;

        template <typename Env>
        friend constexpr auto tag_invoke(
            hpx::execution::experimental::get_completion_signatures_t,
            parallel_sender const&, Env) noexcept -> completion_signatures
        {
            return {};
        }

        template <typename Receiver>
        friend auto tag_invoke(hpx::execution::experimental::connect_t,
            parallel_sender&& s, Receiver& r)
        {
            return detail::parallel_operation_state<parallel_scheduler,
                Receiver>{HPX_MOVE(s.scheduler), r};
        }

        template <typename Receiver>
        friend auto tag_invoke(hpx::execution::experimental::connect_t,
            parallel_sender& s, Receiver& r)
        {
            return detail::parallel_operation_state<parallel_scheduler,
                Receiver>{s.scheduler, r};
        }

        template <typename Shape, typename F>
        friend auto tag_invoke(hpx::execution::experimental::bulk_t,
            parallel_sender&& s, Shape shape, F&& f)
        {
            return detail::parallel_bulk_sender<parallel_sender, Shape, F>{
                HPX_MOVE(s.scheduler), shape, HPX_FORWARD(F, f)};
        }

        template <typename Shape, typename F, typename Receiver>
        friend auto tag_invoke(hpx::execution::experimental::connect_t,
            detail::parallel_bulk_sender<parallel_sender, Shape, F>&& s,
            Receiver& r)
        {
            return detail::parallel_bulk_operation_state<Receiver, Shape, F>{
                HPX_MOVE(s.scheduler), r, s.shape, HPX_FORWARD(F, s.f)};
        }

#if defined(HPX_HAVE_STDEXEC)
        struct env
        {
            parallel_scheduler const& sched;

            friend auto tag_invoke(
                hpx::execution::experimental::get_completion_scheduler_t<
                    set_value_t>,
                env const& e) noexcept -> parallel_scheduler
            {
                return e.sched;
            }

            friend auto tag_invoke(
                hpx::execution::experimental::get_completion_scheduler_t<
                    set_stopped_t>,
                env const& e) noexcept -> parallel_scheduler
            {
                return e.sched;
            }
        };

        friend env tag_invoke(hpx::execution::experimental::get_env_t,
            parallel_sender const& s) noexcept
        {
            return {s.scheduler};
        }
#endif
    };

    inline parallel_sender tag_invoke(hpx::execution::experimental::schedule_t,
        parallel_scheduler&& sched) noexcept
    {
        return {HPX_MOVE(sched)};
    }

    inline parallel_sender tag_invoke(hpx::execution::experimental::schedule_t,
        parallel_scheduler const& sched) noexcept
    {
        return {sched};
    }

    inline parallel_scheduler get_parallel_scheduler() noexcept
    {
        return parallel_scheduler{};
    }

}    // namespace hpx::execution::experimental
