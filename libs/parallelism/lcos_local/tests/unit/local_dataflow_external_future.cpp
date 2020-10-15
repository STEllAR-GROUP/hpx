//  Copyright (c) 2020 ETH Zurich
//  Copyright (c) 2015 Hartmut Kaiser
//  Copyright (c) 2013 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/chrono.hpp>
#include <hpx/execution.hpp>
#include <hpx/functional.hpp>
#include <hpx/future.hpp>
#include <hpx/modules/memory.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/program_options.hpp>
#include <hpx/tuple.hpp>
#include <hpx/wrap_main.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

std::atomic<bool> done{false};

///////////////////////////////////////////////////////////////////////////////
struct external_future_executor
{
    // This is not actually called by dataflow, but it is used for the return
    // type calculation of it. dataflow_finalize has to set the same type to
    // the future state.
    template <typename F, typename... Ts>
    decltype(auto) async_execute_helper(std::true_type, F&& f, Ts&&... ts)
    {
        // The completion of f is signalled out-of-band.
        hpx::invoke(std::forward<F>(f), std::forward<Ts>(ts)...);
        return hpx::async(
            []() { hpx::util::yield_while([]() { return !done; }); });
    }

    template <typename F, typename... Ts>
    decltype(auto) async_execute_helper(std::false_type, F&& f, Ts&&... ts)
    {
        // The completion of f is signalled out-of-band.
        auto&& r = hpx::invoke(std::forward<F>(f), std::forward<Ts>(ts)...);
        return hpx::async([r = std::move(r)]() {
            hpx::util::yield_while([]() { return !done; });
            return r;
        });
    }

    template <typename F, typename... Ts>
    decltype(auto) async_execute(F&& f, Ts&&... ts)
    {
        using is_void = typename std::is_void<
            typename hpx::util::invoke_result<F, Ts...>::type>;
        return async_execute_helper(
            is_void{}, std::forward<F>(f), std::forward<Ts>(ts)...);
    }

    template <typename Frame, typename F, typename Futures>
    void dataflow_finalize_helper(
        std::true_type, Frame frame, F&& f, Futures&& futures)
    {
        std::exception_ptr p;

        try
        {
            hpx::util::invoke_fused(
                std::forward<F>(f), std::forward<Futures>(futures));

            // Signal completion from another thread/task.
            hpx::intrusive_ptr<typename std::remove_pointer<
                typename std::decay<Frame>::type>::type>
                frame_p(frame);
            hpx::apply([frame_p = std::move(frame_p)]() {
                hpx::util::yield_while([]() { return !done; });
                frame_p->set_data(hpx::util::unused_type{});
            });
            return;
        }
        catch (...)
        {
            p = std::current_exception();
        }

        // The exception is set outside the catch block since
        // set_exception may yield. Ending the catch block on a
        // different worker thread than where it was started may lead
        // to segfaults.
        frame->set_exception(std::move(p));
    }

    template <typename Frame, typename F, typename Futures>
    void dataflow_finalize_helper(
        std::false_type, Frame frame, F&& f, Futures&& futures)
    {
        std::exception_ptr p;

        try
        {
            auto&& r = hpx::util::invoke_fused(
                std::forward<F>(f), std::forward<Futures>(futures));

            // Signal completion from another thread/task.
            hpx::intrusive_ptr<typename std::remove_pointer<
                typename std::decay<Frame>::type>::type>
                frame_p(frame);
            hpx::apply([frame_p = std::move(frame_p), r = std::move(r)]() {
                hpx::util::yield_while([]() { return !done; });
                frame_p->set_data(std::move(r));
            });
            return;
        }
        catch (...)
        {
            p = std::current_exception();
        }

        // The exception is set outside the catch block since
        // set_exception may yield. Ending the catch block on a
        // different worker thread than where it was started may lead
        // to segfaults.
        frame->set_exception(std::move(p));
    }

    template <typename Frame, typename F, typename Futures>
    void dataflow_finalize(Frame&& frame, F&& f, Futures&& futures)
    {
        using is_void = typename std::remove_pointer<
            typename std::decay<Frame>::type>::type::is_void;
        dataflow_finalize_helper(is_void{}, std::forward<Frame>(frame),
            std::forward<F>(f), std::forward<Futures>(futures));
    }
};

struct additional_argument
{
};

struct external_future_additional_argument_executor
{
    // This is not actually called by dataflow, but it is used for the return
    // type calculation of it. dataflow_finalize has to set the same type to
    // the future state.
    template <typename F, typename... Ts>
    decltype(auto) async_execute_helper(std::true_type, F&& f, Ts&&... ts)
    {
        // The completion of f is signalled out-of-band.
        hpx::invoke(
            std::forward<F>(f), additional_argument{}, std::forward<Ts>(ts)...);
        return hpx::async(
            []() { hpx::util::yield_while([]() { return !done; }); });
    }

    template <typename F, typename... Ts>
    decltype(auto) async_execute_helper(std::false_type, F&& f, Ts&&... ts)
    {
        // The completion of f is signalled out-of-band.
        auto&& r = hpx::invoke(
            std::forward<F>(f), additional_argument{}, std::forward<Ts>(ts)...);
        return hpx::async([r = std::move(r)]() {
            hpx::util::yield_while([]() { return !done; });
            return r;
        });
    }

    template <typename F, typename... Ts>
    decltype(auto) async_execute(F&& f, Ts&&... ts)
    {
        using is_void =
            typename std::is_void<typename hpx::util::invoke_result<F,
                additional_argument, Ts...>::type>;
        return async_execute_helper(
            is_void{}, std::forward<F>(f), std::forward<Ts>(ts)...);
    }

    template <typename Frame, typename F, typename Futures>
    void dataflow_finalize_helper(
        std::true_type, Frame frame, F&& f, Futures&& futures)
    {
        std::exception_ptr p;

        try
        {
            additional_argument a{};
            hpx::util::invoke_fused(std::forward<F>(f),
                hpx::tuple_cat(hpx::tie(a), std::forward<Futures>(futures)));

            // Signal completion from another thread/task.
            hpx::intrusive_ptr<typename std::remove_pointer<
                typename std::decay<Frame>::type>::type>
                frame_p(frame);
            hpx::apply([frame_p = std::move(frame_p)]() {
                hpx::util::yield_while([]() { return !done; });
                frame_p->set_data(hpx::util::unused_type{});
            });
            return;
        }
        catch (...)
        {
            p = std::current_exception();
        }

        // The exception is set outside the catch block since
        // set_exception may yield. Ending the catch block on a
        // different worker thread than where it was started may lead
        // to segfaults.
        frame->set_exception(std::move(p));
    }

    template <typename Frame, typename F, typename Futures>
    void dataflow_finalize_helper(
        std::false_type, Frame frame, F&& f, Futures&& futures)
    {
        std::exception_ptr p;

        try
        {
            additional_argument a{};
            auto&& r = hpx::util::invoke_fused(std::forward<F>(f),
                hpx::tuple_cat(hpx::tie(a), std::forward<Futures>(futures)));

            // Signal completion from another thread/task.
            hpx::intrusive_ptr<typename std::remove_pointer<
                typename std::decay<Frame>::type>::type>
                frame_p(frame);
            hpx::apply([frame_p = std::move(frame_p), r = std::move(r)]() {
                hpx::util::yield_while([]() { return !done; });
                frame_p->set_data(std::move(r));
            });
            return;
        }
        catch (...)
        {
            p = std::current_exception();
        }

        // The exception is set outside the catch block since
        // set_exception may yield. Ending the catch block on a
        // different worker thread than where it was started may lead
        // to segfaults.
        frame->set_exception(std::move(p));
    }

    template <typename Frame, typename F, typename Futures>
    void dataflow_finalize(Frame&& frame, F&& f, Futures&& futures)
    {
        using is_void = typename std::remove_pointer<
            typename std::decay<Frame>::type>::type::is_void;
        dataflow_finalize_helper(is_void{}, std::forward<Frame>(frame),
            std::forward<F>(f), std::forward<Futures>(futures));
    }
};

namespace hpx { namespace parallel { namespace execution {
    template <>
    struct is_two_way_executor<external_future_executor> : std::true_type
    {
    };

    template <>
    struct is_two_way_executor<external_future_additional_argument_executor>
      : std::true_type
    {
    };
}}}    // namespace hpx::parallel::execution

int main()
{
    // We time the spawn and the wait. The wait should take significantly
    // longer than the spawn, and the wait should be long.
    {
        external_future_executor exec;
        hpx::chrono::high_resolution_timer t;
        hpx::future<void> f = hpx::dataflow(exec, []() {
            // This represents an asynchronous operation which has an
            // out-of-band mechanism for signaling completion.
            hpx::apply([]() {
                hpx::this_thread::sleep_for(std::chrono::milliseconds(500));
                done = true;
            });
        });
        double spawn_time = t.elapsed();
        t.restart();
        f.get();
        double wait_time = t.elapsed();
        HPX_TEST_LT(spawn_time, wait_time);
        HPX_TEST_LT(0.3, wait_time);
    }

    {
        done = false;

        external_future_executor exec;
        hpx::chrono::high_resolution_timer t;
        hpx::future<int> f = hpx::dataflow(exec, []() {
            // This represents an asynchronous operation which has an
            // out-of-band mechanism for signaling completion.
            hpx::apply([]() {
                hpx::this_thread::sleep_for(std::chrono::milliseconds(500));
                done = true;
            });

            return 42;
        });
        double spawn_time = t.elapsed();
        t.restart();
        int r = f.get();
        HPX_TEST_EQ(r, 42);
        double wait_time = t.elapsed();
        HPX_TEST_LT(spawn_time, wait_time);
        HPX_TEST_LT(0.3, wait_time);
    }

    {
        done = false;

        external_future_additional_argument_executor exec;
        hpx::chrono::high_resolution_timer t;
        hpx::future<void> f = hpx::dataflow(exec, [](additional_argument) {
            // This represents an asynchronous operation which has an
            // out-of-band mechanism for signaling completion.
            hpx::apply([]() {
                hpx::this_thread::sleep_for(std::chrono::milliseconds(500));
                done = true;
            });
        });
        double spawn_time = t.elapsed();
        t.restart();
        f.get();
        double wait_time = t.elapsed();
        HPX_TEST_LT(spawn_time, wait_time);
        HPX_TEST_LT(0.3, wait_time);
    }

    {
        done = false;

        external_future_additional_argument_executor exec;
        hpx::chrono::high_resolution_timer t;
        hpx::future<int> f = hpx::dataflow(exec, [](additional_argument) {
            // This represents an asynchronous operation which has an
            // out-of-band mechanism for signaling completion.
            hpx::apply([]() {
                hpx::this_thread::sleep_for(std::chrono::milliseconds(500));
                done = true;
            });

            return 42;
        });
        double spawn_time = t.elapsed();
        t.restart();
        int r = f.get();
        HPX_TEST_EQ(r, 42);
        double wait_time = t.elapsed();
        HPX_TEST_LT(spawn_time, wait_time);
        HPX_TEST_LT(0.3, wait_time);
    }

    return 0;
}
