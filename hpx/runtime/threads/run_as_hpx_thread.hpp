//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_THREADS_RUN_AS_HPX_THREAD_MAR_12_2016_0202PM)
#define HPX_THREADS_RUN_AS_HPX_THREAD_MAR_12_2016_0202PM

#include <hpx/config.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/util/invoke_fused.hpp>
#include <hpx/util/tuple.hpp>

#include <cstdlib>
#include <type_traits>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>

#if defined(HPX_HAVE_CXX1Y_EXPERIMENTAL_OPTIONAL)
#include <experimental/optional>
#else
#include <boost/optional.hpp>
#endif

namespace hpx { namespace threads
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        // This is the overload for running functions which return a value.
        template <typename F, typename... Ts>
        typename std::result_of<F(Ts &&...)>::type
        run_as_hpx_thread(std::false_type, F const& f, Ts &&... ts)
        {
            std::mutex mtx;
            std::condition_variable cond;
            bool stopping = false;

            typedef typename std::result_of<F(Ts &&...)>::type result_type;

            // Using the optional for storing the returned result value
            // allows to support non-default-constructible and move-only
            // types.
#if defined(HPX_HAVE_CXX1Y_EXPERIMENTAL_OPTIONAL)
            std::experimental::optional<result_type> result;
#else
            boost::optional<result_type> result;
#endif

            // This lambda function will be scheduled to run as an HPX
            // thread
            auto && args = util::forward_as_tuple(std::forward<Ts>(ts)...);
            auto && wrapper =
                [&]() mutable
                {
                    // Execute the given function, forward all parameters,
                    // store result.

#if defined(HPX_HAVE_CXX1Y_EXPERIMENTAL_OPTIONAL)
                    result.emplace(util::invoke_fused(f, std::move(args)));
#elif BOOST_VERSION < 105600
                    result = boost::in_place(
                                util::invoke_fused(f, std::move(args)));
#else
                    result.emplace(util::invoke_fused(f, std::move(args)));
#endif

                    // Now signal to the waiting thread that we're done.
                    {
                        std::lock_guard<std::mutex> lk(mtx);
                        stopping = true;
                    }
                    cond.notify_all();
                };

            // Create the HPX thread
            hpx::threads::register_thread_nullary(std::ref(wrapper));

            // wait for the HPX thread to exit
            std::unique_lock<std::mutex> lk(mtx);
            while (!stopping)
                cond.wait(lk);

            return std::move(*result);
        }

        // This is the overload for running functions which return void.
        template <typename F, typename... Ts>
        void run_as_hpx_thread(std::true_type, F const& f, Ts &&... ts)
        {
            std::mutex mtx;
            std::condition_variable cond;
            bool stopping = false;

            // This lambda function will be scheduled to run as an HPX
            // thread
            auto && args = util::forward_as_tuple(std::forward<Ts>(ts)...);
            auto && wrapper =
                [&]() mutable
                {
                    // Execute the given function, forward all parameters.
                    util::invoke_fused(f, std::move(args));

                    // Now signal to the waiting thread that we're done.
                    {
                        std::lock_guard<std::mutex> lk(mtx);
                        stopping = true;
                    }
                    cond.notify_all();
                };

            // Create an HPX thread
            hpx::threads::register_thread_nullary(std::ref(wrapper));

            // wait for the HPX thread to exit
            std::unique_lock<std::mutex> lk(mtx);
            while (!stopping)
                cond.wait(lk);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename... Ts>
    typename std::result_of<F(Ts &&...)>::type
    run_as_hpx_thread(F const& f, Ts &&... vs)
    {
        typedef typename std::is_void<
                typename std::result_of<F(Ts &&...)>::type
            >::type result_is_void;

        return detail::run_as_hpx_thread(result_is_void(),
            f, std::forward<Ts>(vs)...);
    }
}}

#endif
