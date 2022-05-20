//  Copyright (c) 2007-2022 Hartmut Kaiser
//  Copyright (c)      2017 Taeguk Kwon
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/modules/async_local.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/parallel/util/detail/handle_exception_termination_handler.hpp>

#include <array>
#include <cstddef>
#include <exception>
#include <list>
#include <type_traits>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace util { namespace detail {

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy>
    struct handle_local_exceptions
    {
        ///////////////////////////////////////////////////////////////////////
        // std::bad_alloc has to be handled separately
#if defined(HPX_COMPUTE_DEVICE_CODE)
        [[noreturn]] static void call(std::exception_ptr const&)
        {
            std::terminate();
        }
#else
        [[noreturn]] static void call(std::exception_ptr const& e)
        {
            try
            {
                std::rethrow_exception(e);
            }
            catch (std::bad_alloc const& ba)
            {
                throw ba;
            }
            catch (...)
            {
                throw exception_list(e);
            }
        }
#endif

        static void call(
            std::exception_ptr const& e, std::list<std::exception_ptr>& errors)
        {
#if defined(HPX_COMPUTE_DEVICE_CODE)
            HPX_UNUSED(e);
            HPX_UNUSED(errors);
            HPX_ASSERT(false);
#else
            try
            {
                std::rethrow_exception(e);
            }
            catch (std::bad_alloc const& ba)
            {
                throw ba;
            }
            catch (...)
            {
                errors.push_back(e);
            }
#endif
        }

        static void call(std::list<std::exception_ptr>& errors)
        {
            if (!errors.empty())
            {
                throw exception_list(HPX_MOVE(errors));
            }
        }

    private:
        template <typename Future>
        static void call_helper_single(Future const& f,
            std::list<std::exception_ptr>& errors, bool throw_errors)
        {
#if defined(HPX_COMPUTE_DEVICE_CODE)
            HPX_UNUSED(f);
            HPX_UNUSED(errors);
            HPX_UNUSED(throw_errors);
            HPX_ASSERT(false);
#else
            // extract exception from future and handle as needed
            if (f.has_exception())
            {
                call(f.get_exception_ptr(), errors);
            }

            if (throw_errors && !errors.empty())
            {
                throw exception_list(HPX_MOVE(errors));
            }
#endif
        }

        template <typename Future>
        static void call_helper_single(Future const& f, bool throw_errors)
        {
#if defined(HPX_COMPUTE_DEVICE_CODE)
            HPX_UNUSED(f);
            HPX_UNUSED(throw_errors);
            HPX_ASSERT(false);
#else
            // extract exception from future and handle as needed
            if (f.has_exception())
            {
                std::list<std::exception_ptr> errors;
                call(f.get_exception_ptr(), errors);
                if (throw_errors && !errors.empty())
                {
                    throw exception_list(HPX_MOVE(errors));
                }
            }
#endif
        }

    public:
        template <typename T>
        static void call(hpx::future<T> const& f,
            std::list<std::exception_ptr>& errors, bool throw_errors = true)
        {
            return call_helper_single(f, errors, throw_errors);
        }

        template <typename T>
        static void call(hpx::shared_future<T> const& f,
            std::list<std::exception_ptr>& errors, bool throw_errors = true)
        {
            return call_helper_single(f, errors, throw_errors);
        }

        template <typename T>
        static void call(hpx::future<T> const& f, bool throw_errors = true)
        {
            return call_helper_single(f, throw_errors);
        }

        template <typename T>
        static void call(
            hpx::shared_future<T> const& f, bool throw_errors = true)
        {
            return call_helper_single(f, throw_errors);
        }

    private:
        template <typename Cont>
        static void call_helper(Cont const& workitems,
            std::list<std::exception_ptr>& errors, bool throw_errors)
        {
#if defined(HPX_COMPUTE_DEVICE_CODE)
            HPX_UNUSED(workitems);
            HPX_UNUSED(errors);
            HPX_UNUSED(throw_errors);
            HPX_ASSERT(false);
#else
            if (workitems.empty() && errors.empty())
            {
                return;
            }

            // first extract exception from all futures
            for (auto const& f : workitems)
            {
                if (f.has_exception())
                {
                    // rethrow std::bad_alloc or store exception
                    call(f.get_exception_ptr(), errors);
                }
            }

            if (throw_errors && !errors.empty())
            {
                throw exception_list(HPX_MOVE(errors));
            }
#endif
        }

        template <typename Cont>
        static void call_helper(Cont const& workitems, bool throw_errors)
        {
#if defined(HPX_COMPUTE_DEVICE_CODE)
            HPX_UNUSED(workitems);
            HPX_UNUSED(throw_errors);
            HPX_ASSERT(false);
#else
            if (workitems.empty())
            {
                return;
            }

            // first detect whether there are exceptional futures
            bool found_exception = false;
            for (auto const& f : workitems)
            {
                if (f.has_exception())
                {
                    found_exception = true;
                    break;
                }
            }

            // nothing needs to be done
            if (!found_exception)
            {
                return;
            }

            // now extract exception from all futures
            std::list<std::exception_ptr> errors;
            for (auto const& f : workitems)
            {
                if (f.has_exception())
                {
                    // rethrow std::bad_alloc or store exception
                    call(f.get_exception_ptr(), errors);
                }
            }

            if (throw_errors && !errors.empty())
            {
                throw exception_list(HPX_MOVE(errors));
            }
#endif
        }

    public:
        template <typename T>
        static void call(std::vector<hpx::future<T>> const& workitems,
            std::list<std::exception_ptr>& errors, bool throw_errors = true)
        {
            return call_helper(workitems, errors, throw_errors);
        }

        template <typename T, std::size_t N>
        static void call(std::array<hpx::future<T>, N> const& workitems,
            std::list<std::exception_ptr>& errors, bool throw_errors = true)
        {
            return call_helper(workitems, errors, throw_errors);
        }

        template <typename T>
        static void call(std::vector<hpx::shared_future<T>> const& workitems,
            std::list<std::exception_ptr>& errors, bool throw_errors = true)
        {
            return call_helper(workitems, errors, throw_errors);
        }

        template <typename T, std::size_t N>
        static void call(std::array<hpx::shared_future<T>, N> const& workitems,
            std::list<std::exception_ptr>& errors, bool throw_errors = true)
        {
            return call_helper(workitems, errors, throw_errors);
        }

        template <typename T>
        static void call(std::vector<hpx::future<T>> const& workitems,
            bool throw_errors = true)
        {
            return call_helper(workitems, throw_errors);
        }

        template <typename T, std::size_t N>
        static void call(std::array<hpx::future<T>, N> const& workitems,
            bool throw_errors = true)
        {
            return call_helper(workitems, throw_errors);
        }

        template <typename T>
        static void call(std::vector<hpx::shared_future<T>> const& workitems,
            bool throw_errors = true)
        {
            return call_helper(workitems, throw_errors);
        }

        template <typename T, std::size_t N>
        static void call(std::array<hpx::shared_future<T>, N> const& workitems,
            bool throw_errors = true)
        {
            return call_helper(workitems, throw_errors);
        }

    private:
        template <typename Future>
        static void call_with_cleanup_helper_single(Future const& f,
            std::list<std::exception_ptr>& errors, bool throw_errors)
        {
#if defined(HPX_COMPUTE_DEVICE_CODE)
            HPX_UNUSED(f);
            HPX_UNUSED(errors);
            HPX_UNUSED(throw_errors);
            HPX_ASSERT(false);
#else
            if (f.has_exception())
            {
                call(f.get_exception_ptr(), errors);
            }

            if (throw_errors && !errors.empty())
            {
                throw exception_list(HPX_MOVE(errors));
            }
#endif
        }

        template <typename Future>
        static void call_with_cleanup_helper_single(
            Future const& f, bool throw_errors)
        {
#if defined(HPX_COMPUTE_DEVICE_CODE)
            HPX_UNUSED(f);
            HPX_UNUSED(throw_errors);
            HPX_ASSERT(false);
#else
            if (f.has_exception())
            {
                std::list<std::exception_ptr> errors;
                call(f.get_exception_ptr(), errors);
                if (throw_errors && !errors.empty())
                {
                    throw exception_list(HPX_MOVE(errors));
                }
            }
#endif
        }

    public:
        template <typename T, typename Cleanup>
        static void call_with_cleanup(hpx::future<T> const& f,
            std::list<std::exception_ptr>& errors, Cleanup&&,
            bool throw_errors = true)
        {
            return call_with_cleanup_helper_single(f, errors, throw_errors);
        }

        template <typename T, typename Cleanup,
            typename Enable = std::enable_if_t<!std::is_same_v<
                std::decay_t<Cleanup>, std::list<std::exception_ptr>>>>
        static void call_with_cleanup(
            hpx::future<T> const& f, Cleanup&&, bool throw_errors = true)
        {
            return call_with_cleanup_helper_single(f, throw_errors);
        }

    private:
        template <typename Cont, typename Cleanup>
        static void call_with_cleanup_helper(Cont& workitems,
            std::list<std::exception_ptr>& errors, Cleanup&& cleanup,
            bool throw_errors)
        {
#if defined(HPX_COMPUTE_DEVICE_CODE)
            HPX_UNUSED(workitems);
            HPX_UNUSED(errors);
            HPX_UNUSED(cleanup);
            HPX_UNUSED(throw_errors);
            HPX_ASSERT(false);
#else
            if (workitems.empty() && errors.empty())
            {
                return;
            }

            bool has_exception = false;
            std::exception_ptr bad_alloc_exception;
            for (auto& f : workitems)
            {
                if (f.has_exception())
                {
                    std::exception_ptr e = f.get_exception_ptr();
                    try
                    {
                        std::rethrow_exception(e);
                    }
                    catch (std::bad_alloc const&)
                    {
                        bad_alloc_exception = e;
                    }
                    catch (...)
                    {
                        errors.push_back(e);
                    }
                    has_exception = true;
                }
            }

            // If at least one partition failed with an exception, call
            // the cleanup function for all others (the failed partitioned
            // are assumed to have already run the cleanup).
            if (has_exception)
            {
                for (auto& f : workitems)
                {
                    if (!f.has_exception())
                    {
                        cleanup(f.get());
                    }
                }
            }

            if (bad_alloc_exception)
            {
                std::rethrow_exception(bad_alloc_exception);
            }

            if (throw_errors && !errors.empty())
            {
                throw exception_list(HPX_MOVE(errors));
            }
#endif
        }

        template <typename Cont, typename Cleanup>
        static void call_with_cleanup_helper(
            Cont& workitems, Cleanup&& cleanup, bool throw_errors)
        {
#if defined(HPX_COMPUTE_DEVICE_CODE)
            HPX_UNUSED(workitems);
            HPX_UNUSED(cleanup);
            HPX_UNUSED(throw_errors);
            HPX_ASSERT(false);
#else
            if (workitems.empty())
            {
                return;
            }

            // first detect whether there are exceptional futures
            bool found_exception = false;
            for (auto const& f : workitems)
            {
                if (f.has_exception())
                {
                    found_exception = true;
                    break;
                }
            }

            // nothing needs to be done
            if (!found_exception)
            {
                return;
            }

            // now, handle exceptions and cleanup partitions
            std::list<std::exception_ptr> errors;
            std::exception_ptr bad_alloc_exception;
            for (auto& f : workitems)
            {
                if (f.has_exception())
                {
                    std::exception_ptr e = f.get_exception_ptr();
                    bool store_exception = false;
                    try
                    {
                        std::rethrow_exception(e);
                    }
                    catch (std::bad_alloc const&)
                    {
                        bad_alloc_exception = e;
                    }
                    catch (...)
                    {
                        store_exception = true;
                    }

                    // store the exception outside of the catch(...) block to
                    // avoid problems caused by suspending threads
                    if (store_exception)
                    {
                        errors.push_back(e);
                    }
                }
            }

            // As at least one partition failed with an exception, call
            // the cleanup function for all others (the failed partitioned
            // are assumed to have already run the cleanup).
            for (auto& f : workitems)
            {
                if (!f.has_exception())
                {
                    cleanup(f.get());
                }
            }

            if (bad_alloc_exception)
            {
                std::rethrow_exception(bad_alloc_exception);
            }

            if (throw_errors && !errors.empty())
            {
                throw exception_list(HPX_MOVE(errors));
            }
#endif
        }

    public:
        template <typename T, typename Cleanup>
        static void call_with_cleanup(std::vector<hpx::future<T>>& workitems,
            std::list<std::exception_ptr>& errors, Cleanup&& cleanup,
            bool throw_errors = true)
        {
            return call_with_cleanup_helper(
                workitems, errors, HPX_FORWARD(Cleanup, cleanup), throw_errors);
        }

        template <typename T, std::size_t N, typename Cleanup>
        static void call_with_cleanup(std::array<hpx::future<T>, N>& workitems,
            std::list<std::exception_ptr>& errors, Cleanup&& cleanup,
            bool throw_errors = true)
        {
            return call_with_cleanup_helper(
                workitems, errors, HPX_FORWARD(Cleanup, cleanup), throw_errors);
        }

        template <typename T, typename Cleanup,
            typename Enable = std::enable_if_t<!std::is_same_v<
                std::decay_t<Cleanup>, std::list<std::exception_ptr>>>>
        static void call_with_cleanup(std::vector<hpx::future<T>>& workitems,
            Cleanup&& cleanup, bool throw_errors = true)
        {
            return call_with_cleanup_helper(
                workitems, HPX_FORWARD(Cleanup, cleanup), throw_errors);
        }

        template <typename T, std::size_t N, typename Cleanup,
            typename Enable = std::enable_if_t<!std::is_same_v<
                std::decay_t<Cleanup>, std::list<std::exception_ptr>>>>
        static void call_with_cleanup(std::array<hpx::future<T>, N>& workitems,
            Cleanup&& cleanup, bool throw_errors = true)
        {
            return call_with_cleanup_helper(
                workitems, HPX_FORWARD(Cleanup, cleanup), throw_errors);
        }
    };

    template <>
    struct handle_local_exceptions<hpx::execution::parallel_unsequenced_policy>
    {
        ///////////////////////////////////////////////////////////////////////
#if defined(HPX_COMPUTE_DEVICE_CODE)
        [[noreturn]] static void call(std::exception_ptr const&)
        {
            std::terminate();
        }
#else
        [[noreturn]] static void call(std::exception_ptr const&)
        {
            parallel_exception_termination_handler();
        }
#endif

#if defined(HPX_COMPUTE_DEVICE_CODE)
        [[noreturn]] static void call(
            std::exception_ptr const&, std::list<std::exception_ptr>&)
        {
            std::terminate();
        }
#else
        [[noreturn]] static void call(
            std::exception_ptr const&, std::list<std::exception_ptr>&)
        {
            parallel_exception_termination_handler();
        }
#endif

        static void call(std::list<std::exception_ptr>& errors)
        {
            if (!errors.empty())
            {
                parallel_exception_termination_handler();
            }
        }

    private:
        template <typename Future>
        static void call_helper_single(Future const& f)
        {
#if defined(HPX_COMPUTE_DEVICE_CODE)
            HPX_UNUSED(f);
            HPX_ASSERT(false);
#else
            if (f.has_exception())
            {
                parallel_exception_termination_handler();
            }
#endif
        }

    public:
        template <typename T>
        static void call(hpx::future<T> const& f,
            std::list<std::exception_ptr>&, bool = true)
        {
            return call_helper_single(f);
        }

        template <typename T>
        static void call(hpx::shared_future<T> const& f,
            std::list<std::exception_ptr>&, bool = true)
        {
            return call_helper_single(f);
        }

        template <typename T>
        static void call(hpx::future<T> const& f, bool = true)
        {
            return call_helper_single(f);
        }

        template <typename T>
        static void call(hpx::shared_future<T> const& f, bool = true)
        {
            return call_helper_single(f);
        }

    private:
        template <typename Cont>
        static void call_helper(Cont const& workitems)
        {
#if defined(HPX_COMPUTE_DEVICE_CODE)
            HPX_UNUSED(workitems);
            HPX_ASSERT(false);
#else
            for (auto const& f : workitems)
            {
                if (f.has_exception())
                {
                    parallel_exception_termination_handler();
                }
            }
#endif
        }

    public:
        template <typename T>
        static void call(std::vector<hpx::future<T>> const& workitems,
            std::list<std::exception_ptr>&, bool = true)
        {
            return call_helper(workitems);
        }

        template <typename T, std::size_t N>
        static void call(std::array<hpx::future<T>, N> const& workitems,
            std::list<std::exception_ptr>&, bool = true)
        {
            return call_helper(workitems);
        }

        template <typename T>
        static void call(std::vector<hpx::shared_future<T>> const& workitems,
            std::list<std::exception_ptr>&, bool = true)
        {
            return call_helper(workitems);
        }

        template <typename T, std::size_t N>
        static void call(std::array<hpx::shared_future<T>, N> const& workitems,
            std::list<std::exception_ptr>&, bool = true)
        {
            return call_helper(workitems);
        }

        template <typename T>
        static void call(
            std::vector<hpx::future<T>> const& workitems, bool = true)
        {
            return call_helper(workitems);
        }

        template <typename T, std::size_t N>
        static void call(
            std::array<hpx::future<T>, N> const& workitems, bool = true)
        {
            return call_helper(workitems);
        }

        template <typename T>
        static void call(
            std::vector<hpx::shared_future<T>> const& workitems, bool = true)
        {
            return call_helper(workitems);
        }

        template <typename T, std::size_t N>
        static void call(
            std::array<hpx::shared_future<T>, N> const& workitems, bool = true)
        {
            return call_helper(workitems);
        }

    private:
        template <typename Future>
        static void call_with_cleanup_helper_single(Future const& f)
        {
#if defined(HPX_COMPUTE_DEVICE_CODE)
            HPX_UNUSED(f);
            HPX_ASSERT(false);
#else
            if (f.has_exception())
            {
                parallel_exception_termination_handler();
            }
#endif
        }

    public:
        template <typename T, typename Cleanup>
        static void call_with_cleanup(hpx::future<T> const& f,
            std::list<std::exception_ptr>&, Cleanup&&, bool = true)
        {
            call_with_cleanup_helper_single(f);
        }

        template <typename T, typename Cleanup,
            typename Enable = std::enable_if_t<!std::is_same_v<
                std::decay_t<Cleanup>, std::list<std::exception_ptr>>>>
        static void call_with_cleanup(
            hpx::future<T> const& f, Cleanup&&, bool = true)
        {
            call_with_cleanup_helper_single(f);
        }

    private:
        template <typename Cont>
        static void call_with_cleanup_helper(Cont const& workitems)
        {
#if defined(HPX_COMPUTE_DEVICE_CODE)
            HPX_UNUSED(workitems);
            HPX_ASSERT(false);
#else
            for (auto const& f : workitems)
            {
                if (f.has_exception())
                {
                    parallel_exception_termination_handler();
                }
            }
#endif
        }

    public:
        template <typename T, typename Cleanup>
        static void call_with_cleanup(
            std::vector<hpx::future<T>> const& workitems,
            std::list<std::exception_ptr>&, Cleanup&&, bool = true)
        {
            call_with_cleanup_helper(workitems);
        }

        template <typename T, std::size_t N, typename Cleanup>
        static void call_with_cleanup(
            std::array<hpx::future<T>, N> const& workitems,
            std::list<std::exception_ptr>&, Cleanup&&, bool = true)
        {
            call_with_cleanup_helper(workitems);
        }

        template <typename T, typename Cleanup,
            typename Enable = std::enable_if_t<!std::is_same_v<
                std::decay_t<Cleanup>, std::list<std::exception_ptr>>>>
        static void call_with_cleanup(
            std::vector<hpx::future<T>> const& workitems, Cleanup&&,
            bool = true)
        {
            call_with_cleanup_helper(workitems);
        }

        template <typename T, std::size_t N, typename Cleanup,
            typename Enable = std::enable_if_t<!std::is_same_v<
                std::decay_t<Cleanup>, std::list<std::exception_ptr>>>>
        static void call_with_cleanup(
            std::array<hpx::future<T>, N> const& workitems, Cleanup&&,
            bool = true)
        {
            call_with_cleanup_helper(workitems);
        }
    };

    template <>
    struct handle_local_exceptions<
        hpx::execution::parallel_unsequenced_task_policy>
    {
        ///////////////////////////////////////////////////////////////////////
#if defined(HPX_COMPUTE_DEVICE_CODE)
        [[noreturn]] static void call(std::exception_ptr const&)
        {
            std::terminate();
        }
#else
        [[noreturn]] static void call(std::exception_ptr const&)
        {
            parallel_exception_termination_handler();
        }
#endif

#if defined(HPX_COMPUTE_DEVICE_CODE)
        [[noreturn]] static void call(
            std::exception_ptr const&, std::list<std::exception_ptr>&)
        {
            std::terminate();
        }
#else
        [[noreturn]] static void call(
            std::exception_ptr const&, std::list<std::exception_ptr>&)
        {
            parallel_exception_termination_handler();
        }
#endif

        static void call(std::list<std::exception_ptr>& errors)
        {
            if (!errors.empty())
            {
                parallel_exception_termination_handler();
            }
        }

    private:
        template <typename Future>
        static void call_helper_single(Future const& f)
        {
#if defined(HPX_COMPUTE_DEVICE_CODE)
            HPX_UNUSED(f);
            HPX_ASSERT(false);
#else
            if (f.has_exception())
            {
                parallel_exception_termination_handler();
            }
#endif
        }

    public:
        template <typename T>
        static void call(hpx::future<T> const& f,
            std::list<std::exception_ptr>&, bool = true)
        {
            return call_helper_single(f);
        }

        template <typename T>
        static void call(hpx::shared_future<T> const& f,
            std::list<std::exception_ptr>&, bool = true)
        {
            return call_helper_single(f);
        }

        template <typename T>
        static void call(hpx::future<T> const& f, bool = true)
        {
            return call_helper_single(f);
        }

        template <typename T>
        static void call(hpx::shared_future<T> const& f, bool = true)
        {
            return call_helper_single(f);
        }

    private:
        template <typename Cont>
        static void call_helper(Cont const& workitems)
        {
#if defined(HPX_COMPUTE_DEVICE_CODE)
            HPX_UNUSED(workitems);
            HPX_ASSERT(false);
#else
            for (auto const& f : workitems)
            {
                if (f.has_exception())
                {
                    parallel_exception_termination_handler();
                }
            }
#endif
        }

    public:
        template <typename T>
        static void call(std::vector<hpx::future<T>> const& workitems,
            std::list<std::exception_ptr>&, bool = true)
        {
            return call_helper(workitems);
        }

        template <typename T, std::size_t N>
        static void call(std::array<hpx::future<T>, N> const& workitems,
            std::list<std::exception_ptr>&, bool = true)
        {
            return call_helper(workitems);
        }

        template <typename T>
        static void call(std::vector<hpx::shared_future<T>> const& workitems,
            std::list<std::exception_ptr>&, bool = true)
        {
            return call_helper(workitems);
        }

        template <typename T, std::size_t N>
        static void call(std::array<hpx::shared_future<T>, N> const& workitems,
            std::list<std::exception_ptr>&, bool = true)
        {
            return call_helper(workitems);
        }

        template <typename T>
        static void call(
            std::vector<hpx::future<T>> const& workitems, bool = true)
        {
            return call_helper(workitems);
        }

        template <typename T, std::size_t N>
        static void call(
            std::array<hpx::future<T>, N> const& workitems, bool = true)
        {
            return call_helper(workitems);
        }

        template <typename T>
        static void call(
            std::vector<hpx::shared_future<T>> const& workitems, bool = true)
        {
            return call_helper(workitems);
        }

        template <typename T, std::size_t N>
        static void call(
            std::array<hpx::shared_future<T>, N> const& workitems, bool = true)
        {
            return call_helper(workitems);
        }

    private:
        template <typename Future>
        static void call_with_cleanup_helper_single(Future const& f)
        {
#if defined(HPX_COMPUTE_DEVICE_CODE)
            HPX_UNUSED(f);
            HPX_ASSERT(false);
#else
            if (f.has_exception())
            {
                parallel_exception_termination_handler();
            }
#endif
        }

    public:
        template <typename T, typename Cleanup>
        static void call_with_cleanup(hpx::future<T> const& f,
            std::list<std::exception_ptr>&, Cleanup&&, bool = true)
        {
            call_with_cleanup_helper_single(f);
        }

        template <typename T, typename Cleanup,
            typename Enable = std::enable_if_t<!std::is_same_v<
                std::decay_t<Cleanup>, std::list<std::exception_ptr>>>>
        static void call_with_cleanup(
            hpx::future<T> const& f, Cleanup&&, bool = true)
        {
            call_with_cleanup_helper_single(f);
        }

    private:
        template <typename Cont>
        static void call_with_cleanup_helper(Cont const& workitems)
        {
#if defined(HPX_COMPUTE_DEVICE_CODE)
            HPX_UNUSED(workitems);
            HPX_ASSERT(false);
#else
            for (auto const& f : workitems)
            {
                if (f.has_exception())
                {
                    parallel_exception_termination_handler();
                }
            }
#endif
        }

    public:
        template <typename T, typename Cleanup>
        static void call_with_cleanup(
            std::vector<hpx::future<T>> const& workitems,
            std::list<std::exception_ptr>&, Cleanup&&, bool = true)
        {
            call_with_cleanup_helper(workitems);
        }

        template <typename T, std::size_t N, typename Cleanup>
        static void call_with_cleanup(
            std::array<hpx::future<T>, N> const& workitems,
            std::list<std::exception_ptr>&, Cleanup&&, bool = true)
        {
            call_with_cleanup_helper(workitems);
        }

        template <typename T, typename Cleanup,
            typename Enable = std::enable_if_t<!std::is_same_v<
                std::decay_t<Cleanup>, std::list<std::exception_ptr>>>>
        static void call_with_cleanup(
            std::vector<hpx::future<T>> const& workitems, Cleanup&&,
            bool = true)
        {
            call_with_cleanup_helper(workitems);
        }

        template <typename T, std::size_t N, typename Cleanup,
            typename Enable = std::enable_if_t<!std::is_same_v<
                std::decay_t<Cleanup>, std::list<std::exception_ptr>>>>
        static void call_with_cleanup(
            std::array<hpx::future<T>, N> const& workitems, Cleanup&&,
            bool = true)
        {
            call_with_cleanup_helper(workitems);
        }
    };

    template <typename Executor, typename Parameters>
    struct handle_local_exceptions<
        hpx::execution::parallel_unsequenced_policy_shim<Executor, Parameters>>
    {
        ///////////////////////////////////////////////////////////////////////
#if defined(HPX_COMPUTE_DEVICE_CODE)
        [[noreturn]] static void call(std::exception_ptr const&)
        {
            std::terminate();
        }
#else
        [[noreturn]] static void call(std::exception_ptr const&)
        {
            parallel_exception_termination_handler();
        }
#endif

#if defined(HPX_COMPUTE_DEVICE_CODE)
        [[noreturn]] static void call(
            std::exception_ptr const&, std::list<std::exception_ptr>&)
        {
            std::terminate();
        }
#else
        [[noreturn]] static void call(
            std::exception_ptr const&, std::list<std::exception_ptr>&)
        {
            parallel_exception_termination_handler();
        }
#endif

        static void call(std::list<std::exception_ptr>& errors)
        {
            if (!errors.empty())
            {
                parallel_exception_termination_handler();
            }
        }

    private:
        template <typename Future>
        static void call_helper_single(Future const& f)
        {
#if defined(HPX_COMPUTE_DEVICE_CODE)
            HPX_UNUSED(f);
            HPX_ASSERT(false);
#else
            if (f.has_exception())
            {
                parallel_exception_termination_handler();
            }
#endif
        }

    public:
        template <typename T>
        static void call(hpx::future<T> const& f,
            std::list<std::exception_ptr>&, bool = true)
        {
            return call_helper_single(f);
        }

        template <typename T>
        static void call(hpx::shared_future<T> const& f,
            std::list<std::exception_ptr>&, bool = true)
        {
            return call_helper_single(f);
        }

        template <typename T>
        static void call(hpx::future<T> const& f, bool = true)
        {
            return call_helper_single(f);
        }

        template <typename T>
        static void call(hpx::shared_future<T> const& f, bool = true)
        {
            return call_helper_single(f);
        }

    private:
        template <typename Cont>
        static void call_helper(Cont const& workitems)
        {
#if defined(HPX_COMPUTE_DEVICE_CODE)
            HPX_UNUSED(workitems);
            HPX_ASSERT(false);
#else
            for (auto const& f : workitems)
            {
                if (f.has_exception())
                {
                    parallel_exception_termination_handler();
                }
            }
#endif
        }

    public:
        template <typename T>
        static void call(std::vector<hpx::future<T>> const& workitems,
            std::list<std::exception_ptr>&, bool = true)
        {
            return call_helper(workitems);
        }

        template <typename T, std::size_t N>
        static void call(std::array<hpx::future<T>, N> const& workitems,
            std::list<std::exception_ptr>&, bool = true)
        {
            return call_helper(workitems);
        }

        template <typename T>
        static void call(std::vector<hpx::shared_future<T>> const& workitems,
            std::list<std::exception_ptr>&, bool = true)
        {
            return call_helper(workitems);
        }

        template <typename T, std::size_t N>
        static void call(std::array<hpx::shared_future<T>, N> const& workitems,
            std::list<std::exception_ptr>&, bool = true)
        {
            return call_helper(workitems);
        }

        template <typename T>
        static void call(
            std::vector<hpx::future<T>> const& workitems, bool = true)
        {
            return call_helper(workitems);
        }

        template <typename T, std::size_t N>
        static void call(
            std::array<hpx::future<T>, N> const& workitems, bool = true)
        {
            return call_helper(workitems);
        }

        template <typename T>
        static void call(
            std::vector<hpx::shared_future<T>> const& workitems, bool = true)
        {
            return call_helper(workitems);
        }

        template <typename T, std::size_t N>
        static void call(
            std::array<hpx::shared_future<T>, N> const& workitems, bool = true)
        {
            return call_helper(workitems);
        }

    private:
        template <typename Future>
        static void call_with_cleanup_helper_single(Future const& f)
        {
#if defined(HPX_COMPUTE_DEVICE_CODE)
            HPX_UNUSED(f);
            HPX_ASSERT(false);
#else
            if (f.has_exception())
            {
                parallel_exception_termination_handler();
            }
#endif
        }

    public:
        template <typename T, typename Cleanup>
        static void call_with_cleanup(hpx::future<T> const& f,
            std::list<std::exception_ptr>&, Cleanup&&, bool = true)
        {
            call_with_cleanup_helper_single(f);
        }

        template <typename T, typename Cleanup,
            typename Enable = std::enable_if_t<!std::is_same_v<
                std::decay_t<Cleanup>, std::list<std::exception_ptr>>>>
        static void call_with_cleanup(
            hpx::future<T> const& f, Cleanup&&, bool = true)
        {
            call_with_cleanup_helper_single(f);
        }

    private:
        template <typename Cont>
        static void call_with_cleanup_helper(Cont const& workitems)
        {
#if defined(HPX_COMPUTE_DEVICE_CODE)
            HPX_UNUSED(workitems);
            HPX_ASSERT(false);
#else
            for (auto const& f : workitems)
            {
                if (f.has_exception())
                {
                    parallel_exception_termination_handler();
                }
            }
#endif
        }

    public:
        template <typename T, typename Cleanup>
        static void call_with_cleanup(
            std::vector<hpx::future<T>> const& workitems,
            std::list<std::exception_ptr>&, Cleanup&&, bool = true)
        {
            call_with_cleanup_helper(workitems);
        }

        template <typename T, std::size_t N, typename Cleanup>
        static void call_with_cleanup(
            std::array<hpx::future<T>, N> const& workitems,
            std::list<std::exception_ptr>&, Cleanup&&, bool = true)
        {
            call_with_cleanup_helper(workitems);
        }

        template <typename T, typename Cleanup,
            typename Enable = std::enable_if_t<!std::is_same_v<
                std::decay_t<Cleanup>, std::list<std::exception_ptr>>>>
        static void call_with_cleanup(
            std::vector<hpx::future<T>> const& workitems, Cleanup&&,
            bool = true)
        {
            call_with_cleanup_helper(workitems);
        }

        template <typename T, std::size_t N, typename Cleanup,
            typename Enable = std::enable_if_t<!std::is_same_v<
                std::decay_t<Cleanup>, std::list<std::exception_ptr>>>>
        static void call_with_cleanup(
            std::array<hpx::future<T>, N> const& workitems, Cleanup&&,
            bool = true)
        {
            call_with_cleanup_helper(workitems);
        }
    };

    template <typename Executor, typename Parameters>
    struct handle_local_exceptions<hpx::execution::
            parallel_unsequenced_task_policy_shim<Executor, Parameters>>
    {
        ///////////////////////////////////////////////////////////////////////
#if defined(HPX_COMPUTE_DEVICE_CODE)
        [[noreturn]] static void call(std::exception_ptr const&)
        {
            std::terminate();
        }
#else
        [[noreturn]] static void call(std::exception_ptr const&)
        {
            parallel_exception_termination_handler();
        }
#endif

#if defined(HPX_COMPUTE_DEVICE_CODE)
        [[noreturn]] static void call(
            std::exception_ptr const&, std::list<std::exception_ptr>&)
        {
            std::terminate();
        }
#else
        [[noreturn]] static void call(
            std::exception_ptr const&, std::list<std::exception_ptr>&)
        {
            parallel_exception_termination_handler();
        }
#endif

        static void call(std::list<std::exception_ptr>& errors)
        {
            if (!errors.empty())
            {
                parallel_exception_termination_handler();
            }
        }

    private:
        template <typename Future>
        static void call_helper_single(Future const& f)
        {
#if defined(HPX_COMPUTE_DEVICE_CODE)
            HPX_UNUSED(f);
            HPX_ASSERT(false);
#else
            if (f.has_exception())
            {
                parallel_exception_termination_handler();
            }
#endif
        }

    public:
        template <typename T>
        static void call(hpx::future<T> const& f,
            std::list<std::exception_ptr>&, bool = true)
        {
            return call_helper_single(f);
        }

        template <typename T>
        static void call(hpx::shared_future<T> const& f,
            std::list<std::exception_ptr>&, bool = true)
        {
            return call_helper_single(f);
        }

        template <typename T>
        static void call(hpx::future<T> const& f, bool = true)
        {
            return call_helper_single(f);
        }

        template <typename T>
        static void call(hpx::shared_future<T> const& f, bool = true)
        {
            return call_helper_single(f);
        }

    private:
        template <typename Cont>
        static void call_helper(Cont const& workitems)
        {
#if defined(HPX_COMPUTE_DEVICE_CODE)
            HPX_UNUSED(workitems);
            HPX_ASSERT(false);
#else
            for (auto const& f : workitems)
            {
                if (f.has_exception())
                {
                    parallel_exception_termination_handler();
                }
            }
#endif
        }

    public:
        template <typename T>
        static void call(std::vector<hpx::future<T>> const& workitems,
            std::list<std::exception_ptr>&, bool = true)
        {
            return call_helper(workitems);
        }

        template <typename T, std::size_t N>
        static void call(std::array<hpx::future<T>, N> const& workitems,
            std::list<std::exception_ptr>&, bool = true)
        {
            return call_helper(workitems);
        }

        template <typename T>
        static void call(std::vector<hpx::shared_future<T>> const& workitems,
            std::list<std::exception_ptr>&, bool = true)
        {
            return call_helper(workitems);
        }

        template <typename T, std::size_t N>
        static void call(std::array<hpx::shared_future<T>, N> const& workitems,
            std::list<std::exception_ptr>&, bool = true)
        {
            return call_helper(workitems);
        }

        template <typename T>
        static void call(
            std::vector<hpx::future<T>> const& workitems, bool = true)
        {
            return call_helper(workitems);
        }

        template <typename T, std::size_t N>
        static void call(
            std::array<hpx::future<T>, N> const& workitems, bool = true)
        {
            return call_helper(workitems);
        }

        template <typename T>
        static void call(
            std::vector<hpx::shared_future<T>> const& workitems, bool = true)
        {
            return call_helper(workitems);
        }

        template <typename T, std::size_t N>
        static void call(
            std::array<hpx::shared_future<T>, N> const& workitems, bool = true)
        {
            return call_helper(workitems);
        }

    private:
        template <typename Future>
        static void call_with_cleanup_helper_single(Future const& f)
        {
#if defined(HPX_COMPUTE_DEVICE_CODE)
            HPX_UNUSED(f);
            HPX_ASSERT(false);
#else
            if (f.has_exception())
            {
                parallel_exception_termination_handler();
            }
#endif
        }

    public:
        template <typename T, typename Cleanup>
        static void call_with_cleanup(hpx::future<T> const& f,
            std::list<std::exception_ptr>&, Cleanup&&, bool = true)
        {
            call_with_cleanup_helper_single(f);
        }

        template <typename T, typename Cleanup,
            typename Enable = std::enable_if_t<!std::is_same_v<
                std::decay_t<Cleanup>, std::list<std::exception_ptr>>>>
        static void call_with_cleanup(
            hpx::future<T> const& f, Cleanup&&, bool = true)
        {
            call_with_cleanup_helper_single(f);
        }

    private:
        template <typename Cont>
        static void call_with_cleanup_helper(Cont const& workitems)
        {
#if defined(HPX_COMPUTE_DEVICE_CODE)
            HPX_UNUSED(workitems);
            HPX_ASSERT(false);
#else
            for (auto const& f : workitems)
            {
                if (f.has_exception())
                {
                    parallel_exception_termination_handler();
                }
            }
#endif
        }

    public:
        template <typename T, typename Cleanup>
        static void call_with_cleanup(
            std::vector<hpx::future<T>> const& workitems,
            std::list<std::exception_ptr>&, Cleanup&&, bool = true)
        {
            call_with_cleanup_helper(workitems);
        }

        template <typename T, std::size_t N, typename Cleanup>
        static void call_with_cleanup(
            std::array<hpx::future<T>, N> const& workitems,
            std::list<std::exception_ptr>&, Cleanup&&, bool = true)
        {
            call_with_cleanup_helper(workitems);
        }

        template <typename T, typename Cleanup,
            typename Enable = std::enable_if_t<!std::is_same_v<
                std::decay_t<Cleanup>, std::list<std::exception_ptr>>>>
        static void call_with_cleanup(
            std::vector<hpx::future<T>> const& workitems, Cleanup&&,
            bool = true)
        {
            call_with_cleanup_helper(workitems);
        }

        template <typename T, std::size_t N, typename Cleanup,
            typename Enable = std::enable_if_t<!std::is_same_v<
                std::decay_t<Cleanup>, std::list<std::exception_ptr>>>>
        static void call_with_cleanup(
            std::array<hpx::future<T>, N> const& workitems, Cleanup&&,
            bool = true)
        {
            call_with_cleanup_helper(workitems);
        }
    };

    template <>
    struct handle_local_exceptions<hpx::execution::unsequenced_policy>
      : handle_local_exceptions<hpx::execution::parallel_unsequenced_policy>
    {
    };

    template <>
    struct handle_local_exceptions<hpx::execution::unsequenced_task_policy>
      : handle_local_exceptions<
            hpx::execution::parallel_unsequenced_task_policy>
    {
    };

    template <typename Executor, typename Parameters>
    struct handle_local_exceptions<
        hpx::execution::unsequenced_policy_shim<Executor, Parameters>>
      : handle_local_exceptions<hpx::execution::
                parallel_unsequenced_policy_shim<Executor, Parameters>>
    {
    };

    template <typename Executor, typename Parameters>
    struct handle_local_exceptions<
        hpx::execution::unsequenced_task_policy_shim<Executor, Parameters>>
      : handle_local_exceptions<hpx::execution::
                parallel_unsequenced_task_policy_shim<Executor, Parameters>>
    {
    };
}}}}    // namespace hpx::parallel::util::detail
