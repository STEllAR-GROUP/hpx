//  Copyright (c) 2007-2016 Hartmut Kaiser
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

#include <exception>
#include <list>
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
        static void call(std::exception_ptr const& e)
        {
            HPX_ASSERT(false);
        }
#else
        HPX_NORETURN static void call(std::exception_ptr const& e)
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

        template <typename T>
        static void call(std::vector<hpx::future<T>> const& workitems,
            std::list<std::exception_ptr>& errors, bool throw_errors = true)
        {
#if defined(HPX_COMPUTE_DEVICE_CODE)
            HPX_ASSERT(false);
#else
            for (hpx::future<T> const& f : workitems)
            {
                if (f.has_exception())
                    call(f.get_exception_ptr(), errors);
            }

            if (throw_errors && !errors.empty())
                throw exception_list(std::move(errors));
#endif
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename T>
        static void call(std::vector<hpx::shared_future<T>> const& workitems,
            std::list<std::exception_ptr>& errors, bool throw_errors = true)
        {
#if defined(HPX_COMPUTE_DEVICE_CODE)
            HPX_ASSERT(false);
#else
            for (hpx::shared_future<T> const& f : workitems)
            {
                if (f.has_exception())
                    call(f.get_exception_ptr(), errors);
            }

            if (throw_errors && !errors.empty())
                throw exception_list(std::move(errors));
#endif
        }

        template <typename T, typename Cleanup>
        static void call_with_cleanup(std::vector<hpx::future<T>>& workitems,
            std::list<std::exception_ptr>& errors, Cleanup&& cleanup,
            bool throw_errors = true)
        {
#if defined(HPX_COMPUTE_DEVICE_CODE)
            HPX_ASSERT(false);
#else
            bool has_exception = false;
            std::exception_ptr bad_alloc_exception;
            for (hpx::future<T>& f : workitems)
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
                for (hpx::future<T>& f : workitems)
                {
                    if (!f.has_exception())
                        cleanup(f.get());
                }
            }

            if (bad_alloc_exception)
                std::rethrow_exception(bad_alloc_exception);

            if (throw_errors && !errors.empty())
                throw exception_list(std::move(errors));
#endif
        }
    };

    template <>
    struct handle_local_exceptions<hpx::execution::parallel_unsequenced_policy>
    {
        ///////////////////////////////////////////////////////////////////////
#if defined(HPX_COMPUTE_DEVICE_CODE)
        static void call(std::exception_ptr const&)
        {
            HPX_ASSERT(false);
        }
#else
        HPX_NORETURN static void call(std::exception_ptr const&)
        {
            parallel_exception_termination_handler();
        }
#endif

#if defined(HPX_COMPUTE_DEVICE_CODE)
        static void call(
            std::exception_ptr const&, std::list<std::exception_ptr>&)
        {
            HPX_ASSERT(false);
        }
#else
        HPX_NORETURN static void call(
            std::exception_ptr const&, std::list<std::exception_ptr>&)
        {
            parallel_exception_termination_handler();
        }
#endif

        template <typename T>
        static void call(std::vector<hpx::future<T>> const& workitems,
            std::list<std::exception_ptr>&, bool = true)
        {
#if defined(HPX_COMPUTE_DEVICE_CODE)
            HPX_ASSERT(false);
#else
            for (hpx::future<T> const& f : workitems)
            {
                if (f.has_exception())
                    parallel_exception_termination_handler();
            }
#endif
        }

        template <typename T>
        static void call(std::vector<hpx::shared_future<T>> const& workitems,
            std::list<std::exception_ptr>&, bool = true)
        {
#if defined(HPX_COMPUTE_DEVICE_CODE)
            HPX_ASSERT(false);
#else
            for (hpx::shared_future<T> const& f : workitems)
            {
                if (f.has_exception())
                    parallel_exception_termination_handler();
            }
#endif
        }

        template <typename T, typename Cleanup>
        static void call_with_cleanup(
            std::vector<hpx::future<T>> const& workitems,
            std::list<std::exception_ptr>&, Cleanup&&, bool = true)
        {
#if defined(HPX_COMPUTE_DEVICE_CODE)
            HPX_ASSERT(false);
#else
            for (hpx::future<T> const& f : workitems)
            {
                if (f.has_exception())
                    parallel_exception_termination_handler();
            }
#endif
        }
    };
}}}}    // namespace hpx::parallel::util::detail
