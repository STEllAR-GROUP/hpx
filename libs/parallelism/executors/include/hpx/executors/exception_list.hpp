//  Copyright (c) 2007-2017 Hartmut Kaiser
//                2017-2018 Taeguk Kwon
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/executors/execution_policy_fwd.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/functional.hpp>
#include <hpx/type_support/decay.hpp>

#include <exception>
#include <utility>

namespace hpx { namespace parallel { inline namespace v1 {
    namespace detail {
        /// \cond NOINTERNAL
        template <typename ExPolicy, typename Result = void>
        struct handle_exception_impl
        {
            typedef Result type;

            HPX_NORETURN static Result call()
            {
                try
                {
                    throw;    //-V667
                }
                catch (std::bad_alloc const&)
                {
                    throw;
                }
                catch (hpx::exception_list const&)
                {
                    throw;
                }
                catch (...)
                {
                    throw hpx::exception_list(std::current_exception());
                }
            }

            static Result call(hpx::future<Result> f)
            {
                HPX_ASSERT(f.has_exception());

                return f.get();
            }

            HPX_NORETURN static Result call(std::exception_ptr const& e)
            {
                try
                {
                    std::rethrow_exception(e);
                }
                catch (std::bad_alloc const&)
                {
                    throw;
                }
                catch (hpx::exception_list const&)
                {
                    throw;
                }
                catch (...)
                {
                    throw hpx::exception_list(std::current_exception());
                }
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Result>
        struct handle_exception_task_impl
        {
            typedef future<Result> type;

            static future<Result> call()
            {
                try
                {
                    throw;    //-V667
                }
                catch (std::bad_alloc const& e)
                {
                    return hpx::make_exceptional_future<Result>(e);
                }
                catch (hpx::exception_list const& el)
                {
                    return hpx::make_exceptional_future<Result>(el);
                }
                catch (...)
                {
                    return hpx::make_exceptional_future<Result>(
                        hpx::exception_list(std::current_exception()));
                }
            }

            static future<Result> call(future<Result> f)
            {
                HPX_ASSERT(f.has_exception());
                // Intel complains if this is not explicitly moved
#if defined(HPX_INTEL_VERSION)
                return std::move(f);
#else
                return f;
#endif
            }

            static future<Result> call(std::exception_ptr const& e)
            {
                try
                {
                    std::rethrow_exception(e);
                }
                catch (std::bad_alloc const&)
                {
                    return hpx::make_exceptional_future<Result>(
                        std::current_exception());
                }
                catch (hpx::exception_list const& el)
                {
                    return hpx::make_exceptional_future<Result>(el);
                }
                catch (...)
                {
                    return hpx::make_exceptional_future<Result>(
                        hpx::exception_list(std::current_exception()));
                }
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Result>
        struct handle_exception_impl<hpx::execution::sequenced_task_policy,
            Result> : handle_exception_task_impl<Result>
        {
        };

        template <typename Executor, typename Parameters, typename Result>
        struct handle_exception_impl<
            hpx::execution::sequenced_task_policy_shim<Executor, Parameters>,
            Result> : handle_exception_task_impl<Result>
        {
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Result>
        struct handle_exception_impl<hpx::execution::parallel_task_policy,
            Result> : handle_exception_task_impl<Result>
        {
        };

        template <typename Executor, typename Parameters, typename Result>
        struct handle_exception_impl<
            hpx::execution::parallel_task_policy_shim<Executor, Parameters>,
            Result> : handle_exception_task_impl<Result>
        {
        };

#if defined(HPX_HAVE_DATAPAR)
        ///////////////////////////////////////////////////////////////////////
        template <typename Result>
        struct handle_exception_impl<hpx::execution::dataseq_task_policy,
            Result> : handle_exception_task_impl<Result>
        {
        };

        template <typename Result>
        struct handle_exception_impl<hpx::execution::datapar_task_policy,
            Result> : handle_exception_task_impl<Result>
        {
        };
#endif

        using exception_list_termination_handler_type =
            hpx::util::function_nonser<void()>;
        HPX_PARALLELISM_EXPORT void set_exception_list_termination_handler(
            exception_list_termination_handler_type f);
        HPX_NORETURN HPX_PARALLELISM_EXPORT void
        exception_list_termination_handler();

        ///////////////////////////////////////////////////////////////////////
        template <typename Result>
        struct handle_exception_impl<
            hpx::execution::parallel_unsequenced_policy, Result>
        {
            typedef Result type;

            HPX_NORETURN static Result call()
            {
                // any exceptions thrown by algorithms executed with the
                // parallel_unsequenced_policy are to call terminate.
                exception_list_termination_handler();
            }

            HPX_NORETURN
            static hpx::future<Result> call(hpx::future<Result>&&)
            {
                exception_list_termination_handler();
            }

            HPX_NORETURN
            static hpx::future<Result> call(std::exception_ptr const&)
            {
                exception_list_termination_handler();
            }
        };

        template <typename ExPolicy, typename Result = void>
        struct handle_exception
          : handle_exception_impl<typename hpx::util::decay<ExPolicy>::type,
                Result>
        {
        };
        /// \endcond
    }    // namespace detail

    // we're just reusing our existing implementation

    using hpx::exception_list;
}}}    // namespace hpx::parallel::v1
