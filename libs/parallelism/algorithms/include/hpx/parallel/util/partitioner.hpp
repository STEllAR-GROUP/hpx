//  Copyright (c) 2007-2018 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/async_local/dataflow.hpp>
#endif
#include <hpx/async_combinators/wait_all.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/type_support/unused.hpp>

#include <hpx/execution/executors/execution.hpp>
#include <hpx/execution/executors/execution_parameters.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/util/detail/chunk_size.hpp>
#include <hpx/parallel/util/detail/handle_local_exceptions.hpp>
#include <hpx/parallel/util/detail/partitioner_iteration.hpp>
#include <hpx/parallel/util/detail/scoped_executor_parameters.hpp>
#include <hpx/parallel/util/detail/select_partitioner.hpp>

#include <cstddef>
#include <exception>
#include <iterator>
#include <list>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace util {
    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        template <typename Result, typename ExPolicy, typename FwdIter,
            typename F>
        std::vector<hpx::future<Result>> partition(
            ExPolicy&& policy, FwdIter first, std::size_t count, F&& f)
        {
            // estimate a chunk size based on number of cores used
            using parameters_type =
                typename std::decay<ExPolicy>::type::executor_parameters_type;
            using has_variable_chunk_size =
                typename execution::extract_has_variable_chunk_size<
                    parameters_type>::type;

            std::vector<hpx::future<Result>> inititems;
            auto shape = detail::get_bulk_iteration_shape(
                has_variable_chunk_size{}, std::forward<ExPolicy>(policy),
                inititems, f, first, count, 1);

            std::vector<hpx::future<Result>> workitems =
                execution::bulk_async_execute(policy.executor(),
                    partitioner_iteration<Result, F>{std::forward<F>(f)},
                    std::move(shape));

            if (inititems.empty())
                return workitems;

            // add the newly created workitems to the list
            inititems.insert(inititems.end(),
                std::make_move_iterator(workitems.begin()),
                std::make_move_iterator(workitems.end()));
            return inititems;
        }

        template <typename Result, typename ExPolicy, typename FwdIter,
            typename Stride, typename F>
        std::vector<hpx::future<Result>> partition_with_index(ExPolicy&& policy,
            FwdIter first, std::size_t count, Stride stride, F&& f)
        {
            // estimate a chunk size based on number of cores used
            using parameters_type =
                typename std::decay<ExPolicy>::type::executor_parameters_type;
            using has_variable_chunk_size =
                typename execution::extract_has_variable_chunk_size<
                    parameters_type>::type;

            std::vector<hpx::future<Result>> inititems;
            auto shape = detail::get_bulk_iteration_shape_idx(
                has_variable_chunk_size{}, std::forward<ExPolicy>(policy),
                inititems, f, first, count, stride);

            std::vector<hpx::future<Result>> workitems =
                execution::bulk_async_execute(policy.executor(),
                    partitioner_iteration<Result, F>{std::forward<F>(f)},
                    std::move(shape));

            if (inititems.empty())
                return workitems;

            // add the newly created workitems to the list
            inititems.insert(inititems.end(),
                std::make_move_iterator(workitems.begin()),
                std::make_move_iterator(workitems.end()));
            return inititems;
        }

        template <typename Result, typename ExPolicy, typename FwdIter,
            typename Data, typename F>
        // requires is_container<Data>
        std::vector<hpx::future<Result>> partition_with_data(ExPolicy&& policy,
            FwdIter first, std::size_t count,
            std::vector<std::size_t> const& chunk_sizes, Data&& data, F&& f)
        {
            HPX_ASSERT(hpx::util::size(data) >= hpx::util::size(chunk_sizes));

            typedef typename std::decay<Data>::type data_type;

            typename data_type::const_iterator data_it = hpx::util::begin(data);
            typename std::vector<std::size_t>::const_iterator chunk_size_it =
                hpx::util::begin(chunk_sizes);

            typedef typename hpx::tuple<typename data_type::value_type, FwdIter,
                std::size_t>
                tuple_type;

            // schedule every chunk on a separate thread
            std::vector<tuple_type> shape;
            shape.reserve(chunk_sizes.size());

            while (count != 0)
            {
                std::size_t chunk = (std::min)(count, *chunk_size_it);
                HPX_ASSERT(chunk != 0);

                shape.push_back(hpx::make_tuple(*data_it, first, chunk));

                count -= chunk;
                std::advance(first, chunk);

                ++data_it;
                ++chunk_size_it;
            }
            HPX_ASSERT(chunk_size_it == chunk_sizes.end());

            return execution::bulk_async_execute(policy.executor(),
                partitioner_iteration<Result, F>{std::forward<F>(f)},
                std::move(shape));
        }

        ///////////////////////////////////////////////////////////////////////
        // The static partitioner simply spawns one chunk of iterations for
        // each available core.
        template <typename ExPolicy, typename R, typename Result>
        struct static_partitioner
        {
            using parameters_type = typename ExPolicy::executor_parameters_type;
            using executor_type = typename ExPolicy::executor_type;

            using scoped_executor_parameters =
                detail::scoped_executor_parameters_ref<parameters_type,
                    executor_type>;

            using handle_local_exceptions =
                detail::handle_local_exceptions<ExPolicy>;

            template <typename ExPolicy_, typename FwdIter, typename F1,
                typename F2>
            static R call(ExPolicy_&& policy, FwdIter first, std::size_t count,
                F1&& f1, F2&& f2)
            {
                // inform parameter traits
                scoped_executor_parameters scoped_params(
                    policy.parameters(), policy.executor());

                std::vector<hpx::future<Result>> workitems;
                std::list<std::exception_ptr> errors;
                try
                {
                    workitems = detail::partition<Result>(
                        std::forward<ExPolicy_>(policy), first, count,
                        std::forward<F1>(f1));

                    scoped_params.mark_end_of_scheduling();
                }
                catch (...)
                {
                    handle_local_exceptions::call(
                        std::current_exception(), errors);
                }
                return reduce(std::move(workitems), std::move(errors),
                    std::forward<F2>(f2));
            }

            template <typename ExPolicy_, typename FwdIter, typename Stride,
                typename F1, typename F2>
            static R call_with_index(ExPolicy_&& policy, FwdIter first,
                std::size_t count, Stride stride, F1&& f1, F2&& f2)
            {
                // inform parameter traits
                scoped_executor_parameters scoped_params(
                    policy.parameters(), policy.executor());

                std::vector<hpx::future<Result>> workitems;
                std::list<std::exception_ptr> errors;
                try
                {
                    workitems = detail::partition_with_index<Result>(
                        std::forward<ExPolicy_>(policy), first, count, stride,
                        std::forward<F1>(f1));

                    scoped_params.mark_end_of_scheduling();
                }
                catch (...)
                {
                    handle_local_exceptions::call(
                        std::current_exception(), errors);
                }
                return reduce(std::move(workitems), std::move(errors),
                    std::forward<F2>(f2));
            }

            template <typename ExPolicy_, typename FwdIter, typename F1,
                typename F2, typename Data>
            // requires is_container<Data>
            static R call_with_data(ExPolicy_&& policy, FwdIter first,
                std::size_t count, F1&& f1, F2&& f2,
                std::vector<std::size_t> const& chunk_sizes, Data&& data)
            {
                // inform parameter traits
                scoped_executor_parameters scoped_params(
                    policy.parameters(), policy.executor());

                std::vector<hpx::future<Result>> workitems;
                std::list<std::exception_ptr> errors;
                try
                {
                    workitems = detail::partition_with_data<Result>(
                        std::forward<ExPolicy_>(policy), first, count,
                        chunk_sizes, std::forward<Data>(data),
                        std::forward<F1>(f1));

                    scoped_params.mark_end_of_scheduling();
                }
                catch (...)
                {
                    handle_local_exceptions::call(
                        std::current_exception(), errors);
                }
                return reduce(std::move(workitems), std::move(errors),
                    std::forward<F2>(f2));
            }

        private:
            template <typename F>
            static R reduce(std::vector<hpx::future<Result>>&& workitems,
                std::list<std::exception_ptr>&& errors, F&& f)
            {
                // wait for all tasks to finish
                hpx::wait_all(workitems);

                // always rethrow if 'errors' is not empty or workitems has
                // exceptional future
                handle_local_exceptions::call(workitems, errors);

                try
                {
                    return f(std::move(workitems));
                }
                catch (...)
                {
                    // rethrow either bad_alloc or exception_list
                    handle_local_exceptions::call(std::current_exception());
                    HPX_ASSERT(false);
                    return f(std::move(workitems));
                }
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename R, typename Result>
        struct task_static_partitioner
        {
            using parameters_type = typename ExPolicy::executor_parameters_type;
            using executor_type = typename ExPolicy::executor_type;

            using scoped_executor_parameters =
                detail::scoped_executor_parameters<parameters_type,
                    executor_type>;

            using handle_local_exceptions =
                detail::handle_local_exceptions<ExPolicy>;

            template <typename ExPolicy_, typename FwdIter, typename F1,
                typename F2>
            static hpx::future<R> call(ExPolicy_&& policy, FwdIter first,
                std::size_t count, F1&& f1, F2&& f2)
            {
                // inform parameter traits
                std::shared_ptr<scoped_executor_parameters> scoped_params =
                    std::make_shared<scoped_executor_parameters>(
                        policy.parameters(), policy.executor());

                std::vector<hpx::future<Result>> workitems;
                std::list<std::exception_ptr> errors;
                try
                {
                    workitems = detail::partition<Result>(
                        std::forward<ExPolicy_>(policy), first, count,
                        std::forward<F1>(f1));

                    scoped_params->mark_end_of_scheduling();
                }
                catch (std::bad_alloc const&)
                {
                    return hpx::make_exceptional_future<R>(
                        std::current_exception());
                }
                catch (...)
                {
                    handle_local_exceptions::call(
                        std::current_exception(), errors);
                }
                return reduce(std::move(scoped_params), std::move(workitems),
                    std::move(errors), std::forward<F2>(f2));
            }

            template <typename ExPolicy_, typename FwdIter, typename Stride,
                typename F1, typename F2>
            static hpx::future<R> call_with_index(ExPolicy_&& policy,
                FwdIter first, std::size_t count, Stride stride, F1&& f1,
                F2&& f2)
            {
                // inform parameter traits
                std::shared_ptr<scoped_executor_parameters> scoped_params =
                    std::make_shared<scoped_executor_parameters>(
                        policy.parameters(), policy.executor());

                std::vector<hpx::future<Result>> workitems;
                std::list<std::exception_ptr> errors;
                try
                {
                    workitems = detail::partition_with_index<Result>(
                        std::forward<ExPolicy_>(policy), first, count, stride,
                        std::forward<F1>(f1));

                    scoped_params->mark_end_of_scheduling();
                }
                catch (std::bad_alloc const&)
                {
                    return hpx::make_exceptional_future<R>(
                        std::current_exception());
                }
                catch (...)
                {
                    handle_local_exceptions::call(
                        std::current_exception(), errors);
                }
                return reduce(std::move(scoped_params), std::move(workitems),
                    std::move(errors), std::forward<F2>(f2));
            }

            template <typename ExPolicy_, typename FwdIter, typename F1,
                typename F2, typename Data>
            // requires is_container<Data>
            static hpx::future<R> call_with_data(ExPolicy_&& policy,
                FwdIter first, std::size_t count, F1&& f1, F2&& f2,
                std::vector<std::size_t> const& chunk_sizes, Data&& data)
            {
                // inform parameter traits
                std::shared_ptr<scoped_executor_parameters> scoped_params =
                    std::make_shared<scoped_executor_parameters>(
                        policy.parameters(), policy.executor());

                std::vector<hpx::future<Result>> workitems;
                std::list<std::exception_ptr> errors;
                try
                {
                    workitems = detail::partition_with_data<Result>(
                        std::forward<ExPolicy_>(policy), first, count,
                        chunk_sizes, std::forward<Data>(data),
                        std::forward<F1>(f1));

                    scoped_params->mark_end_of_scheduling();
                }
                catch (std::bad_alloc const&)
                {
                    return hpx::make_exceptional_future<R>(
                        std::current_exception());
                }
                catch (...)
                {
                    handle_local_exceptions::call(
                        std::current_exception(), errors);
                }
                return reduce(std::move(scoped_params), std::move(workitems),
                    std::move(errors), std::forward<F2>(f2));
            }

        private:
            template <typename F>
            static hpx::future<R> reduce(
                std::shared_ptr<scoped_executor_parameters>&& scoped_params,
                std::vector<hpx::future<Result>>&& workitems,
                std::list<std::exception_ptr>&& errors, F&& f)
            {
#if defined(HPX_COMPUTE_DEVICE_CODE)
                HPX_UNUSED(scoped_params);
                HPX_UNUSED(workitems);
                HPX_UNUSED(errors);
                HPX_UNUSED(f);
                HPX_ASSERT(false);
                return hpx::future<R>();
#else
                // wait for all tasks to finish
                return hpx::dataflow(
                    [errors = std::move(errors),
                        scoped_params = std::move(scoped_params),
                        f = std::forward<F>(f)](
                        std::vector<hpx::future<Result>>&& r) mutable -> R {
                        HPX_UNUSED(scoped_params);

                        handle_local_exceptions::call(r, errors);
                        return f(std::move(r));
                    },
                    std::move(workitems));
#endif
            }
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    // ExPolicy: execution policy
    // R:        overall result type
    // Result:   intermediate result type of first step
    template <typename ExPolicy, typename R = void, typename Result = R>
    struct partitioner
      : detail::select_partitioner<typename std::decay<ExPolicy>::type,
            detail::static_partitioner,
            detail::task_static_partitioner>::template apply<R, Result>
    {
    };
}}}    // namespace hpx::parallel::util
