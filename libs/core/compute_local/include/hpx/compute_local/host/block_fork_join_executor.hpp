//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file block_fork_join_executor.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/compute_local/host/numa_domains.hpp>
#include <hpx/compute_local/host/target.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/errors/try_catch_exception_ptr.hpp>
#include <hpx/execution/executors/execution_parameters.hpp>
#include <hpx/execution_base/this_thread.hpp>
#include <hpx/execution_base/traits/is_executor.hpp>
#include <hpx/executors/fork_join_executor.hpp>
#include <hpx/iterator_support/counting_shape.hpp>
#include <hpx/iterator_support/iterator_range.hpp>
#include <hpx/modules/concepts.hpp>
#include <hpx/modules/topology.hpp>
#include <hpx/resource_partitioner/detail/partitioner.hpp>

#include <chrono>
#include <cstddef>
#include <exception>
#include <iterator>
#include <string>
#include <type_traits>
#include <vector>

namespace hpx::execution::experimental {

    /// \brief An executor with fork-join (blocking) semantics.
    ///
    /// The block_fork_join_executor creates on construction a set of worker
    /// threads that are kept alive for the duration of the executor. Copying
    /// the executor has reference semantics, i.e. copies of a
    /// fork_join_executor hold a reference to the worker threads of the
    /// original instance. Scheduling work through the executor concurrently
    /// from different threads is undefined behaviour.
    ///
    /// The executor keeps a set of worker threads alive for the lifetime of the
    /// executor, meaning other work will not be executed while the executor is
    /// busy or waiting for work. The executor has a customizable delay after
    /// which it will yield to other work.  Since starting and resuming the
    /// worker threads is a slow operation the executor should be reused
    /// whenever possible for multiple adjacent parallel algorithms or
    /// invocations of bulk_(a)sync_execute.
    ///
    /// This behaviour is similar to the plain \a fork_join_executor except that
    /// the block_fork_join_executor creates a hierarchy of fork_join_executors,
    /// one for each target used to initialize it.
    class block_fork_join_executor
    {
        static hpx::threads::mask_type cores_for_targets(
            std::vector<compute::host::target> const& targets)
        {
            auto& rp = hpx::resource::get_partitioner();
            std::size_t this_pu = rp.get_pu_num(hpx::get_worker_thread_num());
            if (targets.size() == 1)
            {
                // don't build a hierarchy of executors if there is only one
                // mask provided
                auto target_mask = targets[0].native_handle().get_device();
                if (!hpx::threads::test(target_mask, this_pu))
                {
                    HPX_THROW_EXCEPTION(hpx::error::bad_parameter,
                        "block_fork_join_executor::cores_for_targets",
                        "The thread used to initialize the "
                        "block_fork_join_executor should be part of the given "
                        "target");
                }
                return target_mask;
            }

            // This makes sure that each given set of targets gets exactly one
            // core assigned that will be used as the 'main' thread for the
            // corresponding fork_join_executor instance. This makes also sure
            // that the executing (current) thread is associated with one of the
            // targets.
            hpx::threads::mask_type mask(hpx::threads::hardware_concurrency());
            bool this_thread_is_represented = false;
            for (auto const& t : targets)
            {
                auto target_mask = t.native_handle().get_device();
                if (!this_thread_is_represented &&
                    hpx::threads::test(target_mask, this_pu))
                {
                    hpx::threads::set(mask, this_pu);
                    this_thread_is_represented = true;
                }
                else
                {
                    hpx::threads::set(
                        mask, hpx::threads::find_first(target_mask));
                }
            }

            // The block_fork_join_executor will expose bad performance if the
            // current thread is not part of any of the given targets.
            if (!this_thread_is_represented)
            {
                HPX_THROW_EXCEPTION(hpx::error::bad_parameter,
                    "block_fork_join_executor::cores_for_targets",
                    "The thread used to initialize the "
                    "block_fork_join_executor should be part of at least one "
                    "of the given targets");
            }
            return mask;
        }

    public:
        /// \cond NOINTERNAL
        using execution_category = hpx::execution::parallel_execution_tag;
        using executor_parameters_type =
            hpx::execution::experimental::static_chunk_size;

        bool operator==(block_fork_join_executor const& rhs) const noexcept
        {
            return exec_ == rhs.exec_ && block_execs_ == rhs.block_execs_;
        }

        bool operator!=(block_fork_join_executor const& rhs) const noexcept
        {
            return !(*this == rhs);
        }

        block_fork_join_executor const& context() const noexcept
        {
            return *this;
        }
        /// \endcond

        /// \brief Construct a block_fork_join_executor.
        ///
        /// \param priority The priority of the worker threads.
        /// \param stacksize The stacksize of the worker threads. Must not be
        ///                  nostack.
        /// \param schedule The loop schedule of the parallel regions.
        /// \param yield_delay The time after which the executor yields to
        ///        other work if it hasn't received any new work for bulk
        ///        execution.
        ///
        /// \note   This constructor will create one fork_join_executor for
        ///         each numa domain
        explicit block_fork_join_executor(
            threads::thread_priority priority = threads::thread_priority::bound,
            threads::thread_stacksize stacksize =
                threads::thread_stacksize::small_,
            fork_join_executor::loop_schedule schedule =
                fork_join_executor::loop_schedule::static_,
            std::chrono::nanoseconds yield_delay = std::chrono::milliseconds(1))
          : block_fork_join_executor(compute::host::numa_domains(), priority,
                stacksize, schedule, yield_delay)
        {
        }

        /// \brief Construct a block_fork_join_executor.
        ///
        /// \param targets  The list of targets to use for thread placement
        /// \param priority The priority of the worker threads.
        /// \param stacksize The stacksize of the worker threads. Must not be
        ///                  nostack.
        /// \param schedule The loop schedule of the parallel regions.
        /// \param yield_delay The time after which the executor yields to
        ///        other work if it hasn't received any new work for bulk
        ///        execution.
        ///
        /// \note   This constructor will create one fork_join_executor for
        ///         each given target
        explicit block_fork_join_executor(
            std::vector<compute::host::target> const& targets,
            threads::thread_priority priority = threads::thread_priority::bound,
            threads::thread_stacksize stacksize =
                threads::thread_stacksize::small_,
            fork_join_executor::loop_schedule schedule =
                fork_join_executor::loop_schedule::static_,
            std::chrono::nanoseconds yield_delay = std::chrono::milliseconds(1))
          : exec_(cores_for_targets(targets), priority, stacksize,
                targets.size() == 1 ?
                    schedule :
                    fork_join_executor::loop_schedule::static_,
                yield_delay)
        {
            // don't build a hierarchy of executors if there is only one target
            // mask given
            if (targets.size() > 1)
            {
                block_execs_.reserve(targets.size());
                for (std::size_t i = 0; i != targets.size(); ++i)
                {
                    block_execs_.emplace_back(
                        fork_join_executor::init_mode::no_init);
                }

                auto init_f = [&](std::size_t index) {
                    // create the sub-executors
                    block_execs_[index] = fork_join_executor(
                        targets[index].native_handle().get_device(), priority,
                        stacksize, schedule, yield_delay);
                };

                hpx::parallel::execution::bulk_sync_execute(
                    exec_, init_f, hpx::util::counting_shape(targets.size()));
            }
        }

        template <typename F, typename S, typename... Ts>
        void bulk_sync_execute_helper(F&& f, S const& shape, Ts&&... ts)
        {
            std::size_t num_targets = block_execs_.size();
            if (num_targets == 0)
            {
                // simply forward call if there is no executor hierarchy
                hpx::parallel::execution::bulk_sync_execute(
                    exec_, HPX_FORWARD(F, f), shape, HPX_FORWARD(Ts, ts)...);
                return;
            }

            std::size_t size = std::size(shape);
            auto outer_func = [&](std::size_t index, auto&& f,
                                  auto const& shape, auto&&... ts) {
                // calculate the inner shape dimensions
                auto const part_begin = (index * size) / num_targets;
                auto const part_end = ((index + 1) * size) / num_targets;

                auto begin = std::next(std::begin(shape), part_begin);
                auto inner_shape = hpx::util::iterator_range(
                    begin, std::next(begin, part_end - part_begin));

                // invoke bulk_sync_execute on one of the inner executors
                hpx::parallel::execution::bulk_sync_execute(block_execs_[index],
                    HPX_FORWARD(decltype(f), f), inner_shape,
                    HPX_FORWARD(decltype(ts), ts)...);
            };

            auto outer_shape = hpx::util::counting_shape(num_targets);

            hpx::parallel::execution::bulk_sync_execute(exec_, outer_func,
                outer_shape, HPX_FORWARD(F, f), shape, HPX_FORWARD(Ts, ts)...);
        }

        template <typename F, typename S, typename... Ts>
        friend void tag_invoke(hpx::parallel::execution::bulk_sync_execute_t,
            block_fork_join_executor& exec, F&& f, S const& shape, Ts&&... ts)
        {
            exec.bulk_sync_execute_helper(
                HPX_FORWARD(F, f), shape, HPX_FORWARD(Ts, ts)...);
        }

        template <typename F, typename S, typename... Ts>
        friend decltype(auto) tag_invoke(
            hpx::parallel::execution::bulk_async_execute_t,
            block_fork_join_executor& exec, F&& f, S const& shape, Ts&&... ts)
        {
            // Forward to the synchronous version as we can't create
            // futures to the completion of the parallel region (this HPX
            // thread participates in computation).
            return hpx::detail::try_catch_exception_ptr(
                [&]() {
                    exec.bulk_sync_execute_helper(
                        HPX_FORWARD(F, f), shape, HPX_FORWARD(Ts, ts)...);
                    return hpx::make_ready_future();
                },
                [&](std::exception_ptr&& ep) {
                    return hpx::make_exceptional_future<void>(HPX_MOVE(ep));
                });
        }

        // support all properties that are exposed by the fork_join_executor
        // clang-format off
        template <typename Tag, typename Property,
            HPX_CONCEPT_REQUIRES_(
                hpx::execution::experimental::is_scheduling_property_v<Tag> &&
                hpx::functional::is_tag_invocable_v<
                    Tag, fork_join_executor, Property>
            )>
        // clang-format on
        friend block_fork_join_executor tag_invoke(Tag tag,
            block_fork_join_executor const& exec, Property&& prop) noexcept
        {
            auto exec_with_prop = exec;
            exec_with_prop.exec_ = hpx::functional::tag_invoke(
                tag, exec.exec_, HPX_FORWARD(Property, prop));
            return exec_with_prop;
        }

        // clang-format off
        template <typename Tag,
            HPX_CONCEPT_REQUIRES_(
                hpx::execution::experimental::is_scheduling_property_v<Tag> &&
                hpx::functional::is_tag_invocable_v<Tag, fork_join_executor>
            )>
        // clang-format on
        friend decltype(auto) tag_invoke(
            Tag tag, block_fork_join_executor const& exec) noexcept
        {
            return hpx::functional::tag_invoke(tag, exec.exec_);
        }

    private:
        fork_join_executor exec_;
        std::vector<fork_join_executor> block_execs_;
    };
}    // namespace hpx::execution::experimental

namespace hpx::parallel::execution {
    /// \cond NOINTERNAL
    template <>
    struct is_bulk_one_way_executor<
        hpx::execution::experimental::block_fork_join_executor> : std::true_type
    {
    };

    template <>
    struct is_bulk_two_way_executor<
        hpx::execution::experimental::block_fork_join_executor> : std::true_type
    {
    };
    /// \endcond
}    // namespace hpx::parallel::execution
