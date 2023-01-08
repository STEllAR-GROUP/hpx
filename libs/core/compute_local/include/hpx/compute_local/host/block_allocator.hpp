///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2020 ETH Zurich
//  Copyright (c) 2016 Thomas Heller
//  Copyright (c) 2016-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <hpx/config.hpp>

#include <hpx/allocator_support/detail/new.hpp>
#include <hpx/compute_local/host/block_executor.hpp>
#include <hpx/compute_local/host/target.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/execution/executors/static_chunk_size.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/executors/restricted_thread_pool_executor.hpp>
#include <hpx/functional/invoke_fused.hpp>
#include <hpx/iterator_support/counting_shape.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/parallel/container_algorithms/for_each.hpp>
#include <hpx/parallel/util/adapt_sharing_mode.hpp>
#include <hpx/parallel/util/cancellation_token.hpp>
#include <hpx/parallel/util/partitioner_with_cleanup.hpp>
#include <hpx/topology/topology.hpp>

#include <cstddef>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace compute { namespace host {
    namespace detail {
        /// The policy_allocator allocates blocks of memory touched according to
        /// the distribution policy of the given executor.
        template <typename T, typename Policy,
            typename Enable =
                std::enable_if_t<hpx::is_execution_policy_v<Policy>>>
        struct policy_allocator
        {
            using policy_type = Policy;
            using target_type = host::target;

            using value_type = T;
            using pointer = T*;
            using const_pointer = T const*;
            using reference = T&;
            using const_reference = T const&;
            using size_type = std::size_t;

            using propagate_on_container_move_assignment = std::true_type;

            template <typename U>
            struct rebind
            {
                using other = policy_allocator<U, policy_type>;
            };

            policy_allocator(Policy&& policy)
              : policy_(HPX_MOVE(policy))
            {
            }

            policy_allocator(Policy const& policy)
              : policy_(policy)
            {
            }

            policy_type const& policy() const
            {
                return policy_;
            }

            // Returns the actual address of x even in presence of overloaded
            // operator&
            pointer address(reference x) const noexcept
            {
                return &x;
            }

            const_pointer address(const_reference x) const noexcept
            {
                return &x;
            }

            // Allocates n * sizeof(T) bytes of uninitialized storage by calling
            // topo.allocate(). The pointer hint may be used to provide locality of
            // reference: the allocator, if supported by the implementation, will
            // attempt to allocate the new memory block as close as possible to hint.
            pointer allocate(size_type n, void const* /* hint */ = nullptr)
            {
                return reinterpret_cast<pointer>(
                    hpx::threads::create_topology().allocate(n * sizeof(T)));
            }

            // Deallocates the storage referenced by the pointer p, which must be a
            // pointer obtained by an earlier call to allocate(). The argument n
            // must be equal to the first argument of the call to allocate() that
            // originally produced p; otherwise, the behavior is undefined.
            void deallocate(pointer p, size_type n) noexcept
            {
                try
                {
                    hpx::threads::create_topology().deallocate(p, n);
                }
                catch (...)
                {
                    ;    // just ignore errors from create_topology
                }
            }

            // Returns the maximum theoretically possible value of n, for which the
            // call allocate(n, 0) could succeed. In most implementations, this
            // returns std::numeric_limits<size_type>::max() / sizeof(value_type).
            size_type max_size() const noexcept
            {
                return (std::numeric_limits<size_type>::max)();
            }

        public:
            // Constructs count objects of type T in allocated uninitialized
            // storage pointed to by p, using placement-new. This will use the
            // underlying executors to distribute the memory according to
            // first touch memory placement.
            template <typename U, typename... Args>
            void bulk_construct(U* p, std::size_t count, Args&&... args)
            {
                if (count == std::size_t(0))
                {
                    return;
                }

                auto irange = hpx::util::counting_shape(count);

                using iterator_type = hpx::util::counting_iterator<std::size_t>;
                using partition_result_type =
                    std::pair<iterator_type, iterator_type>;

                using partitioner =
                    parallel::util::partitioner_with_cleanup<decltype(policy_),
                        void, partition_result_type>;
                using cancellation_token = parallel::util::cancellation_token<
                    parallel::util::detail::no_data>;

                auto&& arguments =
                    hpx::forward_as_tuple(HPX_FORWARD(Args, args)...);

                decltype(auto) hinted_policy =
                    parallel::util::adapt_sharing_mode(policy_,
                        hpx::threads::thread_sharing_hint::
                            do_not_share_function);

                cancellation_token tok;
                partitioner::call(
                    hinted_policy, util::begin(irange), count,
                    [&arguments, p, &tok](
                        iterator_type it, std::size_t part_size) mutable
                    -> partition_result_type {
                        iterator_type last =
                            parallel::util::loop_with_cleanup_n_with_token(
                                it, part_size, tok,
                                [&arguments, p](iterator_type it) {
                                    using hpx::util::functional::
                                        placement_new_one;
                                    hpx::invoke_fused(
                                        placement_new_one<U>(p + *it),
                                        arguments);
                                },
                                // cleanup function, called for all elements of
                                // current partition which succeeded before exception
                                [p](iterator_type it) {
                                    std::destroy_at(p + *it);
                                });
                        return std::make_pair(it, last);
                    },
                    // finalize, called once if no error occurred
                    [](auto&&) {
                        // do nothing
                    },
                    // cleanup function, called for each partition which
                    // didn't fail, but only if at least one failed
                    [p](partition_result_type&& r) -> void {
                        while (r.first != r.second)
                        {
                            std::destroy_at(p + *r.first);
                            ++r.first;
                        }
                    });
            }

            // Constructs an object of type T in allocated uninitialized storage
            // pointed to by p, using placement-new
            template <typename U, typename... Args>
            void construct(U* p, Args&&... args)
            {
                hpx::parallel::execution::sync_execute(
                    hpx::util::functional::placement_new<U>(), p,
                    HPX_FORWARD(Args, args)...);
            }

            // Calls the destructor of count objects pointed to by p
            template <typename U>
            void bulk_destroy(U* p, std::size_t count)
            {
                if (count == std::size_t(0))
                {
                    return;
                }

                // keep memory locality, use executor...
                auto irange = hpx::util::counting_shape(count);
                hpx::ranges::for_each(policy_, irange,
                    [p](std::size_t i) { std::destroy_at(p + i); });
            }

            // Calls the destructor of the object pointed to by p
            template <typename U>
            void destroy(U* p)
            {
                std::destroy_at(p);
            }

            // Required by hpx::compute::traits::allocator_traits. Return
            // default target.
            target_type const& target() const noexcept
            {
                return target_;
            }

        private:
            target_type target_;
            policy_type policy_;
        };
    }    // namespace detail

    /// The block_allocator allocates blocks of memory evenly divided onto the
    /// passed vector of targets. This is done by using first touch memory
    /// placement.
    ///
    /// This allocator can be used to write NUMA aware algorithms:
    ///
    /// using allocator_type = hpx::compute::host::block_allocator<int>;
    /// using vector_type = hpx::compute::vector<int, allocator_type>;
    ///
    /// auto numa_nodes = hpx::compute::host::numa_domains();
    /// std::size_t N = 2048;
    /// vector_type v(N, allocator_type(numa_nodes));
    ///
    template <typename T,
        typename Executor =
            hpx::parallel::execution::restricted_thread_pool_executor>
    struct block_allocator
      : public detail::policy_allocator<T,
            hpx::execution::detail::parallel_policy_shim<
                block_executor<Executor>,
                typename block_executor<Executor>::executor_parameters_type>>
    {
        using executor_type = block_executor<Executor>;
        using executor_parameters_type =
            typename executor_type::executor_parameters_type;
        using policy_type =
            hpx::execution::detail::parallel_policy_shim<executor_type,
                executor_parameters_type>;
        using base_type = detail::policy_allocator<T, policy_type>;
        using target_type = std::vector<host::target>;

        block_allocator()
          : base_type(policy_type(
                executor_type(target_type(1)), executor_parameters_type()))
        {
        }

        block_allocator(target_type const& targets)
          : base_type(
                policy_type(executor_type(targets), executor_parameters_type()))
        {
        }

        block_allocator(target_type&& targets)
          : base_type(policy_type(
                executor_type(HPX_MOVE(targets)), executor_parameters_type()))
        {
        }

        // Access the underlying target (device)
        target_type const& target() const noexcept
        {
            return this->policy().executor().targets();
        }
    };
}}}    // namespace hpx::compute::host
