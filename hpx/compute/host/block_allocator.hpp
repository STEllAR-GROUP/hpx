///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#ifndef HPX_COMPUTE_HOST_BLOCK_ALLOCATOR_HPP
#define HPX_COMPUTE_HOST_BLOCK_ALLOCATOR_HPP

#include <hpx/config.hpp>

#include <hpx/compute/host/block_executor.hpp>
#include <hpx/compute/host/target.hpp>
#include <hpx/parallel/algorithms/for_each.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/executors/static_chunk_size.hpp>
#include <hpx/parallel/util/cancellation_token.hpp>
#include <hpx/parallel/util/partitioner_with_cleanup.hpp>
#include <hpx/runtime/threads/executors/thread_pool_attached_executors.hpp>
#include <hpx/runtime/threads/topology.hpp>
#include <hpx/util/functional/new.hpp>
#include <hpx/util/invoke_fused.hpp>
#include <hpx/util/tuple.hpp>

#include <boost/range/iterator_range_core.hpp>

#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace compute { namespace host
{
    /// The block_allocator allocates blocks of memory evenly divided onto the
    /// passed vector of targets. This is done by using first touch memory
    /// placement. (maybe better methods will be used in the future...);
    ///
    /// This allocator can be used to write NUMA aware algorithms:
    ///
    /// typedef hpx::compute::host::block_allocator<int> allocator_type;
    /// typedef hpx::compute::vector<int, allocator_type> vector_type;
    ///
    /// auto numa_nodes = hpx::compute::host::numa_domains();
    /// std::size_t N = 2048;
    /// vector_type v(N, allocator_type(numa_nodes));
    ///
    template <typename T, typename Executor =
        hpx::threads::executors::local_priority_queue_attached_executor>
    struct block_allocator
    {
        typedef T value_type;
        typedef T* pointer;
        typedef const T* const_pointer;
        typedef T& reference;
        typedef T const& const_reference;
        typedef std::size_t size_type;
        typedef std::ptrdiff_t difference_type;

        typedef Executor executor_type;

        template <typename U>
        struct rebind
        {
            typedef block_allocator<U> other;
        };

        typedef std::false_type is_always_equal;
        typedef std::true_type propagate_on_container_move_assignment;

        typedef std::vector<host::target> target_type;

        block_allocator()
          : executor_(target_type(1))
        {}

        block_allocator(target_type const& targets)
          : executor_(targets)
        {}

        block_allocator(target_type && targets)
          : executor_(targets)
        {}

        block_allocator(block_allocator const& alloc)
          : executor_(alloc.executor_)
        {}

        block_allocator(block_allocator && alloc)
          : executor_(std::move(alloc.executor_))
        {}

        template <typename U>
        block_allocator(block_allocator<U> const& alloc)
          : executor_(alloc.executor_)
        {}

        template <typename U>
        block_allocator(block_allocator<U> && alloc)
          : executor_(std::move(alloc.executor_))
        {}

        block_allocator& operator=(block_allocator const& rhs)
        {
            executor_ = rhs.executor_;
            return *this;
        }
        block_allocator& operator=(block_allocator && rhs)
        {
            executor_ = std::move(rhs.executor_);
            return *this;
        }

        // Returns the actual address of x even in presence of overloaded
        // operator&
        pointer address(reference x) const HPX_NOEXCEPT
        {
            return &x;
        }

        const_pointer address(const_reference x) const HPX_NOEXCEPT
        {
            return &x;
        }

        // Allocates n * sizeof(T) bytes of uninitialized storage by calling
        // topo.allocate(). The pointer hint may be used to provide locality of
        // reference: the allocator, if supported by the implementation, will
        // attempt to allocate the new memory block as close as possible to hint.
        pointer allocate(size_type n, std::allocator<void>::const_pointer hint = 0)
        {
            return reinterpret_cast<pointer>(
                hpx::threads::get_topology().allocate(n * sizeof(T)));
        }

        // Deallocates the storage referenced by the pointer p, which must be a
        // pointer obtained by an earlier call to allocate(). The argument n
        // must be equal to the first argument of the call to allocate() that
        // originally produced p; otherwise, the behavior is undefined.
        void deallocate(pointer p, size_type n)
        {
            hpx::threads::get_topology().deallocate(p, n);
        }

        // Returns the maximum theoretically possible value of n, for which the
        // call allocate(n, 0) could succeed. In most implementations, this
        // returns std::numeric_limits<size_type>::max() / sizeof(value_type).
        size_type max_size() const HPX_NOEXCEPT
        {
            return (std::numeric_limits<size_type>::max)();
        }

    public:
        // Constructs count objects of type T in allocated uninitialized
        // storage pointed to by p, using placement-new. This will use the
        // underlying executors to distribute the memory according to
        // first touch memory placement.
        template <typename U, typename ... Args>
        void bulk_construct(U* p, std::size_t count, Args &&... args)
        {
            auto irange = boost::irange(std::size_t(0), count);
            auto policy =
                hpx::parallel::par
                    .on(executor_)
                    .with(hpx::parallel::static_chunk_size());

            typedef boost::range_detail::integer_iterator<std::size_t>
                iterator_type;
            typedef std::pair<iterator_type, iterator_type> partition_result_type;

            typedef parallel::util::partitioner_with_cleanup<
                    decltype(policy), void, partition_result_type
                > partitioner;
            typedef parallel::util::cancellation_token<
                    parallel::util::detail::no_data
                > cancellation_token;

            auto && arguments =
                hpx::util::forward_as_tuple(std::forward<Args>(args)...);

            cancellation_token tok;
            partitioner::call(std::move(policy),
                boost::begin(irange), count,
                [&arguments, p, tok](iterator_type it, std::size_t part_size)
                    mutable -> partition_result_type
                {
                    iterator_type last =
                        parallel::util::loop_with_cleanup_n_with_token(
                            it, part_size, tok,
                            [&arguments, p](iterator_type it)
                            {
                                using hpx::util::functional::placement_new_one;
                                hpx::util::invoke_fused(
                                    placement_new_one<U>(p + *it), arguments);
                            },
                            // cleanup function, called for all elements of
                            // current partition which succeeded before exception
                            [p](iterator_type it)
                            {
                                (p + *it)->~U();
                            });
                    return std::make_pair(it, last);
                },
                // finalize, called once if no error occurred
                [](std::vector<hpx::future<partition_result_type> > &&)
                {
                    // do nothing
                },
                // cleanup function, called for each partition which
                // didn't fail, but only if at least one failed
                [p](partition_result_type && r) -> void
                {
                    while (r.first != r.second)
                    {
                        (p + *r.first)->~U();
                        ++r.first;
                    }
                });
        }

        // Constructs an object of type T in allocated uninitialized storage
        // pointed to by p, using placement-new
        template <typename U, typename ... Args>
        void construct(U* p, Args &&... args)
        {
            executor_.execute(
                hpx::util::functional::placement_new<U>(),
                p, std::forward<Args>(args)...);
        }

        // Calls the destructor of count objects pointed to by p
        template <typename U>
        void bulk_destroy(U* p, std::size_t count)
        {
            // keep memory locality, use executor...
            auto irange = boost::irange(std::size_t(0), count);
            hpx::parallel::for_each(
                hpx::parallel::par
                    .on(executor_)
                    .with(hpx::parallel::static_chunk_size()),
                boost::begin(irange), boost::end(irange),
                [p](std::size_t i)
                {
                    (p + i)->~U();
                }
            );
        }

        // Calls the destructor of the object pointed to by p
        template <typename U>
        void destroy(U* p)
        {
            p->~U();
        }

        // Access the underlying target (device)
        target_type const& target() const HPX_NOEXCEPT
        {
            return executor_.targets();
        }

    private:
        block_executor<executor_type> executor_;
    };
}}}

#endif
