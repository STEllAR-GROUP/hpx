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
#include <hpx/runtime/threads/executors/thread_pool_attached_executors.hpp>
#include <hpx/util/functional/new.hpp>

#include <boost/range/iterator_range_core.hpp>

#include <vector>

namespace hpx { namespace compute { namespace host
{
    /// The block_allocator allocates blocks of memory evenly divided onto the
    /// passed vector of targets. This is done by using first touch memory
    /// placement. (maybe better methods will be used in the future...);
    ///
    /// This allocator can be used to write numa aware algorithms:
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

        block_allocator(block_allocator&& alloc)
          : executor_(std::move(alloc.executor_))
        {}

        template <typename U>
        block_allocator(block_allocator<U> const& alloc)
          : executor_(alloc.executor_)
        {}

        template <typename U>
        block_allocator(block_allocator<U>&& alloc)
          : executor_(std::move(alloc.executor_))
        {}

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
        // ::operator new. The pointer hint may be used to provide locality of
        // reference: the allocator, if supported by the implementation, will
        // attempt to allocate the new memory block as close as possible to hint.
        pointer allocate(size_type n, std::allocator<void>::const_pointer hint = 0)
        {
            return static_cast<pointer>(::operator new(n * sizeof(T)));
        }

        // Deallocates the storage referenced by the pointer p, which must be a
        // pointer obtained by an earlier call to allocate(). The argument n
        // must be equal to the first argument of the call to allocate() that
        // originally produced p; otherwise, the behavior is undefined.
        void deallocate(pointer p, size_type n)
        {
            delete p;
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
            // FIXME: Handle exceptions thrown from constructors

            // first touch policy, distribute evenly onto targets
            auto irange = boost::irange(std::size_t(0), count);
            hpx::parallel::for_each(
                hpx::parallel::par
                    .on(executor_)
                    .with(hpx::parallel::static_chunk_size()),
                boost::begin(irange), boost::end(irange),
                [p, args...](std::size_t i)
                {
                    ::new (p + i) U (std::move(args)...);
                }
            );
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
