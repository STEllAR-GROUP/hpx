//  Copyright (c) 2015 Thomas Heller
//  Copyright (c) 2015-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_NUMA_ALLOCATOR_HPP
#define HPX_UTIL_NUMA_ALLOCATOR_HPP

#include <hpx/config.hpp>
#include <hpx/assertion.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/wait_all.hpp>
#include <hpx/parallel/algorithms/for_each.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/executors/execution_information.hpp>
#include <hpx/parallel/executors/static_chunk_size.hpp>
#include <hpx/runtime/get_worker_thread_num.hpp>
#include <hpx/runtime/threads/topology.hpp>

#include <cstddef>
#include <limits>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
//
// WARNING: This class is highly experimental, it might not do what you expect
//
///////////////////////////////////////////////////////////////////////////////

namespace hpx { namespace parallel { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Executors>
    class numa_allocator
    {
        typedef typename Executors::value_type executor_type;
    public:
        // typedefs
        typedef T value_type;
        typedef value_type* pointer;
        typedef value_type const* const_pointer;
        typedef value_type& reference;
        typedef value_type const& const_reference;
        typedef std::size_t size_type;
        typedef std::ptrdiff_t difference_type;

    public:
        // convert an allocator<T> to allocator<U>
        template <typename U>
        struct rebind
        {
            typedef numa_allocator<U, Executors> other;
        };

    public:
        numa_allocator(Executors const& executors, hpx::threads::topology& topo)
          : executors_(executors), topo_(topo)
        {}

        numa_allocator(numa_allocator const& rhs)
          : executors_(rhs.executors_), topo_(rhs.topo_)
        {}

        template <typename U>
        numa_allocator(numa_allocator<U, Executors> const& rhs)
          : executors_(rhs.executors_), topo_(rhs.topo_)
        {}

        // address
        pointer address(reference r) { return &r; }
        const_pointer address(const_reference r) { return &r; }

        // memory allocation
        pointer allocate(size_type cnt,
            typename std::allocator<void>::const_pointer = nullptr)
        {
            // allocate memory
            pointer p = reinterpret_cast<pointer>(topo_.allocate(cnt * sizeof(T)));

            // first touch policy, distribute evenly onto executors
            std::size_t part_size = cnt / executors_.size();
            std::vector<hpx::future<void> > first_touch;
            first_touch.reserve(executors_.size());

            for (std::size_t i = 0; i != executors_.size(); ++i)
            {
                pointer begin = p + i * part_size;
                pointer end = begin + part_size;
                first_touch.push_back(
                    hpx::parallel::for_each(
                        hpx::parallel::execution::par(hpx::parallel::execution::task)
                            .on(executors_[i])
                            .with(hpx::parallel::execution::static_chunk_size()),
                        begin, end,
#if defined(HPX_DEBUG)
                        [this, i]
#else
                        []
#endif
                        (T& val)
                        {
                            // touch first byte of every object
                            *reinterpret_cast<char*>(&val) = 0;

#if defined(HPX_DEBUG)
                            // make sure memory was placed appropriately
                            hpx::threads::mask_type mem_mask =
                                topo_.get_thread_affinity_mask_from_lva(
                                    reinterpret_cast<hpx::naming::address_type>(&val));

                            std::size_t thread_num = hpx::get_worker_thread_num();
                            hpx::threads::mask_cref_type thread_mask =
                                hpx::parallel::execution::get_pu_mask(
                                    executors_[i], topo_, thread_num);

                            HPX_ASSERT(threads::mask_size(mem_mask) ==
                                threads::mask_size(thread_mask));
                            HPX_ASSERT(threads::bit_and(mem_mask, thread_mask,
                                threads::mask_size(mem_mask)));
#endif
                        })
                );
            }
            hpx::wait_all(first_touch);

            for (auto && f : first_touch)
            {
                f.get();        // rethrow exceptions
            }

            // return the overall memory block
            return p;
        }

        void deallocate(pointer p, size_type cnt)
        {
            topo_.deallocate(p, cnt * sizeof(T));
        }

        // size
        size_type max_size() const
        {
            return (std::numeric_limits<size_type>::max)() / sizeof(T);
        }

        // construction/destruction
        void construct(pointer p, const T& t) { new(p) T(t); }
        void destroy(pointer p) { p->~T(); }

        friend bool operator==(numa_allocator const&, numa_allocator const&)
        {
            return true;
        }

        friend bool operator!=(numa_allocator const& l, numa_allocator const& r)
        {
            return !(l == r);
        }

    private:
        template <typename, typename>
        friend class numa_allocator;

        Executors const& executors_;
        hpx::threads::topology& topo_;
    };
}}}

#endif
