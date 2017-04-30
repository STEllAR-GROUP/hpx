//  Copyright (c) 2014-2016 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_RMA_ALLOCATOR
#define HPX_PARCELSET_POLICIES_RMA_ALLOCATOR

#include <hpx/config/parcelport_defines.hpp>
//
#include <hpx/runtime/parcelset/rma/detail/memory_region_impl.hpp>
#include <hpx/runtime/parcelset/rma/detail/memory_region_allocator.hpp>
#include <hpx/runtime/parcelset/rma/memory_pool.hpp>

namespace hpx {
namespace parcelset {
namespace rma {

    // ---------------------------------------------------------------------------
    // This is an abstract allocator that must be overridden by parcelports
    // in order to provide access to registered memory by STL like algorithms.
    // The allocator must use virtual functions because the memory pool is templated
    // over the region provider type and cannot
    //
    // Note that native HPX code does not need to use these allocators and can
    // instead get memory regions directly from the parcelport
    // ---------------------------------------------------------------------------
    template <typename T>
    struct rma_allocator
    {
        using size_type         = std::size_t;
        using difference_type   = std::ptrdiff_t;
        using pointer           = T*;
        using const_pointer     = T const*;
        using reference         = T&;
        using const_reference   = T const&;
        using value_type        = T;

        // STL allocators need a default constructor
        rma_allocator() : pool_(nullptr) {}
        //
        rma_allocator(memory_pool_base *pool) : pool_(pool) {}
        //
        rma_allocator(const rma_allocator<T> &other) : pool_(other.pool_) {}

        // polymorphic destructor
        virtual ~rma_allocator() {}

        template <typename U>
        struct rebind {
            using other = rma_allocator<U>;
        };

        virtual T* allocate(std::size_t const n)
        {
            return static_cast<T*>(pool_->allocate(n));
        }

        virtual void deallocate(T* const p, std::size_t const n)
        {
            pool_->deallocate(p, n);
        }

        template <class U, class ...A>
        void construct(U* const p, A&& ...args)
        {
          new (p) U(std::forward<A>(args)...);
        }

        template <class U>
        void destroy(U* const p)
        {
          p->~U();
        }

        memory_pool_base *get_memory_pool() {
            return pool_;
        }

    protected:
        memory_pool_base *pool_;
    };

    namespace detail
    {
        // ---------------------------------------------------------------------------
        template <typename T, typename RegionProvider>
        struct rma_allocator_impl : rma_allocator<T>
        {
            typedef rma::memory_pool<RegionProvider> memory_pool_type;
            //
            rma_allocator_impl(memory_pool_type *pool) : rma_allocator<T>(pool) {}
        };
    }

}}}

#endif
