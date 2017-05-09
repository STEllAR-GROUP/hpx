//  Copyright (c) 2014-2016 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_RMA_ALLOCATOR
#define HPX_PARCELSET_POLICIES_RMA_ALLOCATOR

#include <hpx/config/parcelport_defines.hpp>
//
#include <plugins/parcelport/parcelport_logging.hpp>
#include <hpx/runtime/parcelset/rma/detail/memory_region_impl.hpp>
#include <hpx/runtime/parcelset/rma/detail/memory_region_allocator.hpp>
#include <hpx/runtime/parcelset/rma/memory_pool.hpp>

namespace hpx {
namespace parcelset {
namespace rma {

    namespace detail
    {
        // ---------------------------------------------------------------------------
        // to avoid including the whole runtime we use a helper function to init
        // the allocator
        HPX_EXPORT memory_pool_base *get_parcelport_allocator_pool();
    }

    // ---------------------------------------------------------------------------
    // This is an abstract allocator that must be overridden by parcelports
    // in order to provide access to registered memory by STL-like algorithms.
    // The allocator must use virtual functions because the memory pool is templated
    // over the region provider type - we cannot have templated virtual methods so
    // an abstract getter is used to a type erased memory region.
    //
    // Note that native HPX code does not need to use these allocators and can
    // instead get memory regions directly from the parcelport.
    // The parcelport uses a more optimized, non virtual interface.
    // ---------------------------------------------------------------------------
    template <typename T>
    struct allocator
    {
        using value_type = T;

        // STL allocators need a default constructor
        allocator() noexcept
            : pool_(detail::get_parcelport_allocator_pool())
        {
        }

        allocator(memory_pool_base *pool) noexcept
            : pool_(pool)
        {
        }

        template <class U> allocator(allocator<U> const &other) noexcept
            : pool_(other.pool_)
        {
            LOG_DEVEL_MSG("Copy constructor for allocator "
                << hexpointer(pool_));
        }

        // polymorphic destructor
        virtual ~allocator() {}

        // @TODO: remove virtaul interface
        // virtual allocate so that a concrete instance
        virtual value_type* allocate(std::size_t const n)
        {
            return static_cast<value_type*>(pool_->allocate(n*sizeof(value_type)));
        }

        virtual void deallocate(value_type* const p, std::size_t const n)
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

        // ---------------------------------------------------------------------------
        // access the underlying pool from where allocations are made
        memory_pool_base *get_memory_pool() {
            return pool_;
        }

        // ---------------------------------------------------------------------------
        // retrieve a memory region handle from the address (presumed to have been
        // supplied by the allocator previously)
        memory_region *get_memory_region(void const * addr) const {
            return pool_->region_from_address(addr);
        }

    protected:

        // this function must be used to initialize the allocator
        void setup_memory_pool();

        memory_pool_base *pool_;
    };

    namespace detail
    {
        // ---------------------------------------------------------------------------
        template <typename T, typename RegionProvider>
        struct allocator_impl : allocator<T>
        {
            typedef rma::memory_pool<RegionProvider> memory_pool_type;
            //
            allocator_impl(memory_pool_type *pool) : allocator<T>(pool) {}
        };
    }

}}}

#endif
