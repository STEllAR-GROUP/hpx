//  Copyright (c) 2014-2017 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_RMA_MEMORY_POOL
#define HPX_PARCELSET_POLICIES_RMA_MEMORY_POOL

#include <hpx/lcos/local/mutex.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/util/atomic_count.hpp>
//
#include <hpx/config/parcelport_defines.hpp>
//
#include <hpx/runtime/parcelset/rma/detail/memory_region_impl.hpp>
#include <hpx/runtime/parcelset/rma/detail/memory_region_allocator.hpp>
#include <hpx/runtime/parcelset/rma/detail/memory_pool_stack.hpp>
#include <plugins/parcelport/parcelport_logging.hpp>
#include <plugins/parcelport/performance_counter.hpp>
#include <plugins/parcelport/unordered_map.hpp>
//
#include <boost/lockfree/stack.hpp>
//
#include <atomic>
#include <stack>
#include <unordered_map>
#include <iostream>
#include <cstddef>
#include <memory>
#include <array>
#include <sstream>
#include <string>

// the default memory chunk size in bytes
#define RDMA_POOL_1K_CHUNK_SIZE     0x001*0x0400 //  1KB
#define RDMA_POOL_SMALL_CHUNK_SIZE  0x010*0x0400 // 16KB
#define RDMA_POOL_MEDIUM_CHUNK_SIZE 0x040*0x0400 // 64KB
#define RDMA_POOL_LARGE_CHUNK_SIZE  0x400*0x0400 //  1MB

#define RDMA_POOL_MAX_1K_CHUNKS     1024
#define RDMA_POOL_MAX_SMALL_CHUNKS  2048
#define RDMA_POOL_MAX_MEDIUM_CHUNKS 128
#define RDMA_POOL_MAX_LARGE_CHUNKS  16

// if the HPX configuration has set a different value, use it
#if defined(HPX_PARCELPORT_LIBFABRIC_MEMORY_CHUNK_SIZE)
# undef RDMA_POOL_SMALL_CHUNK_SIZE
# define RDMA_POOL_SMALL_CHUNK_SIZE HPX_PARCELPORT_LIBFABRIC_MEMORY_CHUNK_SIZE
#endif

static_assert ( HPX_PARCELPORT_LIBFABRIC_MEMORY_CHUNK_SIZE<RDMA_POOL_MEDIUM_CHUNK_SIZE ,
"Default memory Chunk size must be less than medium chunk size" );

namespace hpx {
namespace parcelset {
namespace rma
{

    //----------------------------------------------------------------------------
    // memory pool base class we need so that STL compatible allocate/deallocate
    // routines can be piggybacked onto our registered memory pool API using an
    // abstract allocator interface.
    // For performance reasons - The only member functions that are declared virtual
    // are the STL compatible ones that are only used by the rma_object API
    //----------------------------------------------------------------------------
    struct memory_pool_base
    {
        virtual ~memory_pool_base() {}

        //----------------------------------------------------------------------------
        virtual char *allocate(size_t length) = 0;

        //----------------------------------------------------------------------------
        virtual void deallocate(void *address, size_t size) = 0;

        //----------------------------------------------------------------------------
        virtual memory_region *region_from_address(void const * addr) = 0;

        //----------------------------------------------------------------------------
        // release a region back to the pool
        virtual void release_region(memory_region *region) = 0;

        //----------------------------------------------------------------------------
        memory_region *get_region(size_t length)
        {
            return region_from_address(allocate(length));
        }
    };

    // ---------------------------------------------------------------------------
    // The memory pool manages a collection of memory stacks, each one of which
    // contains blocks of memory of a fixed size. The memory pool holds 4
    // stacks and gives them out in response to allocation requests.
    // Individual blocks are pushed/popped to the stack of the right size
    // for the requested data
    // ---------------------------------------------------------------------------
    template <typename RegionProvider>
    struct memory_pool : memory_pool_base
    {
        HPX_NON_COPYABLE(memory_pool);

        typedef typename RegionProvider::provider_domain        domain_type;
        typedef detail::memory_region_impl<RegionProvider>      region_type;
        typedef detail::memory_region_allocator<RegionProvider> allocator_type;
        typedef std::shared_ptr<region_type>                    region_ptr;

        //----------------------------------------------------------------------------
        // constructor
        memory_pool(domain_type *pd) :
            protection_domain_(pd),
            tiny_  (pd),
            small_ (pd),
            medium_(pd),
            large_ (pd),
            temp_regions(0),
            user_regions(0)
        {
            tiny_.allocate_pool();
            small_.allocate_pool();
            medium_.allocate_pool();
            large_.allocate_pool();
            LOG_DEBUG_MSG("Completed memory_pool initialization");
        }

        //----------------------------------------------------------------------------
        // destructor
        ~memory_pool()
        {
            deallocate_pools();
        }

        //----------------------------------------------------------------------------
        void deallocate_pools()
        {
            tiny_.DeallocatePool();
            small_.DeallocatePool();
            medium_.DeallocatePool();
            large_.DeallocatePool();
        }

        //----------------------------------------------------------------------------
        // query the pool for a chunk of a given size to see if one is available
        // this function is 'unsafe' because it is not thread safe and another
        // thread may push/pop a block after/during this call and invalidate the result.
        inline bool can_allocate_unsafe(size_t length) const
        {
            if (length<=tiny_.chunk_size()) {
                return !tiny_.free_list_.empty();
            }
            else if (length<=small_.chunk_size()) {
                return !small_.free_list_.empty();
            }
            else if (length<=medium_.chunk_size()) {
                return !medium_.free_list_.empty();
            }
            else if (length<=large_.chunk_size()) {
                return !large_.free_list_.empty();
            }
            return true;
        }

        //----------------------------------------------------------------------------
        // allocate a region, if size=0 a tiny region is returned
        inline region_type *allocate_region(size_t length)
        {
            region_type *region = nullptr;
            //
            if (length<=tiny_.chunk_size()) {
                region = tiny_.pop();
            }
            else if (length<=small_.chunk_size()) {
                region = small_.pop();
            }
            else if (length<=medium_.chunk_size()) {
                region = medium_.pop();
            }
            else if (length<=large_.chunk_size()) {
                region = large_.pop();
            }
            // if we didn't get a block from the cache, create one on the fly
            if (region==nullptr) {
                region = allocate_temporary_region(length);
            }

            LOG_TRACE_MSG("Popping Block "
                << *region
                << tiny_.status()
                << small_.status()
                << medium_.status()
                << large_.status()
                << large_.status()
                << "temp regions " << decnumber(temp_regions));
            //
            return region;
        }

        //----------------------------------------------------------------------------
        // release a region back to the pool
        inline void deallocate(region_type *region)
        {
            // if this region was registered on the fly, then don't return it to the pool
            if (region->get_temp_region() || region->get_user_region()) {
                if (region->get_temp_region()) {
                    --temp_regions;
                    LOG_TRACE_MSG("Deallocating temp region "
                        << *region
                        << "temp regions " << decnumber(temp_regions));
                }
                else if (region->get_user_region()) {
                    --user_regions;
                    LOG_TRACE_MSG("Deleting (user region) "
                        << *region
                        << "user regions " << decnumber(user_regions));
                }
                delete region;
                return;
            }

            // put the block back on the free list
            if (region->get_size()<=tiny_.chunk_size()) {
                tiny_.push(region);
            }
            else if (region->get_size()<=small_.chunk_size()) {
                small_.push(region);
            }
            else if (region->get_size()<=medium_.chunk_size()) {
                medium_.push(region);
            }
            else if (region->get_size()<=large_.chunk_size()) {
                large_.push(region);
            }

            LOG_TRACE_MSG("Pushing Block "
                << *region
                << tiny_.status()
                << small_.status()
                << medium_.status()
                << large_.status()
                << "temp regions " << decnumber(temp_regions));
        }

        //----------------------------------------------------------------------------
        // allocates a region from the heap and registers it, it bypasses the pool
        // when deallocted, it will be unregistered and deleted, not returned to the pool
        region_type* allocate_temporary_region(std::size_t length)
        {
            region_type *region = new region_type();
            region->set_temp_region();
            region->allocate(protection_domain_, length);
            ++temp_regions;
            LOG_TRACE_MSG("Allocating temp region "
                << *region
                << "temp regions " << decnumber(temp_regions));
            return region;
        }

        //----------------------------------------------------------------------------
        // STL compatible allocate/deallocate
        //----------------------------------------------------------------------------
        // allocate a region, returning a memory block address -
        // this is compatible with STL like allocators but should be avoided
        // if the allocate_region method can be used instead, as allocation requires
        // a map insert and deallocation requires a map lookup of the address to
        // find it's block/region
        char *allocate(size_t length) override
        {
            region_type *region = allocate_region(length);
            region_alloc_pointer_map_.insert(
                std::make_pair(region->get_address(),region));
            return region->get_address();
        }

        //----------------------------------------------------------------------------
        // deallocate a region using its memory address as handle
        // this involves a map lookup to find the region and is therefore
        // less efficient than releasing memory via the region pointer
        void deallocate(void *address, size_t size=0) override
        {
            region_type *region = region_from_address(address);
            deallocate(region);
        }

        void release_region(memory_region *region) {
            deallocate(region);
        }

        //----------------------------------------------------------------------------
        // find an verbs_memory_region* from the memory address it wraps
        // this is only valid for regions allocated sing the STL allocate method
        // and if you are cling this, then you probably should have allocated
        // a region instead in the first place
        inline region_type *region_from_address(void const * addr)
        {
            return region_alloc_pointer_map_[addr];
        }

        //----------------------------------------------------------------------------
        // used to map the internal memory address to the region that
        // holds the registration information
        hpx::concurrent::unordered_map<const void *, region_type*>
            region_alloc_pointer_map_;

        //----------------------------------------------------------------------------
        // protection domain that memory is registered with
        domain_type *protection_domain_;

        // maintain 4 pools of thread safe pre-allocated regions of fixed size.
        detail::memory_pool_stack<RegionProvider, allocator_type, detail::pool_tiny,
            RDMA_POOL_1K_CHUNK_SIZE,         RDMA_POOL_MAX_1K_CHUNKS> tiny_;
        detail::memory_pool_stack<RegionProvider, allocator_type, detail::pool_small,
            RDMA_POOL_SMALL_CHUNK_SIZE,   RDMA_POOL_MAX_SMALL_CHUNKS> small_;
        detail::memory_pool_stack<RegionProvider, allocator_type, detail::pool_medium,
            RDMA_POOL_MEDIUM_CHUNK_SIZE, RDMA_POOL_MAX_MEDIUM_CHUNKS> medium_;
        detail::memory_pool_stack<RegionProvider, allocator_type, detail::pool_large,
            RDMA_POOL_LARGE_CHUNK_SIZE,   RDMA_POOL_MAX_LARGE_CHUNKS> large_;

        // counters
        hpx::util::atomic_count temp_regions;
        hpx::util::atomic_count user_regions;
    };

}}}

#endif
