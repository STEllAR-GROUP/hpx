//  Copyright (c) 2014-2016 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_VERBS_MEMORY_POOL
#define HPX_PARCELSET_POLICIES_VERBS_MEMORY_POOL

#include <hpx/lcos/local/mutex.hpp>
#include <hpx/lcos/local/spinlock.hpp>
//
#include <atomic>
#include <stack>
#include <unordered_map>
#include <iostream>
#include <cstddef>
#include <memory>
//
#include <boost/lockfree/stack.hpp>
//
#include <hpx/config/parcelport_verbs_defines.hpp>
//
#include <plugins/parcelport/verbs/rdma/rdma_logging.hpp>
#include <plugins/parcelport/verbs/rdma/rdma_locks.hpp>
#include <plugins/parcelport/verbs/rdma/rdma_chunk_pool.hpp>
#include <plugins/parcelport/verbs/rdma/verbs_protection_domain.hpp>
#include <plugins/parcelport/verbs/rdma/verbs_memory_region.hpp>

// the default memory chunk size in bytes
#define RDMA_POOL_1K_CHUNK          0x001*0x0400 //  1KB
#define RDMA_POOL_SMALL_CHUNK_SIZE  0x010*0x0400 // 16KB
#define RDMA_POOL_MEDIUM_CHUNK_SIZE 0x040*0x0400 // 64KB
#define RDMA_POOL_LARGE_CHUNK_SIZE  0x400*0x0400 //  1MB

#define RDMA_POOL_MAX_1K_CHUNKS     1024
#define RDMA_POOL_MAX_SMALL_CHUNKS  1024
#define RDMA_POOL_MAX_MEDIUM_CHUNKS 64
#define RDMA_POOL_MAX_LARGE_CHUNKS  32

#define RDMA_POOL_USE_LOCKFREE_STACK 1

// if the HPX configuration has set a different value, use it
#if defined(HPX_PARCELPORT_VERBS_MEMORY_CHUNK_SIZE)
# undef RDMA_POOL_SMALL_CHUNK_SIZE
# define RDMA_POOL_SMALL_CHUNK_SIZE HPX_PARCELPORT_VERBS_MEMORY_CHUNK_SIZE
#endif

static_assert ( HPX_PARCELPORT_VERBS_MEMORY_CHUNK_SIZE<RDMA_POOL_MEDIUM_CHUNK_SIZE ,
"Default memory Chunk size must be less than medium chunk size" );


// Description of memory pool objects:
//
// memory_region_allocator:
// An allocator that returns memory of the requested size. The memory is pinned
// and ready to be used for RDMA operations. A memory_region object is
// used, it contains the memory registration information needed by the verbs API.
//
// rdma_chunk_pool :
// This is a class taken from boost that takes a block of memory (in this case provided
// by the memory_region_allocator) and divides it up into N smaller blocks.
// These smaller blocks can be used for individual objects or can be used as buffers.
// If 16 blocks of 1K are requested, it will call the allocator and request
// 16*1K + 16 bytes. The overhead per memory allocation request is 16 bytes
//
// pool_container:
// The pool container wraps an rdma_chunk_pool and provides a stack. When a user
// requests a small block, one is popped off the stack. At startup, the pool_container
// requests a large number of blocks from the rdma_chunk_pool and sets the correct
// address offset within each larger chunk for each small block and pushes the mini
// verbs_memory_region onto the stack. Thus N small rdma_regions are created from a
// single larger one and memory blocks come from contiguous memory.
//
// rdma_memory_pool:
// The rdma_memory_pool maintains 4 pool_container (stacks) of different sized blocks
// so that most user requests can be fulfilled.
// If a request cannot be filled, the pool can generate temporary blocks with
// new allocations and on-the-fly registration of the memory.
// Additionally, it also provides a simple API so users may pass pre-allocated
// memory to the pool for on-the-fly registration (rdma transfer of user memory chunks)
// and later de-registration.

namespace hpx {
namespace parcelset {
namespace policies {
namespace verbs
{
    struct rdma_memory_pool;
}}}}

namespace hpx {
namespace parcelset {
namespace policies {
namespace verbs
{
    // A simple tag type we use for logging assistance (identification)
    struct pool_tiny   { static const char *desc() { return "Tiny ";   } };
    struct pool_small  { static const char *desc() { return "Small ";  } };
    struct pool_medium { static const char *desc() { return "Medium "; } };
    struct pool_large  { static const char *desc() { return "Large ";  } };

    // ---------------------------------------------------------------------------
    // pool_container, collect some routines for reuse with
    // small, medium, large chunks
    // ---------------------------------------------------------------------------
    template <typename pool_chunk_allocator, typename PoolType>
    struct pool_container
    {
#ifndef RDMA_POOL_USE_LOCKFREE_STACK
        typedef hpx::lcos::local::spinlock                               mutex_type;
        typedef hpx::parcelset::policies::verbs::scoped_lock<mutex_type> scoped_lock;
#endif

        // ------------------------------------------------------------------------
        pool_container(verbs_protection_domain_ptr pd, std::size_t chunk_size,
            std::size_t chunks_per_block, std::size_t max_items) :
                chunk_size_(chunk_size), max_chunks_(max_items), used_(0),
                chunk_allocator(pd, chunk_size, chunks_per_block, chunks_per_block)
        {
            LOG_DEBUG_MSG(PoolType::desc() << "Creating with chunk_size "
                << hexnumber(chunk_size_) << "max_chunks " << max_chunks_);
        }

        // ------------------------------------------------------------------------
        bool allocate_pool(std::size_t _num_chunks)
        {
            LOG_DEBUG_MSG(PoolType::desc() << "Allocating " << std::dec << _num_chunks
                << " blocks of " << hexlength(chunk_size_));
            //
            for (std::size_t i=0; i<_num_chunks; i++) {
                LOG_TRACE_MSG(PoolType::desc() << "Allocate Block "
                    << i << " of size " << hexlength(chunk_size_));
                verbs_memory_region region = chunk_allocator.malloc();
                if (region.get_address()!=nullptr) {
                    block_list_[region.get_address()] = region;
                    // we use the pointer to the region
                    verbs_memory_region *r = &block_list_[region.get_address()];
                    push(r);
                }
                else {
                    LOG_ERROR_MSG(PoolType::desc()
                        << "Block Allocation Stopped at " << (i-1));
                    return false;
                }
            }
            used_ = 0;
            return true;
        }

        // ------------------------------------------------------------------------
        int DeallocatePool()
        {
            if (used_!=0) {
                LOG_ERROR_MSG(PoolType::desc()
                    << "Deallocating free_list : Not all blocks were returned "
                    << " refcounts " << decnumber(used_));
            }
            while (!free_list_.empty()) {
                chunk_allocator.free(*pop());
            }
            block_list_.clear();
            chunk_allocator.release_memory();
            return 1;
        }

        // ------------------------------------------------------------------------
        inline void push(verbs_memory_region *region)
        {
#ifndef RDMA_POOL_USE_LOCKFREE_STACK
            scoped_lock lock(memBuffer_mutex_);
#endif
            LOG_TRACE_MSG(PoolType::desc() << "Push block "
                << hexpointer(region->get_address()) << hexlength(region->get_size())
                << decnumber(used_-1));

#ifdef RDMA_POOL_USE_LOCKFREE_STACK
            if (!free_list_.push(region)) {
                LOG_ERROR_MSG(PoolType::desc() << "Error in memory pool push");
            }
#else
            free_list_.push(region);
#endif
            // decrement one reference
            used_--;
        }

        // ------------------------------------------------------------------------
        inline verbs_memory_region *pop()
        {
#ifndef RDMA_POOL_USE_LOCKFREE_STACK
            scoped_lock lock(memBuffer_mutex_);
#endif
            // if we have not exceeded our max size, allocate a new block
            if (free_list_.empty()) {
                //  LOG_TRACE_MSG("Creating new small Block as free list is empty "
                // "but max chunks " << max_small_chunks_ << " not reached");
                //  AllocateRegisteredBlock(length);
                //std::terminate();
                return nullptr;
            }
#ifdef RDMA_POOL_USE_LOCKFREE_STACK
            // get a block
            verbs_memory_region *region = nullptr;
            if (!free_list_.pop(region)) {
                LOG_DEBUG_MSG(PoolType::desc() << "Error in memory pool pop");
            }
#else
            verbs_memory_region *region = free_list_.top();
            free_list_.pop();
#endif
            // Keep reference counts to self so that we can check
            // this pool is not deleted whilst blocks still exist
            used_++;
            LOG_TRACE_MSG(PoolType::desc() << "Pop block "
                << hexpointer(region->get_address()) << hexlength(region->get_size())
                << decnumber(used_));
            //
            return region;
        }

        //
        std::size_t                                 chunk_size_;
        std::size_t                                 max_chunks_;
        std::atomic<int>                            used_;
#ifdef RDMA_POOL_USE_LOCKFREE_STACK
        boost::lockfree::stack<verbs_memory_region*,
            boost::lockfree::capacity<8192>> free_list_;
#else
        std::stack<verbs_memory_region*> free_list_;
        mutex_type                      memBuffer_mutex_;
#endif
        //
        pool_chunk_allocator                           chunk_allocator;
        std::unordered_map<char *, verbs_memory_region> block_list_;
};

    // ---------------------------------------------------------------------------
    // memory pool, holds 4 smaller pools and pops/pushes to the one
    // of the right size for the requested data
    // ---------------------------------------------------------------------------
    struct rdma_memory_pool
    {
        HPX_NON_COPYABLE(rdma_memory_pool);

        //----------------------------------------------------------------------------
        // constructor
        rdma_memory_pool(verbs_protection_domain_ptr pd) :
            protection_domain_(pd),
            tiny_  (pd, RDMA_POOL_1K_CHUNK,         1024, RDMA_POOL_MAX_1K_CHUNKS),
            small_ (pd, RDMA_POOL_SMALL_CHUNK_SIZE, 1024, RDMA_POOL_MAX_SMALL_CHUNKS),
            medium_(pd, RDMA_POOL_MEDIUM_CHUNK_SIZE,  64, RDMA_POOL_MAX_MEDIUM_CHUNKS),
            large_ (pd, RDMA_POOL_LARGE_CHUNK_SIZE,   32, RDMA_POOL_MAX_LARGE_CHUNKS),
            temp_regions(0),
            user_regions(0)
        {
            tiny_.allocate_pool(RDMA_POOL_MAX_1K_CHUNKS);
            small_.allocate_pool(RDMA_POOL_MAX_SMALL_CHUNKS);
            medium_.allocate_pool(RDMA_POOL_MAX_MEDIUM_CHUNKS);
            large_.allocate_pool(RDMA_POOL_MAX_LARGE_CHUNKS);
            LOG_DEBUG_MSG("Completed memory_pool initialization");
        }

        //----------------------------------------------------------------------------
        // destructor
        ~rdma_memory_pool()
        {
            deallocate_pools();
        }

        //----------------------------------------------------------------------------
        int deallocate_pools()
        {
            bool ok = true;
            ok = ok && tiny_.DeallocatePool();
            ok = ok && small_.DeallocatePool();
            ok = ok && medium_.DeallocatePool();
            ok = ok && large_.DeallocatePool();
            return ok;
        }

        // -------------------------
        // User allocation interface
        // -------------------------
        // The verbs_memory_region* versions of allocate/deallocate
        // should be used in preference to the std:: compatible
        // versions using char* for efficiency

        //----------------------------------------------------------------------------
        // query the pool for a chunk of a given size to see if one is available
        // this function is 'unsafe' because it is not thread safe and another
        // thread may push/pop a block after this is called and invalidate the result.
        inline bool can_allocate_unsafe(size_t length) const
        {
            if (length<=tiny_.chunk_size_) {
                return !tiny_.free_list_.empty();
            }
            else if (length<=small_.chunk_size_) {
                return !small_.free_list_.empty();
            }
            else if (length<=medium_.chunk_size_) {
                return !medium_.free_list_.empty();
            }
            else if (length<=large_.chunk_size_) {
                return !large_.free_list_.empty();
            }
            return true;
        }

        //----------------------------------------------------------------------------
        // allocate a region, if size=0 a tiny region is returned
        inline verbs_memory_region *allocate_region(size_t length)
        {
            verbs_memory_region *region = nullptr;
            //
            if (length<=tiny_.chunk_size_) {
                region = tiny_.pop();
            }
            else if (length<=small_.chunk_size_) {
                region = small_.pop();
            }
            else if (length<=medium_.chunk_size_) {
                region = medium_.pop();
            }
            else if (length<=large_.chunk_size_) {
                region = large_.pop();
            }
            // if we didn't get a block from the cache, create one on the fly
            if (region==nullptr) {
                region = allocate_temporary_region(length);
            }

            LOG_TRACE_MSG("Popping Block"
                << " buffer "    << hexpointer(region->get_address())
                << " region "    << hexpointer(region)
                << " size "      << hexlength(region->get_size())
                << " chunksize " << hexlength(small_.chunk_size_) << " "
                << hexlength(medium_.chunk_size_) << " " << hexlength(large_.chunk_size_)
                << " free (t) "  << (RDMA_POOL_MAX_1K_CHUNKS-tiny_.used_)
                << " used "      << decnumber(this->small_.used_)
                << " free (s) "  << (RDMA_POOL_MAX_SMALL_CHUNKS-small_.used_)
                << " used "      << decnumber(this->small_.used_)
                << " free (m) "  << (RDMA_POOL_MAX_MEDIUM_CHUNKS-medium_.used_)
                << " used "      << decnumber(this->medium_.used_)
                << " free (l) "  << (RDMA_POOL_MAX_LARGE_CHUNKS-large_.used_)
                << " used "      << decnumber(this->large_.used_));
            //
            return region;
        }

        //----------------------------------------------------------------------------
        // release a region back to the pool
        inline void deallocate(verbs_memory_region *region)
        {
            // if this region was registered on the fly, then don't return it to the pool
            if (region->get_temp_region() || region->get_user_region()) {
                if (region->get_temp_region()) {
                    temp_regions--;
                    LOG_TRACE_MSG("Deallocating temp registered block "
                        << hexpointer(region->get_address()) << decnumber(temp_regions));
                }
                else if (region->get_user_region()) {
                    user_regions--;
                    LOG_TRACE_MSG("Deleting (user region) "
                        << hexpointer(region->get_address()) << decnumber(user_regions));
                }
                delete region;
                return;
            }

            // put the block back on the free list
            if (region->get_size()<=tiny_.chunk_size_) {
                tiny_.push(region);
            }
            else if (region->get_size()<=small_.chunk_size_) {
                small_.push(region);
            }
            else if (region->get_size()<=medium_.chunk_size_) {
                medium_.push(region);
            }
            else if (region->get_size()<=large_.chunk_size_) {
                large_.push(region);
            }

            LOG_TRACE_MSG("Pushing Block"
                << " buffer "    << hexpointer(region->get_address())
                << " region "    << hexpointer(region)
                << " free (t) "  << (RDMA_POOL_MAX_1K_CHUNKS-tiny_.used_)
                << " used "      << decnumber(this->small_.used_)
                << " free (s) "  << (RDMA_POOL_MAX_SMALL_CHUNKS-small_.used_)
                << " used "      << decnumber(this->small_.used_)
                << " free (m) "  << (RDMA_POOL_MAX_MEDIUM_CHUNKS-medium_.used_)
                << " used "      << decnumber(this->medium_.used_)
                << " free (l) "  << (RDMA_POOL_MAX_LARGE_CHUNKS-large_.used_)
                << " used "      << decnumber(this->large_.used_));
        }

        //----------------------------------------------------------------------------
        // allocates a region from the heap and registers it, it bypasses the pool
        // when deallocted, it will be unregistered and deleted, not returned to the pool
        inline verbs_memory_region* allocate_temporary_region(std::size_t length)
        {
            verbs_memory_region *region = new verbs_memory_region();
            region->set_temp_region();
            region->allocate(protection_domain_, length);
            temp_regions++;
            LOG_TRACE_MSG("Allocating temp registered block "
                << hexpointer(region->get_address()) << hexlength(length)
                << decnumber(temp_regions));
            return region;
        }
/*
        //----------------------------------------------------------------------------
        // allocate a region, returning a memory block address
        // this is compatible with STL like allocators but should be avoided
        // as deallocation requires a map lookup of the address to find it's block
        char *allocate(size_t length)
        {
            verbs_memory_region *region = allocate_region(length);
            return region->get_address();
        }

        //----------------------------------------------------------------------------
        // deallocate a region using its memory address as handle
        // this involves a map lookup to find the region and is therefore
        // less efficient than releasing memory via the region pointer
        void deallocate(void *address, size_t size=0)
        {
            verbs_memory_region *region = pointer_map_[address];
            deallocate(region);
        }

        //----------------------------------------------------------------------------
        // find an verbs_memory_region* from the memory address it wraps
        verbs_memory_region *RegionFromAddress(char * const addr) {
            return pointer_map_[addr];
        }

        //----------------------------------------------------------------------------
        // internal variables
        //----------------------------------------------------------------------------
        // used to map the internal memory address to the region that
        // holds the registration information
        std::unordered_map<const void *, verbs_memory_region*> pointer_map_;
*/
        // protection domain that memory is registered with
        verbs_protection_domain_ptr protection_domain_;

        // maintain 4 pools of thread safe pre-allocated regions of fixed size.
        // they obtain their memory from the segmented storage provided
        pool_container<rdma_chunk_pool<memory_region_allocator>, pool_tiny> tiny_;
        pool_container<rdma_chunk_pool<memory_region_allocator>, pool_small> small_;
        pool_container<rdma_chunk_pool<memory_region_allocator>, pool_medium> medium_;
        pool_container<rdma_chunk_pool<memory_region_allocator>, pool_large> large_;
        //
        // a counter
        std::atomic<int> temp_regions;
        std::atomic<int> user_regions;
    };

    typedef std::shared_ptr<rdma_memory_pool> rdma_memory_pool_ptr;
}}}}

#endif
