//  Copyright (c) 2014-2016 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_VERBS_MEMORY_POOL
#define HPX_PARCELSET_POLICIES_VERBS_MEMORY_POOL
//
#include <hpx/lcos/local/mutex.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/traits/is_chunk_allocator.hpp>
#include <hpx/util/memory_chunk_pool_allocator.hpp>
//
#include <atomic>
#include <stack>
#include <unordered_map>
#include <iostream>
//
#include <boost/noncopyable.hpp>
#include <boost/lockfree/stack.hpp>
//
#include <plugins/parcelport/verbs/rdma/rdma_logging.hpp>
#include <plugins/parcelport/verbs/rdma/rdma_locks.hpp>
#include <plugins/parcelport/verbs/rdma/protection_domain.hpp>
#include <plugins/parcelport/verbs/rdma/memory_region.hpp>
#include <plugins/parcelport/verbs/rdma/rdma_chunk_pool.hpp>

// the default memory chunk size in bytes
#define RDMA_POOL_1K_CHUNK          0x000400 //  1KB
#define RDMA_POOL_SMALL_CHUNK_SIZE  0x001000 //  4KB
#define RDMA_POOL_MEDIUM_CHUNK_SIZE 0x004000 // 16KB
#define RDMA_POOL_LARGE_CHUNK_SIZE  0x100000 //  1MB

// the maximum number of preposted receives (pre receive queue)
#define RDMA_MAX_PREPOSTS 256

#define RDMA_POOL_MAX_1K_CHUNKS     1024
#define RDMA_POOL_MAX_SMALL_CHUNKS  1024
#define RDMA_POOL_MAX_MEDIUM_CHUNKS 64
#define RDMA_POOL_MAX_LARGE_CHUNKS  4

#define RDMA_POOL_USE_LOCKFREE_STACK 1
/*
// if the HPX configuration has set a different value, use it
#if defined(HPX_PARCELPORT_VERBS_MEMORY_CHUNK_SIZE)
# undef RDMA_POOL_SMALL_CHUNK_SIZE
# define RDMA_POOL_SMALL_CHUNK_SIZE HPX_PARCELPORT_VERBS_MEMORY_CHUNK_SIZE
#endif

// if the HPX configuration has set a different value, use it
#if defined(HPX_PARCELPORT_VERBS_MAX_PREPOSTS)
# undef RDMA_MAX_PREPOSTS
# define RDMA_MAX_PREPOSTS HPX_PARCELPORT_VERBS_MAX_PREPOSTS
#endif
*/

namespace hpx {
namespace parcelset {
namespace policies {
namespace verbs
{
    struct rdma_memory_pool;
}}}}

// -------------------------
// specialize chunk pool allocator traits for this memory_chunk_pool
// -------------------------
namespace hpx { namespace traits
{
    // if the chunk pool supplies fixed chunks of memory when the alloc
    // is smaller than some threshold, then the pool must declare
    // std::size_t small_chunk_size_
    template <typename T, typename M>
    struct is_chunk_allocator<util::detail::memory_chunk_pool_allocator<T,
        hpx::parcelset::policies::verbs::rdma_memory_pool,M>>
      : std::false_type {};

    template <>
    struct is_chunk_allocator<hpx::parcelset::policies::verbs::rdma_memory_pool>
      : std::false_type {};
}}

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
        pool_container(rdma_protection_domain_ptr pd, std::size_t chunk_size,
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
                rdma_memory_region region = chunk_allocator.malloc();
                if (region.get_address()!=nullptr) {
                    block_list_[region.get_address()] = region;
                    // we use the pointer to the region
                    rdma_memory_region *r = &block_list_[region.get_address()];
                    push(r);
                }
                else {
                    LOG_ERROR_MSG(PoolType::desc() << "Block Allocation Stopped at " << (i-1));
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
        inline void push(rdma_memory_region *region)
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
        inline rdma_memory_region *pop()
        {
#ifndef RDMA_POOL_USE_LOCKFREE_STACK
            scoped_lock lock(memBuffer_mutex_);
#endif
            // if we have not exceeded our max size, allocate a new block
            if (free_list_.empty()) {
                //  LOG_TRACE_MSG("Creating new small Block as free list is empty but max chunks " << max_small_chunks_ << " not reached");
                //  AllocateRegisteredBlock(length);
                //std::terminate();
                return NULL;
            }
#ifdef RDMA_POOL_USE_LOCKFREE_STACK
            // get a block
            rdma_memory_region *region = NULL;
            if (!free_list_.pop(region)) {
                LOG_DEBUG_MSG(PoolType::desc() << "Error in memory pool pop");
            }
#else
            rdma_memory_region *region = free_list_.top();
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
        boost::lockfree::stack<rdma_memory_region*, boost::lockfree::capacity<8192>> free_list_;
#else
        std::stack<rdma_memory_region*> free_list_;
        mutex_type                      memBuffer_mutex_;
#endif
        //
        pool_chunk_allocator                           chunk_allocator;
        std::unordered_map<char *, rdma_memory_region> block_list_;
};

    // ---------------------------------------------------------------------------
    // memory pool, holds 4 smaller pools and pops/pushes to the one
    // of the right size for the requested data
    // ---------------------------------------------------------------------------
    struct rdma_memory_pool : boost::noncopyable
    {
        //----------------------------------------------------------------------------
        // constructor
        rdma_memory_pool(rdma_protection_domain_ptr pd) :
                protection_domain_(pd),
                tiny_  (pd, RDMA_POOL_1K_CHUNK,          256, RDMA_POOL_MAX_1K_CHUNKS),
                small_ (pd, RDMA_POOL_SMALL_CHUNK_SIZE, 8192, RDMA_POOL_MAX_SMALL_CHUNKS),
                medium_(pd, RDMA_POOL_MEDIUM_CHUNK_SIZE,  16, RDMA_POOL_MAX_MEDIUM_CHUNKS),
                large_ (pd, RDMA_POOL_LARGE_CHUNK_SIZE,    4, RDMA_POOL_MAX_LARGE_CHUNKS),
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
            deallocte_pools();
        }

        //----------------------------------------------------------------------------
        int deallocte_pools()
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
        // The rdma_memory_region* versions of allocate/deallocate
        // should be used in preference to the std:: compatible
        // versions using char* for efficiency

        //----------------------------------------------------------------------------
        // query the pool for a chunk of a given size to see if one is available
        // this function is 'unsafe' because it is not thread safe and another
        // thread may pop a block after this is called and invalidate the result.
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
        inline rdma_memory_region *allocate_region(size_t length)
        {
            rdma_memory_region *region = NULL;
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
            if (region==NULL) {
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
        inline void deallocate(rdma_memory_region *region)
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
        inline rdma_memory_region* allocate_temporary_region(std::size_t length)
        {
            rdma_memory_region *region = new rdma_memory_region();
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
            rdma_memory_region *region = allocate_region(length);
            return region->get_address();
        }

        //----------------------------------------------------------------------------
        // deallocate a region using its memory address as handle
        // this involves a map lookup to find the region and is therefore
        // less efficient than releasing memory via the region pointer
        void deallocate(void *address, size_t size=0)
        {
            rdma_memory_region *region = pointer_map_[address];
            deallocate(region);
        }

        //----------------------------------------------------------------------------
        // find an rdma_memory_region* from the memory address it wraps
        rdma_memory_region *RegionFromAddress(char * const addr) {
            return pointer_map_[addr];
        }

        //----------------------------------------------------------------------------
        // internal variables
        //----------------------------------------------------------------------------
        // used to map the internal memory address to the region that
        // holds the registration information
        std::unordered_map<const void *, rdma_memory_region*> pointer_map_;
*/
        // protection domain that memory is registered with
        rdma_protection_domain_ptr protection_domain_;

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
