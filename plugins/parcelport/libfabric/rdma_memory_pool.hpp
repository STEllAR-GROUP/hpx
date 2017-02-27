//  Copyright (c) 2014-2016 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_LIBFABRIC_MEMORY_POOL
#define HPX_PARCELSET_POLICIES_LIBFABRIC_MEMORY_POOL

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
#include <hpx/config/parcelport_defines.hpp>
//
#include <plugins/parcelport/parcelport_logging.hpp>
#include <plugins/parcelport/libfabric/rdma_locks.hpp>
#include <plugins/parcelport/libfabric/libfabric_memory_region.hpp>

// the default memory chunk size in bytes
#define RDMA_POOL_1K_CHUNK_SIZE     0x001*0x0400 //  1KB
#define RDMA_POOL_SMALL_CHUNK_SIZE  0x010*0x0400 // 16KB
#define RDMA_POOL_MEDIUM_CHUNK_SIZE 0x040*0x0400 // 64KB
#define RDMA_POOL_LARGE_CHUNK_SIZE  0x400*0x0400 //  1MB

#define RDMA_POOL_MAX_1K_CHUNKS     1024
#define RDMA_POOL_MAX_SMALL_CHUNKS  512
#define RDMA_POOL_MAX_MEDIUM_CHUNKS 128
#define RDMA_POOL_MAX_LARGE_CHUNKS  16

// if the HPX configuration has set a different value, use it
#if defined(HPX_PARCELPORT_LIBFABRIC_MEMORY_CHUNK_SIZE)
# undef RDMA_POOL_SMALL_CHUNK_SIZE
# define RDMA_POOL_SMALL_CHUNK_SIZE HPX_PARCELPORT_LIBFABRIC_MEMORY_CHUNK_SIZE
#endif

static_assert ( HPX_PARCELPORT_LIBFABRIC_MEMORY_CHUNK_SIZE<RDMA_POOL_MEDIUM_CHUNK_SIZE ,
"Default memory Chunk size must be less than medium chunk size" );


// Description of memory pool objects:
//
// memory_region_allocator:
// An allocator that returns memory of the requested size. The memory is pinned
// and ready to be used for RDMA operations. A memory_region object is
// used, it contains the memory registration information needed by the libfabric API.
//
// rdma_chunk_pool :
// Allocate N chunks of memory in one go, a single memory registration is
// used for the whole block, and this is divided into smaller blocks that
// are then used by the pool_container.
// @TODO: Note that originally a boost::pool was used but this was replaced by
// a simple block allocation and needs to be cleaned as it puts blocks onto
// a stack that is duplicated in pool_container.
//
// pool_container:
// The pool container wraps an rdma_chunk_pool and provides a stack. When a user
// requests a small block, one is popped off the stack. At startup, the pool_container
// requests a large number of blocks from the rdma_chunk_pool and sets the correct
// address offset within each larger chunk for each small block and pushes the mini
// libfabric_memory_region onto the stack. Thus N small rdma_regions are created from a
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
namespace libfabric
{
    struct rdma_memory_pool;
}}}}

namespace hpx {
namespace parcelset {
namespace policies {
namespace libfabric
{

    namespace bl = boost::lockfree;

    // A simple tag type we use for logging assistance (identification)
    struct pool_tiny   { static const char *desc() { return "Tiny ";   } };
    struct pool_small  { static const char *desc() { return "Small ";  } };
    struct pool_medium { static const char *desc() { return "Medium "; } };
    struct pool_large  { static const char *desc() { return "Large ";  } };

    // --------------------------------------------------------------------
    // allocate a memory_region and register the memory
    struct memory_region_allocator
    {
        memory_region_allocator() {}

        static libfabric_memory_region_ptr malloc(struct fid_domain *pd,
            const std::size_t bytes)
        {
            libfabric_memory_region_ptr region =
                std::make_shared<libfabric_memory_region>();
            LOG_DEBUG_MSG("Allocating " << hexuint32(bytes) << "using chunk mallocator");
            region->allocate(pd, bytes);
            return region;
        }

        static void free(libfabric_memory_region_ptr region) {
            LOG_DEBUG_MSG("Freeing a block from chunk mallocator (ref count) "
                << region.use_count());
            region.reset();
        }
    };

    // ---------------------------------------------------------------------------
    // pool_container, collect some routines for reuse with
    // small, medium, large chunks etc
    // ---------------------------------------------------------------------------
    template <typename chunk_allocator, typename PoolType,
        std::size_t ChunkSize, std::size_t MaxChunks>
    struct pool_container
    {
        // ------------------------------------------------------------------------
        pool_container(struct fid_domain * pd, std::size_t num_chunks) :
            used_(0), pd_(pd)
        {
        }

        // ------------------------------------------------------------------------
        bool allocate_pool(std::size_t num_chunks)
        {
            LOG_DEBUG_MSG(PoolType::desc() << "Allocating "
                << "ChunkSize " << hexuint32(ChunkSize)
                << "num_chunks " << decnumber(num_chunks)
                << "total " << hexuint32(ChunkSize*num_chunks));

            // Allocate one very large registered block for N small blocks
            libfabric_memory_region_ptr block =
                    chunk_allocator().malloc(pd_, ChunkSize*num_chunks);
            // store a copy of this to make sure it is 'alive'
            block_list_[block->get_address()] = block;

            // break the large region into N small regions
            uint64_t offset = 0;
            for (std::size_t i=0; i<num_chunks; ++i) {
                LOG_TRACE_MSG(PoolType::desc() << "Allocate Block "
                    << i << " of size " << hexlength(ChunkSize));

                // we must keep a copy of the sub-region since we only pass
                // pointers to regions around the code.
                region_list_[i] = libfabric_memory_region(
                    block->get_region(),
                    static_cast<char*>(block->get_base_address()) + offset,
                    static_cast<char*>(block->get_base_address()),
                    libfabric_memory_region::BLOCK_PARTIAL,
                    ChunkSize
                );
                // push the pointer onto our stack
                push(&region_list_[i]);
                offset += ChunkSize;
            }
            used_ = 0;
            return true;
        }

        // ------------------------------------------------------------------------
        void DeallocatePool()
        {
            if (used_!=0) {
                LOG_ERROR_MSG(PoolType::desc()
                    << "Deallocating free_list : Not all blocks were returned "
                    << " refcounts " << decnumber(used_));
            }
            while (!free_list_.empty()) {
                // clear our stack
                pop();
            }
            // wipe our copies of sub-regions (no clear function for std::array)
            std::fill(region_list_.begin(), region_list_.end(), libfabric_memory_region());
            // release references to shared arrays
            block_list_.clear();
        }

        // ------------------------------------------------------------------------
        inline void push(libfabric_memory_region *region)
        {
            LOG_TRACE_MSG(PoolType::desc() << "Push block "
                << hexpointer(region->get_address()) << hexlength(region->get_size())
                << decnumber(used_-1));

            uintptr_t val = uintptr_t(region->get_address());
            LOG_TRACE_MSG(PoolType::desc()
                << "Writing 0xdeadbeef to region address "
                << hexpointer(region->get_address()));

            if (region->get_address()!=nullptr) {
                // get use the pointer to the region
                uintptr_t *ptr = reinterpret_cast<uintptr_t*>(region->get_address());
                for (unsigned int c=0; c<ChunkSize/8; ++c) {
                    ptr[c] = 0xdeadbeef;
                    ptr[c] = val;
                }
            }

            if (!free_list_.push(region)) {
                LOG_ERROR_MSG(PoolType::desc() << "Error in memory pool push");
            }
            // decrement one reference
            used_--;
        }

        // ------------------------------------------------------------------------
        inline libfabric_memory_region *pop()
        {
            // if we have not exceeded our max size, allocate a new block
            if (free_list_.empty()) {
                //  LOG_TRACE_MSG("Creating new small Block as free list is empty "
                // "but max chunks " << max_small_chunks_ << " not reached");
                //  AllocateRegisteredBlock(length);
                //std::terminate();
                return nullptr;
            }
            // get a block
            libfabric_memory_region *region = nullptr;
            if (!free_list_.pop(region)) {
                LOG_DEBUG_MSG(PoolType::desc() << "Error in memory pool pop");
            }
            // Keep reference counts to self so that we can check
            // this pool is not deleted whilst blocks still exist
            used_++;
            LOG_TRACE_MSG(PoolType::desc() << "Pop block "
                << hexpointer(region->get_address()) << hexlength(region->get_size())
                << decnumber(used_));
            //
            return region;
        }

        void decrement_used_count(uint32_t N) {
            used_ -= N;
        }

        constexpr std::size_t chunk_size() const { return ChunkSize; }
        //
        std::atomic<int>                                              used_;
        struct fid_domain *                                           pd_;
        std::unordered_map<const char *, libfabric_memory_region_ptr> block_list_;
        std::array<libfabric_memory_region, MaxChunks>               region_list_;
        bl::stack<libfabric_memory_region*, bl::capacity<MaxChunks>> free_list_;
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
        rdma_memory_pool(struct fid_domain * pd) :
            protection_domain_(pd),
            tiny_  (pd, RDMA_POOL_MAX_1K_CHUNKS ),
            small_ (pd, RDMA_POOL_MAX_SMALL_CHUNKS ),
            medium_(pd, RDMA_POOL_MAX_MEDIUM_CHUNKS ),
            large_ (pd, RDMA_POOL_MAX_LARGE_CHUNKS ),
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
        void deallocate_pools()
        {
            tiny_.DeallocatePool();
            small_.DeallocatePool();
            medium_.DeallocatePool();
            large_.DeallocatePool();
        }

        // -------------------------
        // User allocation interface
        // -------------------------
        // The libfabric_memory_region* versions of allocate/deallocate
        // should be used in preference to the std:: compatible
        // versions using char* for efficiency

        //----------------------------------------------------------------------------
        // query the pool for a chunk of a given size to see if one is available
        // this function is 'unsafe' because it is not thread safe and another
        // thread may push/pop a block after this is called and invalidate the result.
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
        inline libfabric_memory_region *allocate_region(size_t length)
        {
            libfabric_memory_region *region = nullptr;
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

            LOG_TRACE_MSG("Popping Block"
                << " buffer "    << hexpointer(region->get_address())
                << " region "    << hexpointer(region)
                << " size "      << hexlength(region->get_size())
                << " chunksize "
                << hexlength(tiny_.chunk_size()) << " "
                << hexlength(small_.chunk_size()) << " "
                << hexlength(medium_.chunk_size()) << " "
                << hexlength(large_.chunk_size()) << " "
                << "free (t) "  << (RDMA_POOL_MAX_1K_CHUNKS-tiny_.used_)
                << " used "      << decnumber(this->tiny_.used_)
                << "free (s) "  << (RDMA_POOL_MAX_SMALL_CHUNKS-small_.used_)
                << " used "      << decnumber(this->small_.used_)
                << "free (m) "  << (RDMA_POOL_MAX_MEDIUM_CHUNKS-medium_.used_)
                << " used "      << decnumber(this->medium_.used_)
                << "free (l) "  << (RDMA_POOL_MAX_LARGE_CHUNKS-large_.used_)
                << " used "      << decnumber(this->large_.used_));
            //
            return region;
        }

        //----------------------------------------------------------------------------
        // release a region back to the pool
        inline void deallocate(libfabric_memory_region *region)
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
        inline libfabric_memory_region* allocate_temporary_region(std::size_t length)
        {
            libfabric_memory_region *region = new libfabric_memory_region();
            region->set_temp_region();
            region->allocate(protection_domain_, length);
            temp_regions++;
            LOG_TRACE_MSG("Allocating temp registered block "
                << hexpointer(region->get_address()) << hexlength(length)
                << decnumber(temp_regions));
            return region;
        }

        //----------------------------------------------------------------------------
        // protection domain that memory is registered with
        struct fid_domain * protection_domain_;

        // maintain 4 pools of thread safe pre-allocated regions of fixed size.
        pool_container<memory_region_allocator, pool_tiny,
            RDMA_POOL_1K_CHUNK_SIZE,         RDMA_POOL_MAX_1K_CHUNKS> tiny_;
        pool_container<memory_region_allocator, pool_small,
            RDMA_POOL_SMALL_CHUNK_SIZE,   RDMA_POOL_MAX_SMALL_CHUNKS> small_;
        pool_container<memory_region_allocator, pool_medium,
            RDMA_POOL_MEDIUM_CHUNK_SIZE, RDMA_POOL_MAX_MEDIUM_CHUNKS> medium_;
        pool_container<memory_region_allocator, pool_large,
            RDMA_POOL_LARGE_CHUNK_SIZE,   RDMA_POOL_MAX_LARGE_CHUNKS> large_;

        // counters
        std::atomic<int> temp_regions;
        std::atomic<int> user_regions;
    };

    typedef std::shared_ptr<rdma_memory_pool> rdma_memory_pool_ptr;
}}}}

#endif
