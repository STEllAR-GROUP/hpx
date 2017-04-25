//  Copyright (c) 2014-2016 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_RMA_MEMORY_POOL
#define HPX_PARCELSET_POLICIES_RMA_MEMORY_POOL

#include <hpx/lcos/local/mutex.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/util/atomic_count.hpp>
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
#include <plugins/parcelport/rma_memory_region.hpp>
#include <plugins/parcelport/performance_counter.hpp>

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
// Region onto the stack. Thus N small rdma_regions are created from a
// single larger one and memory blocks come from contiguous memory.
//
// rma_memory_pool:
// The rma_memory_pool maintains 4 pool_container (stacks) of different sized blocks
// so that most user requests can be fulfilled.
// If a request cannot be filled, the pool can generate temporary blocks with
// new allocations and on-the-fly registration of the memory.
// Additionally, it also provides a simple API so users may pass pre-allocated
// memory to the pool for on-the-fly registration (rdma transfer of user memory chunks)
// and later de-registration.


namespace hpx {
namespace parcelset
{

    namespace bl = boost::lockfree;

    // A simple tag type we use for logging assistance (identification)
    struct pool_tiny   { static const char *desc() { return "Tiny ";   } };
    struct pool_small  { static const char *desc() { return "Small ";  } };
    struct pool_medium { static const char *desc() { return "Medium "; } };
    struct pool_large  { static const char *desc() { return "Large ";  } };

    // --------------------------------------------------------------------
    // allocator for memory_regions
    template <typename RegionProvider>
    struct memory_region_allocator
    {
        typedef typename RegionProvider::provider_domain domain_type;
        typedef rma_memory_region<RegionProvider>        region_type;
        typedef std::shared_ptr<region_type>             region_ptr;

        // default empty constructor
        memory_region_allocator() {}

        // allocate a registered memory region
        static region_ptr malloc(domain_type *pd, const std::size_t bytes)
        {
            region_ptr region =
                std::make_shared<region_type>();
            LOG_DEBUG_MSG("Allocating " << hexuint32(bytes) << "using chunk mallocator");
            region->allocate(pd, bytes);
            return region;
        }

        // release a registered memory region
        static void free(region_ptr region) {
            LOG_DEBUG_MSG("Freeing a block from chunk mallocator (ref count) "
                << region.use_count());
            region.reset();
        }
    };

    // ---------------------------------------------------------------------------
    // pool_container, collect some routines for reuse with
    // small, medium, large chunks etc
    // ---------------------------------------------------------------------------
    template <typename RegionProvider,
              typename Allocator,
              typename PoolType,
              std::size_t ChunkSize,
              std::size_t MaxChunks>
    struct pool_container
    {
        typedef typename RegionProvider::provider_domain domain_type;
        typedef rma_memory_region<RegionProvider>        region_type;
        typedef std::shared_ptr<region_type>             region_ptr;

        // ------------------------------------------------------------------------
        pool_container(domain_type *pd) :
            accesses_(0), in_use_(0), pd_(pd)
        {
        }

        // ------------------------------------------------------------------------
        bool allocate_pool()
        {
            LOG_DEBUG_MSG(PoolType::desc() << "Allocating "
                << "ChunkSize " << hexuint32(ChunkSize)
                << "num_chunks " << decnumber(MaxChunks)
                << "total " << hexuint32(ChunkSize*MaxChunks));

            // Allocate one very large registered block for N small blocks
            region_ptr block =
                Allocator().malloc(pd_, ChunkSize*MaxChunks);
            // store a copy of this to make sure it is 'alive'
            block_list_[block->get_address()] = block;

            // break the large region into N small regions
            uint64_t offset = 0;
            for (std::size_t i=0; i<MaxChunks; ++i) {
                // we must keep a copy of the sub-region since we only pass
                // pointers to regions around the code.
                region_list_[i] = region_type(
                    block->get_region(),
                    static_cast<char*>(block->get_base_address()) + offset,
                    static_cast<char*>(block->get_base_address()),
                    ChunkSize,
                    region_type::BLOCK_PARTIAL
                );
                LOG_TRACE_MSG(PoolType::desc() << "Allocate Block "
                    << decnumber(i)
                    << region_list_[i]);
                // push the pointer onto our stack
                push(&region_list_[i]);
                offset += ChunkSize;
            }
            in_use_ = 0;
            return true;
        }

        // ------------------------------------------------------------------------
        void DeallocatePool()
        {
            if (in_use_!=0) {
                LOG_ERROR_MSG(PoolType::desc()
                    << "Deallocating free_list : Not all blocks were returned "
                    << " refcounts " << decnumber(in_use_));
            }
            region_type* region = nullptr;
            while (!free_list_.pop(region)) {
                // clear our stack
                delete region;
            }
            // wipe our copies of sub-regions (no clear function for std::array)
            std::fill(region_list_.begin(), region_list_.end(), region_type());
            // release references to shared arrays
            block_list_.clear();
        }

        // ------------------------------------------------------------------------
        inline void push(region_type *region)
        {
            LOG_TRACE_MSG(PoolType::desc() << "Push block " << *region
                << "Used " << decnumber(in_use_-1)
                << "Accesses " << decnumber(accesses_));

            LOG_EXCLUSIVE(
                uintptr_t val = uintptr_t(region->get_address());
                LOG_TRACE_MSG(PoolType::desc()
                    << "Writing 0xdeadbeef to region address "
                    << hexpointer(val));
                if (region->get_address()!=nullptr) {
                    // get use the pointer to the region
                    uintptr_t *ptr = reinterpret_cast<uintptr_t*>(val);
                    for (unsigned int c=0; c<ChunkSize/8; ++c) {
                        ptr[c] = 0xdeadbeef;
                    }
                }
            );

            if (!free_list_.push(region)) {
                LOG_ERROR_MSG(PoolType::desc() << "Error in memory pool push");
            }
            // decrement one reference
            --in_use_;
        }

        // ------------------------------------------------------------------------
        inline region_type *pop()
        {
            // get a block
            region_type *region = nullptr;
            if (!free_list_.pop(region)) {
                LOG_DEBUG_MSG(PoolType::desc() << "Error in memory pool pop");
                return nullptr;
            }
            ++in_use_;
            ++accesses_;
            LOG_TRACE_MSG(PoolType::desc() << "Pop block "
                << *region
                << "Used " << decnumber(in_use_)
                << "Accesses " << decnumber(accesses_));
            return region;
        }

        // ------------------------------------------------------------------------
        // at shutdown we can disregrad any bocks still prepoted as we can't
        // unpost them
        void decrement_used_count(uint32_t N) {
            in_use_ -= N;
        }

        // ------------------------------------------------------------------------
        // for debug log messages
        std::string status() {
            std::stringstream temp;
            temp << "| " << PoolType::desc()
                 << "ChunkSize " << hexlength(ChunkSize)
                 << "Free " << decnumber(MaxChunks-in_use_)
                 << "Used " << decnumber(in_use_)
                 << "Accesses " << decnumber(accesses_);
            return temp.str();
        }

        // ------------------------------------------------------------------------
        constexpr std::size_t chunk_size() const { return ChunkSize; }
        //
        performance_counter<unsigned int>                             accesses_;
        performance_counter<unsigned int>                             in_use_;
        //
        domain_type *pd_;
        std::unordered_map<const char *, region_ptr>      block_list_;
        std::array<region_type, MaxChunks>                region_list_;
        bl::stack<region_type*, bl::capacity<MaxChunks>>  free_list_;
    };

    // ---------------------------------------------------------------------------
    // memory pool, holds 4 smaller pools and pops/pushes to the one
    // of the right size for the requested data
    // ---------------------------------------------------------------------------
    template <typename RegionProvider>
    struct rma_memory_pool
    {
        HPX_NON_COPYABLE(rma_memory_pool);

        typedef typename RegionProvider::provider_domain domain_type;
        typedef rma_memory_region<RegionProvider>        region_type;
        typedef memory_region_allocator<RegionProvider>  allocator_type;
        typedef std::shared_ptr<region_type>             region_ptr;

        //----------------------------------------------------------------------------
        // constructor
        rma_memory_pool(domain_type *pd) :
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
        ~rma_memory_pool()
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
        // The Region* versions of allocate/deallocate
        // should be used in preference to the std:: compatible
        // versions using char* for efficiency

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
        inline region_type* allocate_temporary_region(std::size_t length)
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
        // protection domain that memory is registered with
        domain_type *protection_domain_;

        // maintain 4 pools of thread safe pre-allocated regions of fixed size.
        pool_container<RegionProvider, allocator_type, pool_tiny,
            RDMA_POOL_1K_CHUNK_SIZE,         RDMA_POOL_MAX_1K_CHUNKS> tiny_;
        pool_container<RegionProvider, allocator_type, pool_small,
            RDMA_POOL_SMALL_CHUNK_SIZE,   RDMA_POOL_MAX_SMALL_CHUNKS> small_;
        pool_container<RegionProvider, allocator_type, pool_medium,
            RDMA_POOL_MEDIUM_CHUNK_SIZE, RDMA_POOL_MAX_MEDIUM_CHUNKS> medium_;
        pool_container<RegionProvider, allocator_type, pool_large,
            RDMA_POOL_LARGE_CHUNK_SIZE,   RDMA_POOL_MAX_LARGE_CHUNKS> large_;

        // counters
        hpx::util::atomic_count temp_regions;
        hpx::util::atomic_count user_regions;
    };

}}

#endif
