//  Copyright (c) 2014-2017 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_RMA_MEMORY_POOL_STACK
#define HPX_PARCELSET_POLICIES_RMA_MEMORY_POOL_STACK

#include <hpx/lcos/local/mutex.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/util/atomic_count.hpp>
//
#include <hpx/config/parcelport_defines.hpp>
//
#include <hpx/runtime/parcelset/rma/detail/memory_region_impl.hpp>
#include <plugins/parcelport/parcelport_logging.hpp>
#include <plugins/parcelport/performance_counter.hpp>
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

namespace hpx {
namespace parcelset {
namespace rma {
namespace detail
{

    namespace bl = boost::lockfree;

    // A simple tag type we use for logging assistance (identification)
    struct pool_tiny   { static const char *desc() { return "Tiny ";   } };
    struct pool_small  { static const char *desc() { return "Small ";  } };
    struct pool_medium { static const char *desc() { return "Medium "; } };
    struct pool_large  { static const char *desc() { return "Large ";  } };

    // ---------------------------------------------------------------------------
    // memory pool stack is responsible for allocating large blocks of memory
    // from the system heap and splitting them into N small equally sized region/blocks
    // that are then stored in a stack and handed out on demand when a block of an
    // appropriate size is required.
    // The memory pool class maintains N of these stacks for different sized blocks
    // ---------------------------------------------------------------------------
    template <typename RegionProvider,
              typename Allocator,
              typename PoolType,
              std::size_t ChunkSize,
              std::size_t MaxChunks>
    struct memory_pool_stack
    {
        typedef typename RegionProvider::provider_domain domain_type;
        typedef memory_region_impl<RegionProvider>        region_type;
        typedef std::shared_ptr<region_type>             region_ptr;

        // ------------------------------------------------------------------------
        memory_pool_stack(domain_type *pd) :
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
        // at shutdown we might want to disregrad any bocks still preposted as
        // we can't unpost them
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
        performance_counter<unsigned int>                 accesses_;
        performance_counter<unsigned int>                 in_use_;
        //
        domain_type                                      *pd_;
        std::unordered_map<const char *, region_ptr>      block_list_;
        std::array<region_type, MaxChunks>                region_list_;
        bl::stack<region_type*, bl::capacity<MaxChunks>>  free_list_;
    };

}}}}

#endif
