//  Copyright (c) 2015-2016 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/parcelport_libfabric/config/defines.hpp>
#include <hpx/parcelport_libfabric/parcelport_logging.hpp>
#include <hpx/parcelport_libfabric/rma_memory_region_traits.hpp>
//
#include <memory>
//

namespace hpx { namespace parcelset {
    // --------------------------------------------------------------------
    // a base class that provides an API for creating/accessing
    // pinned memory blocks. This will be overridden by concrete
    // implementations for each parcelport.
    struct rma_memory_base
    {
        rma_memory_base();
    };

    // --------------------------------------------------------------------
    template <typename RegionProvider>
    struct rma_memory_region
    {
        typedef typename RegionProvider::provider_domain provider_domain;
        typedef typename RegionProvider::provider_region provider_region;

        // --------------------------------------------------------------------
        rma_memory_region()
          : region_(nullptr)
          , address_(nullptr)
          , base_addr_(nullptr)
          , size_(0)
          , used_space_(0)
          , flags_(0)
        {
        }

        // --------------------------------------------------------------------
        rma_memory_region(provider_region* region, char* address,
            char* base_address, uint64_t size, uint32_t flags)
          : region_(region)
          , address_(address)
          , base_addr_(base_address)
          , size_(size)
          , used_space_(0)
          , flags_(flags)
        {
        }

        // --------------------------------------------------------------------
        // construct a memory region object by registering an existing address buffer
        rma_memory_region(
            provider_domain* pd, const void* buffer, const uint64_t length)
        {
            address_ = static_cast<char*>(const_cast<void*>(buffer));
            base_addr_ = address_;
            size_ = length;
            used_space_ = length;
            flags_ = BLOCK_USER;

            int ret = traits::rma_memory_region_traits<
                RegionProvider>::register_memory(pd, const_cast<void*>(buffer),
                length,
                traits::rma_memory_region_traits<RegionProvider>::flags(), 0,
                (uint64_t) address_, 0, &(region_), nullptr);

            if (ret)
            {
                LOG_ERROR_MSG("error registering region " << hexpointer(buffer)
                                                          << hexlength(length));
                throw std::runtime_error("error in memory registration");
            }
            else
            {
                LOG_DEBUG_MSG("OK registering region "
                    << hexpointer(buffer) << hexpointer(address_) << "desc "
                    << hexpointer(fi_mr_desc(region_)) << "rkey "
                    << hexpointer(fi_mr_key(region_)) << "length "
                    << hexlength(size_));
            }
        }

        // --------------------------------------------------------------------
        // allocate a block of size length and register it
        int allocate(provider_domain* pd, uint64_t length)
        {
            // Allocate storage for the memory region.
            void* buffer = new char[length];
            if (buffer != nullptr)
            {
                LOG_DEBUG_MSG(
                    "allocated storage for memory region with malloc OK "
                    << hexnumber(length));
            }
            address_ = static_cast<char*>(buffer);
            base_addr_ = static_cast<char*>(buffer);
            size_ = length;
            used_space_ = 0;

            int ret = traits::rma_memory_region_traits<
                RegionProvider>::register_memory(pd, const_cast<void*>(buffer),
                length,
                traits::rma_memory_region_traits<RegionProvider>::flags(), 0,
                (uint64_t) address_, 0, &(region_), nullptr);

            if (ret)
            {
                LOG_ERROR_MSG("error registering region " << hexpointer(buffer)
                                                          << hexlength(length));
                throw std::runtime_error("error in memory registration");
            }
            else
            {
                LOG_DEBUG_MSG("OK registering region "
                    << hexpointer(buffer) << hexpointer(address_) << "desc "
                    << hexpointer(fi_mr_desc(region_)) << "rkey "
                    << hexpointer(fi_mr_key(region_)) << "length "
                    << hexlength(size_));
            }

            LOG_DEBUG_MSG("allocated/registered memory region "
                << hexpointer(this) << "with desc " << hexnumber(get_desc())
                << "at address " << hexpointer(get_address()) << "with length "
                << hexlength(get_size()));
            return 0;
        }

        // --------------------------------------------------------------------
        // destroy the region and memory according to flag settings
        ~rma_memory_region()
        {
            if (get_partial_region())
                return;
            release();
        }

        // --------------------------------------------------------------------
        // Deregister and free the memory region.
        // returns 0 when successful, -1 otherwise
        int release(void)
        {
            if (region_ != nullptr)
            {
                LOG_TRACE_MSG("About to release memory region with desc "
                    << hexpointer(get_desc()));
                // get these before deleting/unregistering (for logging)
                const void* buffer = get_base_address();
                LOG_EXCLUSIVE(uint32_t length = get_size(););
                //
                if (traits::rma_memory_region_traits<
                        RegionProvider>::unregister_memory(region_))
                {
                    LOG_ERROR_MSG("Error, fi_close mr failed\n");
                    return -1;
                }
                else
                {
                    LOG_DEBUG_MSG("deregistered memory region with desc "
                        << hexpointer(get_desc()) << "at address "
                        << hexpointer(buffer) << "with length "
                        << hexlength(length));
                }
                if (!get_user_region())
                {
                    delete[](static_cast<const char*>(buffer));
                }
                region_ = nullptr;
            }
            return 0;
        }

        // --------------------------------------------------------------------
        // return the address of this memory region block. If this
        // is a partial region, then the address will be offset from the
        // base address
        inline char* get_address(void) const
        {
            return address_;
        }

        // --------------------------------------------------------------------
        // Get the address of the base memory region.
        // This is the address of the memory allocated from the system
        inline char* get_base_address(void) const
        {
            return base_addr_;
        }

        // --------------------------------------------------------------------
        // Get the size of the memory chunk usable by this memory region,
        // this may be smaller than the value returned by get_length
        // if the region is a sub region (partial region) within another block
        inline uint64_t get_size(void) const
        {
            return size_;
        }

        // --------------------------------------------------------------------
        // Get the local descriptor of the memory region.
        inline void* get_desc(void) const
        {
            return fi_mr_desc(region_);
        }

        // --------------------------------------------------------------------
        // Get the remote key of the memory region.
        inline uint64_t get_remote_key(void) const
        {
            return fi_mr_key(region_);
        }

        // --------------------------------------------------------------------
        // Set the size used by a message in the memory region.
        inline void set_message_length(uint32_t length)
        {
            used_space_ = length;
        }

        // --------------------------------------------------------------------
        // Get the size used by a message in the memory region.
        inline uint32_t get_message_length(void) const
        {
            return used_space_;
        }

        // --------------------------------------------------------------------
        // return the underlying infiniband region handle
        inline struct fid_mr* get_region()
        {
            return region_;
        }

        // --------------------------------------------------------------------
        // flags used for management of lifetime
        enum
        {
            BLOCK_USER = 1,
            BLOCK_TEMP = 2,
            BLOCK_PARTIAL = 4,
        };

        // --------------------------------------------------------------------
        // A user allocated region use memory allocated by the user.
        // on destruction, the memory is unregistered, but not deleted
        inline void set_user_region()
        {
            flags_ |= BLOCK_USER;
        }

        inline bool get_user_region() const
        {
            return (flags_ & BLOCK_USER) == BLOCK_USER;
        }

        // --------------------------------------------------------------------
        // A temp region is one that the memory pool is not managing
        // so it is unregistered and deleted when returned to the pool and not reused
        inline void set_temp_region()
        {
            flags_ |= BLOCK_TEMP;
        }

        inline bool get_temp_region() const
        {
            return (flags_ & BLOCK_TEMP) == BLOCK_TEMP;
        }

        // --------------------------------------------------------------------
        // a partial region is a subregion of a larger memory region
        // on destruction, it is not unregistered or deleted as the 'parent' region
        // will delete many partial regions on destruction
        inline void set_partial_region()
        {
            flags_ |= BLOCK_PARTIAL;
        }

        inline bool get_partial_region() const
        {
            return (flags_ & BLOCK_PARTIAL) == BLOCK_PARTIAL;
        }

        // --------------------------------------------------------------------
        friend std::ostream& operator<<(
            std::ostream& os, rma_memory_region const& region)
        {
            os << "region " << hexpointer(&region) << "base address "
               << hexpointer(region.base_addr_) << "address "
               << hexpointer(region.address_) << "flags "
               << hexbyte(region.flags_) << "size " << hexlength(region.size_)
               << "used_space_ " << hexlength(region.used_space_);
            return os;
        }

    private:
        // The internal network type dependent memory region handle
        provider_region* region_;

        // we may be a piece of a larger region, this gives the start address
        // of this piece of the region. This is the address that should be used for data
        // storage
        char* address_;

        // if we are part of a larger region, this is the base address of
        // that larger region
        char* base_addr_;

        // The size of the memory buffer, if this is a partial region
        // it will be smaller than the value returned by region_->length
        uint64_t size_;

        // space used by a message in the memory region.
        uint64_t used_space_;

        // flags to control lifetime of blocks
        uint32_t flags_;
    };

}}    // namespace hpx::parcelset
