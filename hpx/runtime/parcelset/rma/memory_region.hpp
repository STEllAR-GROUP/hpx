//  Copyright (c) 2015-2016 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_RMA_MEMORY_REGION
#define HPX_PARCELSET_POLICIES_RMA_MEMORY_REGION

#include <hpx/traits/rma_memory_region_traits.hpp>
#include <plugins/parcelport/parcelport_logging.hpp>
//
#include <memory>
#include <iomanip>

namespace hpx {
namespace parcelset {
namespace rma
{
    // --------------------------------------------------------------------
    // a base class that provides an API for creating/accessing
    // pinned memory blocks. This will be overridden by concrete
    // implementations for each parcelport.
    // --------------------------------------------------------------------
    class memory_region
    {
      public:
        // --------------------------------------------------------------------
        // flags used for management of lifetime
        enum {
            BLOCK_USER    = 1,
            BLOCK_TEMP    = 2,
            BLOCK_PARTIAL = 4,
        };

        memory_region() :
            address_(nullptr), base_addr_(nullptr), size_(0), used_space_(0), flags_(0) {}

        memory_region(
            char *address, char *base_address, uint64_t size, uint32_t flags)
            : address_(address), base_addr_(base_address)
            , size_(size), used_space_(0), flags_(flags) {}

        virtual ~memory_region() {}

        // --------------------------------------------------------------------
        // return the address of this memory region block. If this
        // is a partial region, then the address will be offset from the
        // base address
        inline char *get_address(void) const {
            return address_;
        }

        // --------------------------------------------------------------------
        // Get the address of the base memory region.
        // This is the address of the memory allocated from the system
        inline char *get_base_address(void) const {
            return base_addr_;
        }

        // --------------------------------------------------------------------
        // Get the size of the memory chunk usable by this memory region,
        // this may be smaller than the value returned by get_length
        // if the region is a sub region (partial region) within another block
        inline uint64_t get_size(void) const {
            return size_;
        }

        // --------------------------------------------------------------------
        // Set the size used by a message in the memory region.
        inline void set_message_length(uint32_t length) {
            used_space_ = length;
        }

        // --------------------------------------------------------------------
        // Get the size used by a message in the memory region.
        inline uint32_t get_message_length(void) const {
            return used_space_;
        }

        // --------------------------------------------------------------------
        // A user allocated region use memory allocted by the user.
        // on destruction, the memory is unregisterd, but not deleted
        inline void set_user_region() {
            flags_ |= BLOCK_USER;
        }

        inline bool get_user_region() const {
            return (flags_ & BLOCK_USER) == BLOCK_USER;
        }

        // --------------------------------------------------------------------
        // A temp region is one that the memory pool is not managing
        // so it is unregistered and deleted when returned to the pool and not reused
        inline void set_temp_region() {
            flags_ |= BLOCK_TEMP;
        }

        inline bool get_temp_region() const {
            return (flags_ & BLOCK_TEMP) == BLOCK_TEMP;
        }

        // --------------------------------------------------------------------
        // a partial region is a subregion of a larger memory region
        // on destruction, it is not unregistered or deleted as the 'parent' region
        // will delete many partial regions on destruction
        inline void set_partial_region() {
            flags_ |= BLOCK_PARTIAL;
        }

        inline bool get_partial_region() const {
            return (flags_ & BLOCK_PARTIAL) == BLOCK_PARTIAL;
        }

        // --------------------------------------------------------------------
        // Get the local descriptor of the memory region.
        virtual void* get_local_key(void) const = 0;

        // --------------------------------------------------------------------
        // Get the remote key of the memory region.
        virtual uint64_t get_remote_key(void) const = 0;

        // --------------------------------------------------------------------
        friend std::ostream & operator<<(std::ostream & os,
            memory_region const & region)
        {
            os  << "region " << hexpointer(&region)
                << "base address " << hexpointer(region.base_addr_)
                << "address " << hexpointer(region.address_)
                << "flags " << hexbyte(region.flags_)
                << "size " << hexlength(region.size_)
                << "used_space_ " << hexlength(region.used_space_);
            return os;
        }

    public:
        // we may be a piece of a larger region, this gives the start address
        // of this piece of the region. This is the address that should be used for data
        // storage
        char *address_;

        // if we are part of a larger region, this is the base address of
        // that larger region
        char *base_addr_;

        // The size of the memory buffer, if this is a partial region
        // it will be smaller than the value returned by region_->length
        uint64_t size_;

        // space used by a message in the memory region.
        uint64_t used_space_;

        // flags to control lifetime of blocks
        uint32_t flags_;
    };

}}}

#endif
