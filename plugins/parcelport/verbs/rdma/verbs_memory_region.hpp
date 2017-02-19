//  Copyright (c) 2015-2016 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_VERBS_MEMORY_REGION_HPP
#define HPX_PARCELSET_POLICIES_VERBS_MEMORY_REGION_HPP

#include <plugins/parcelport/parcelport_logging.hpp>
#include <plugins/parcelport/verbs/rdma/rdma_error.hpp>
#include <plugins/parcelport/verbs/rdma/verbs_protection_domain.hpp>
//
#include <infiniband/verbs.h>
#include <errno.h>
//
#include <memory>

namespace hpx {
namespace parcelset {
namespace policies {
namespace verbs
{
    struct verbs_memory_region
    {
        // --------------------------------------------------------------------
        verbs_memory_region() :
            region_(nullptr), address_(nullptr), flags_(0), size_(0), used_space_(0) {}

        // --------------------------------------------------------------------
        verbs_memory_region(struct ibv_mr *region, char * address,
            uint32_t flags, uint64_t size) :
                region_(region), address_(address), flags_(flags),
                size_(size), used_space_(0) {}

        // --------------------------------------------------------------------
        // construct a memory region object by registering an existing address buffer
        verbs_memory_region(verbs_protection_domain_ptr pd,
            const void *buffer, const uint64_t length)
        {
            address_    = static_cast<char*>(const_cast<void*>(buffer));
            size_       = length;
            used_space_ = length;
            flags_      = BLOCK_USER;

            region_ = ibv_reg_mr(
                pd->getDomain(),
                const_cast<void*>(buffer), used_space_,
                IBV_ACCESS_LOCAL_WRITE |
                IBV_ACCESS_REMOTE_WRITE |
                IBV_ACCESS_REMOTE_READ);

            if (region_ == nullptr) {
                int err = errno;
                rdma_error(errno, "error registering user mem ibv_reg_mr ");
                LOG_ERROR_MSG(
                    "error registering user mem ibv_reg_mr " << hexpointer(buffer) << " "
                    << hexlength(length) << " error/message: " << err << "/"
                    << rdma_error::error_string(err));
            }
            else {
                LOG_DEBUG_MSG(
                    "OK registering memory ="
                    << hexpointer(buffer) << " : " << hexpointer(region_->addr)
                    << " length " << hexlength(get_length()));
            }

        }

        // --------------------------------------------------------------------
        // allocate a block of size length and register it
        int allocate(verbs_protection_domain_ptr pd, uint64_t length)
        {
            // Allocate storage for the memory region.
            void *buffer = new char[length];
            if (buffer != nullptr) {
                LOG_DEBUG_MSG("allocated storage for memory region with malloc OK "
                    << hexnumber(length));
            }

            region_ = ibv_reg_mr(
                pd->getDomain(),
                buffer, length,
                IBV_ACCESS_LOCAL_WRITE |
                IBV_ACCESS_REMOTE_WRITE |
                IBV_ACCESS_REMOTE_READ);

            if (region_ == nullptr) {
                LOG_ERROR_MSG("error registering ibv_reg_mr : "
                    << " " << errno << " " << rdma_error::error_string(errno));
                return -1;
            }
            else {
                LOG_DEBUG_MSG("OK registering ibv_reg_mr");
            }
            address_ = static_cast<char*>(region_->addr);
            size_    = length;

            LOG_DEBUG_MSG("allocated/registered memory region " << hexpointer(this)
                << " with local key " << get_local_key()
                << " at address " << get_address()
                << " with length " << hexlength(get_length()));
            return 0;
        }

        // --------------------------------------------------------------------
        // destroy the region and memory according to flag settings
        ~verbs_memory_region()
        {
            release();
        }

        // --------------------------------------------------------------------
        // Deregister and free the memory region.
        // returns 0 when successful, -1 otherwise
        int release(void)
        {
            LOG_TRACE_MSG("About to release memory region with local key "
                << get_local_key());
            if (region_ != nullptr) {
                // get these before deleting/unregistering (for logging)
                void *buffer = get_base_address();
                LOG_EXCLUSIVE(
                    uint32_t length = get_length();
                );
                //
                if (!get_partial_region()) {
                    if (ibv_dereg_mr (region_)) {
                        LOG_ERROR_MSG("Error, ibv_dereg_mr() failed\n");
                        return -1;
                    }
                    else {
                        LOG_DEBUG_MSG("deregistered memory region with local key "
                            << get_local_key()
                            << " at address " << buffer
                            << " with length " << hexlength(length));
                    }
                }
                if (!get_partial_region() && !get_user_region()) {
                    delete [](static_cast<char*>(buffer));
                }
                region_ = nullptr;
            }
            return 0;
        }


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
            return static_cast<char*>(region_->addr);
        }

        // --------------------------------------------------------------------
        // Get the allocated length of the internal memory region.
        inline uint64_t get_length(void) const {
            return (uint32_t) region_->length;
        }

        // --------------------------------------------------------------------
        // Get the size memory chunk usable by this memory region,
        // this may be smaller than the value returned by get_length
        // if the region is a sub region (partial region) within another block
        inline uint64_t get_size(void) const {
            return size_;
        }

        // --------------------------------------------------------------------
        // Get the local key of the memory region.
        inline uint32_t get_local_key(void) const {
            return region_->lkey;
        }

        // --------------------------------------------------------------------
        // Get the remote key of the memory region.
        inline uint32_t get_remote_key(void) const {
            return region_->rkey;
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
        // return the underlying infiniband region handle
        inline struct ibv_mr *get_region() { return region_; }

        // --------------------------------------------------------------------
        // flags used for management of lifetime
        enum {
            BLOCK_USER    = 1,
            BLOCK_TEMP    = 2,
            BLOCK_PARTIAL = 4,
        };

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
        // on destruction, it is not unregister or deleted as the 'parent' region
        // will delete many partial regions on destruction
        inline void set_partial_region() {
            flags_ |= BLOCK_PARTIAL;
        }
        inline bool get_partial_region() const {
            return (flags_ & BLOCK_PARTIAL) == BLOCK_PARTIAL;
        }

    private:
        // The internal Infiniband memory region handle
        struct ibv_mr *region_;

        // we may be a piece of a larger region, this gives the start address
        // of this piece of the region. This is the address that should be used for data
        // storage
        char *address_;

        // flags to control lifetime of blocks
        uint32_t flags_;

        // The size of the memory buffer, if this is a partial region
        // it will be smaller than the value returned by region_->length
        uint64_t size_;

        // space used by a message in the memory region.
        uint64_t used_space_;
    };

    // Smart pointer for verbs_memory_region object.
    typedef std::shared_ptr<verbs_memory_region> verbs_memory_region_ptr;

}}}}

#endif
