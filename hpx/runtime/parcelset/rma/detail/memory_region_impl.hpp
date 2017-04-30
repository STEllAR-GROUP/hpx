//  Copyright (c) 2015-2016 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_RMA_MEMORY_REGION
#define HPX_PARCELSET_POLICIES_RMA_MEMORY_REGION

#include <hpx/traits/rma_memory_region_traits.hpp>
#include <hpx/runtime/parcelset/rma/memory_region.hpp>
//
#include <memory>
//
namespace hpx {
namespace parcelset {
namespace rma {
namespace detail
{
    // --------------------------------------------------------------------
    // a memory region is a pinned block of memory that has been specialized
    // by a particular region provider. Each provider (infiniband, libfabric,
    // other) has a different definition for the region object and the protection
    // domain used to limit access.
    // Code that does not 'know' which parcelport is being used, must use
    // the memory_region class to manage regions, but parcelport code
    // may use the correcft type for the parcelport in question.
    // --------------------------------------------------------------------
    template <typename RegionProvider>
    class memory_region_impl : public memory_region
    {
      public:
        typedef typename RegionProvider::provider_domain provider_domain;
        typedef typename RegionProvider::provider_region provider_region;

        // --------------------------------------------------------------------
        memory_region_impl() :
            memory_region(), region_(nullptr) {}

        // --------------------------------------------------------------------
        memory_region_impl(
            provider_region *region, char *address,
            char *base_address, uint64_t size, uint32_t flags)
            : memory_region(address, base_address, size, flags)
            , region_(region) {}

        // --------------------------------------------------------------------
        // construct a memory region object by registering an existing address buffer
        memory_region_impl(provider_domain *pd, const void *buffer, const uint64_t length)
        {
            address_    = static_cast<char *>(const_cast<void *>(buffer));
            base_addr_  = address_;
            size_       = length;
            used_space_ = length;
            flags_      = BLOCK_USER;

            int ret = traits::rma_memory_region_traits<RegionProvider>::register_memory(
                pd, const_cast<void*>(buffer), length,
                traits::rma_memory_region_traits<RegionProvider>::flags(),
                0, (uint64_t)address_, 0, &(region_), nullptr);

            if (ret) {
                LOG_ERROR_MSG(
                    "error registering region "
                    << hexpointer(buffer) << hexlength(length));
                throw std::runtime_error("error in memory registration");
            }
            else {
                LOG_DEBUG_MSG(
                    "OK registering region "
                    << hexpointer(buffer) << hexpointer(address_)
                    << "desc " << hexpointer(fi_mr_desc(region_))
                    << "rkey " << hexpointer(fi_mr_key(region_))
                    << "length " << hexlength(size_));
            }
        }

        // --------------------------------------------------------------------
        // allocate a block of size length and register it
        int allocate(provider_domain *pd, uint64_t length)
        {
            // Allocate storage for the memory region.
            void *buffer = new char[length];
            if (buffer != nullptr) {
                LOG_DEBUG_MSG("allocated storage for memory region with malloc OK "
                    << hexnumber(length));
            }
            address_    = static_cast<char*>(buffer);
            base_addr_  = static_cast<char*>(buffer);
            size_       = length;
            used_space_ = 0;

            int ret = traits::rma_memory_region_traits<RegionProvider>::register_memory(
                pd, const_cast<void*>(buffer), length,
                traits::rma_memory_region_traits<RegionProvider>::flags(),
                0, (uint64_t)address_, 0, &(region_), nullptr);

            if (ret) {
                LOG_ERROR_MSG(
                    "error registering region "
                    << hexpointer(buffer) << hexlength(length));
                throw std::runtime_error("error in memory registration");
            }
            else {
                LOG_DEBUG_MSG(
                    "OK registering region "
                    << hexpointer(buffer) << hexpointer(address_)
                    << "desc " << hexpointer(fi_mr_desc(region_))
                    << "rkey " << hexpointer(fi_mr_key(region_))
                    << "length " << hexlength(size_));
            }

            LOG_DEBUG_MSG("allocated/registered memory region " << hexpointer(this)
                << "with local key " << hexnumber(get_local_key())
                << "at address " << hexpointer(get_address())
                << "with length " << hexlength(get_size()));
            return 0;
        }

        // --------------------------------------------------------------------
        // destroy the region and memory according to flag settings
        ~memory_region_impl()
        {
            if (get_partial_region()) return;
            release();
        }

        // --------------------------------------------------------------------
        // Deregister and free the memory region.
        // returns 0 when successful, -1 otherwise
        int release(void)
        {
            if (region_ != nullptr) {
                LOG_TRACE_MSG("About to release memory region with local key "
                    << hexpointer(get_local_key()));
                // get these before deleting/unregistering (for logging)
                const void *buffer = get_base_address();
                LOG_EXCLUSIVE(
                    uint32_t length = get_size();
                );
                //
                if (traits::rma_memory_region_traits<RegionProvider>::
                    unregister_memory(region_))
                {
                    LOG_ERROR_MSG("Error, fi_close mr failed\n");
                    return -1;
                }
                else {
                    LOG_DEBUG_MSG("deregistered memory region with local key "
                        << hexpointer(get_local_key())
                        << "at address " << hexpointer(buffer)
                        << "with length " << hexlength(length));
                }
                if (!get_user_region()) {
                    delete [](static_cast<const char*>(buffer));
                }
                region_ = nullptr;
            }
            return 0;
        }

        // --------------------------------------------------------------------
        // Get the local descriptor of the memory region.
        virtual void* get_local_key(void) const {
            return
                traits::rma_memory_region_traits<RegionProvider>::get_local_key(region_);
        }

        // --------------------------------------------------------------------
        // Get the remote key of the memory region.
        virtual uint64_t get_remote_key(void) const {
            return
                traits::rma_memory_region_traits<RegionProvider>::get_remote_key(region_);
        }

        // --------------------------------------------------------------------
        // return the underlying infiniband region handle
        inline provider_region *get_region() const { return region_; }

    private:
        // The internal network type dependent memory region handle
        provider_region *region_;

    };

}}}}

#endif
