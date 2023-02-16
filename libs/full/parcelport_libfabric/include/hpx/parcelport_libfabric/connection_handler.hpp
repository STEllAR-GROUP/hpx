//  Copyright (c) 2015 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/parcelport_libfabric/config/defines.hpp>
#include <hpx/parcelport_libfabric/fabric_error.hpp>
#include <hpx/parcelport_libfabric/parcelport_logging.hpp>
#include <hpx/parcelport_libfabric/sender.hpp>
#include <hpx/parcelset/parcelport_impl.hpp>
#include <hpx/parcelset_base/locality.hpp>

#include <sstream>
#include <type_traits>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx::parcelset {

    namespace policies::libfabric {
        struct HPX_EXPORT parcelport;
    }    // namespace policies::libfabric

    using namespace hpx::parcelset::policies::libfabric;

    template <>
    struct connection_handler_traits<policies::libfabric::parcelport>
    {
        using connection_type = policies::libfabric::sender;
        using send_early_parcel = HPX_PARCELPORT_LIBFABRIC_HAVE_BOOTSTRAPPING;
        using do_background_work = std::true_type;
        using send_immediate_parcels = std::true_type;
        using is_connectionless = std::false_type;

        static constexpr const char* type() noexcept
        {
            return "libfabric";
        }

        static constexpr const char* pool_name() noexcept
        {
            return "parcel-pool-libfabric";
        }

        static constexpr const char* pool_name_postfix() noexcept
        {
            return "-libfabric";
        }
    };
    /*
namespace policies {
namespace libfabric
{
        uint32_t get_verbs_device_address(const char *devicename, const char *iface,
            char *hostname)
        {
          FUNC_START_DEBUG_MSG
          // Find the address of the I/O link device.
          verbs_device_ptr linkDevice;
          try {
            linkDevice = verbs_device_ptr(new verbs_device(devicename, iface));
          }
          catch (fabric_error& e) {
            LOG_ERROR_MSG("error opening InfiniBand device: " << e.what());
          }
          LOG_DEBUG_MSG("Created InfiniBand device for "
              << linkDevice->get_device_name() << " using interface "
              << linkDevice->get_interface_name());

          std::stringstream temp;
          in_addr_t addr = linkDevice->get_address();
          temp
            << (int)((uint8_t*)&addr)[0] << "."
            << (int)((uint8_t*)&addr)[1] << "."
            << (int)((uint8_t*)&addr)[2] << "."
            << (int)((uint8_t*)&addr)[3] << std::ends;
          strcpy(hostname, temp.str().c_str());
          //
          LOG_DEBUG_MSG("Generated hostname string " << hostname);

          // print device info for debugging
        //  linkDevice->getDeviceInfo(true);
          FUNC_END_DEBUG_MSG
          return (uint32_t)(addr);
        }

}}
*/
}    // namespace hpx::parcelset

#include <hpx/config/warnings_suffix.hpp>
