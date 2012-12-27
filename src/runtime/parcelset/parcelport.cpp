//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/parcelset/tcp/parcelport.hpp>
#if defined(HPX_USE_SHMEM_PARCELPORT)
#  include <hpx/runtime/parcelset/shmem/parcelport.hpp>
#endif
#include <hpx/util/io_service_pool.hpp>
#include <hpx/util/runtime_configuration.hpp>
#include <hpx/exception.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

namespace hpx { namespace parcelset
{
    boost::shared_ptr<parcelport> parcelport::create(connection_type type,
        util::runtime_configuration const& cfg,
        HPX_STD_FUNCTION<void(std::size_t, char const*)> const& on_start_thread,
        HPX_STD_FUNCTION<void()> const& on_stop_thread)
    {
        switch(type) {
        case connection_tcpip:
            return boost::make_shared<parcelset::tcp::parcelport>(
                cfg, on_start_thread, on_stop_thread);

        case connection_shmem:
            {
#if defined(HPX_USE_SHMEM_PARCELPORT)
                // Create shmem based parcelport only if allowed by the 
                // configuration info.
                std::string enable_shmem = 
                    cfg.get_entry("hpx.parcel.use_shmem_parcelport", "0");

                if (boost::lexical_cast<int>(enable_shmem)) 
                {
                    return boost::make_shared<parcelset::shmem::parcelport>(
                        cfg, on_start_thread, on_stop_thread);
                }
#endif
                HPX_THROW_EXCEPTION(bad_parameter, "parcelport::create",
                    "unsupported connection type 'connection_shmem'");
            }
            break;

        case connection_portals4:
            HPX_THROW_EXCEPTION(bad_parameter, "parcelport::create",
                "unsupported connection type 'connection_portals4'");
            break;

        default:
            HPX_THROW_EXCEPTION(bad_parameter, "parcelport::create",
                "unknown connection type");
            break;
        }

        return boost::shared_ptr<parcelport>();
    }
}}

