//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/parcelset/tcp/parcelport.hpp>
#include <hpx/runtime/parcelset/shmem/parcelport.hpp>
#include <hpx/util/io_service_pool.hpp>
#include <hpx/util/runtime_configuration.hpp>
#include <hpx/exception.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

namespace hpx { namespace parcelset
{
    boost::shared_ptr<parcelport> parcelport::create(connection_type type,
        util::io_service_pool& pool, util::runtime_configuration const& cfg)
    {
        switch(type) {
        case connection_tcpip:
            return boost::make_shared<parcelset::tcp::parcelport>(pool, cfg);

        case connection_shmem:
            return boost::make_shared<parcelset::shmem::parcelport>(pool, cfg);

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

