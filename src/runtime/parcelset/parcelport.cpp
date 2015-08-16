//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2013-2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This is needed to make everything work with the Intel MPI library header
#include <hpx/config/defines.hpp>

#include <hpx/hpx_fwd.hpp>

#include <hpx/runtime/parcelset/parcelport.hpp>
#include <hpx/util/io_service_pool.hpp>
#include <hpx/util/runtime_configuration.hpp>
#include <hpx/util/safe_lexical_cast.hpp>
#include <hpx/exception.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

namespace hpx { namespace parcelset
{
    ///////////////////////////////////////////////////////////////////////////
    parcelport::parcelport(util::runtime_configuration const& ini, locality const & here,
            std::string const& type)
      : parcels_(),
        here_(here),
        max_inbound_message_size_(ini.get_max_inbound_message_size()),
        max_outbound_message_size_(ini.get_max_outbound_message_size()),
        allow_array_optimizations_(true),
        allow_zero_copy_optimizations_(true),
        enable_security_(false),
        async_serialization_(false),
        enable_parcel_handling_(true),
        priority_(hpx::util::get_entry_as<int>(ini, "hpx.parcel." + type + ".priority", "0")),
        type_(type)
    {
        std::string key("hpx.parcel.");
        key += type;

        if (hpx::util::get_entry_as<int>(ini, key + ".array_optimization", "1") == 0) {
            allow_array_optimizations_ = false;
            allow_zero_copy_optimizations_ = false;
        }
        else {
            if (hpx::util::get_entry_as<int>(ini, key + ".zero_copy_optimization", "1") == 0)
                allow_zero_copy_optimizations_ = false;
        }

        if(hpx::util::get_entry_as<int>(ini, key + ".enable_security", "0") != 0)
        {
            enable_security_ = true;
        }

        if(hpx::util::get_entry_as<int>(ini, key + ".async_serialization", "0") != 0)
        {
            async_serialization_ = true;
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    boost::uint64_t HPX_EXPORT get_max_inbound_size(parcelport& pp)
    {
        return pp.get_max_inbound_message_size();
    }
}}

