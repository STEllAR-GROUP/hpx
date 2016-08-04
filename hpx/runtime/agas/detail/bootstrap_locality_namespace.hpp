////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2012-2013 Hartmut Kaiser
//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_AGAS_DETAIL_BOOTSTRAP_LOCALITY_NAMESPACE_HPP)
#define HPX_AGAS_DETAIL_BOOTSTRAP_LOCALITY_NAMESPACE_HPP

#include <hpx/config.hpp>

#include <hpx/lcos/future.hpp>
#include <hpx/runtime/agas_fwd.hpp>
#include <hpx/runtime/agas/server/locality_namespace.hpp>
#include <hpx/runtime/agas/locality_namespace.hpp>
#include <hpx/runtime/parcelset/locality.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/naming/address.hpp>

#include <boost/cstdint.hpp>

#include <string>
#include <vector>

namespace hpx { namespace agas { namespace detail
{
    struct bootstrap_locality_namespace : locality_namespace
    {
        bootstrap_locality_namespace(server::primary_namespace *primary);

        naming::address::address_type ptr() const
        {
            return reinterpret_cast<naming::address::address_type>(&server_);
        }
        naming::address addr() const;
        naming::id_type gid() const;

        boost::uint32_t allocate(
            parcelset::endpoints_type const& endpoints
          , boost::uint64_t count
          , boost::uint32_t num_threads
          , naming::gid_type suggested_prefix
            );

        void free(naming::gid_type locality);

        std::vector<boost::uint32_t> localities();

        parcelset::endpoints_type resolve_locality(naming::gid_type locality);

        boost::uint32_t get_num_localities();
        hpx::future<boost::uint32_t> get_num_localities_async();

        std::vector<boost::uint32_t> get_num_threads();
        hpx::future<std::vector<boost::uint32_t> > get_num_threads_async();

        boost::uint32_t get_num_overall_threads();
        hpx::future<boost::uint32_t> get_num_overall_threads_async();

        naming::gid_type statistics_counter(std::string name);

        void register_counter_types();

        void register_server_instance(boost::uint32_t locality_id);

        void unregister_server_instance(error_code& ec);

    private:
        server::locality_namespace server_;
    };
}}}

#endif
