////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2012-2013 Hartmut Kaiser
//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/config.hpp>

#include <hpx/runtime/agas/detail/bootstrap_locality_namespace.hpp>
#include <hpx/runtime/agas/server/locality_namespace.hpp>
#include <hpx/runtime/parcelset/locality.hpp>
#include <hpx/runtime/naming/name.hpp>

#include <boost/cstdint.hpp>

#include <string>
#include <utility>
#include <vector>

namespace hpx { namespace agas { namespace detail
{
    bootstrap_locality_namespace::bootstrap_locality_namespace(
        server::primary_namespace *primary)
      : server_(primary)
    {}

    naming::address bootstrap_locality_namespace::addr() const
    {
        return naming::address(
            hpx::get_locality(),
            server::locality_namespace::get_component_type(),
            this->ptr()
        );
    }

    naming::id_type bootstrap_locality_namespace::gid() const
    {
        return naming::id_type(
            naming::gid_type(HPX_AGAS_LOCALITY_NS_MSB, HPX_AGAS_LOCALITY_NS_LSB),
            naming::id_type::unmanaged);
    }

    boost::uint32_t bootstrap_locality_namespace::allocate(
        parcelset::endpoints_type const& endpoints
      , boost::uint64_t count
      , boost::uint32_t num_threads
      , naming::gid_type suggested_prefix
        )
    {
        return server_.allocate(endpoints, count, num_threads, suggested_prefix);
    }

    void bootstrap_locality_namespace::free(naming::gid_type locality)
    {
        server_.free(locality);
    }

    std::vector<boost::uint32_t> bootstrap_locality_namespace::localities()
    {
        return server_.localities();
    }

    parcelset::endpoints_type
    bootstrap_locality_namespace::resolve_locality(naming::gid_type locality)
    {
        return server_.resolve_locality(locality);
    }

    boost::uint32_t bootstrap_locality_namespace::get_num_localities()
    {
        return server_.get_num_localities();
    }

    hpx::future<boost::uint32_t> bootstrap_locality_namespace::get_num_localities_async()
    {
        return hpx::make_ready_future(server_.get_num_localities());
    }

    std::vector<boost::uint32_t> bootstrap_locality_namespace::get_num_threads()
    {
        return server_.get_num_threads();
    }

    hpx::future<std::vector<boost::uint32_t> >
    bootstrap_locality_namespace::get_num_threads_async()
    {
        return hpx::make_ready_future(server_.get_num_threads());
    }

    boost::uint32_t bootstrap_locality_namespace::get_num_overall_threads()
    {
        return server_.get_num_overall_threads();
    }

    hpx::future<boost::uint32_t>
    bootstrap_locality_namespace::get_num_overall_threads_async()
    {
        return hpx::make_ready_future(server_.get_num_overall_threads());
    }

    naming::gid_type bootstrap_locality_namespace::statistics_counter(std::string name)
    {
        return server_.statistics_counter(std::move(name));
    }

    void bootstrap_locality_namespace::register_counter_types()
    {
        server::locality_namespace::register_counter_types();
        server::locality_namespace::register_global_counter_types();
    }

    void bootstrap_locality_namespace::register_server_instance(
        boost::uint32_t locality_id)
    {
        HPX_ASSERT(locality_id == 0);
        const char* servicename("locality#0/");
        server_.register_server_instance(servicename);
    }

    void bootstrap_locality_namespace::unregister_server_instance(error_code& ec)
    {
        server_.unregister_server_instance(ec);
    }
}}}
