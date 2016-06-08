//  Copyright (c) 2016 Hartmut Kaiser
//  Copyright (c) 2016 Parsa Amini
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_AGAS_CLIENT_BOOTSTRAP_FEB_05_2016_114AM)
#define HPX_AGAS_CLIENT_BOOTSTRAP_FEB_05_2016_114AM

#include <hpx/runtime/agas/component_namespace.hpp>
#include <hpx/runtime/agas/detail/agas_service_client.hpp>
#include <hpx/runtime/agas/locality_namespace.hpp>
#include <hpx/runtime/agas/primary_namespace.hpp>
#include <hpx/runtime/agas/server/component_namespace.hpp>
#include <hpx/runtime/agas/server/locality_namespace.hpp>
#include <hpx/runtime/agas/server/primary_namespace.hpp>
#include <hpx/runtime/agas/server/symbol_namespace.hpp>
#include <hpx/runtime/agas/symbol_namespace.hpp>

#include <string>
#include <vector>

namespace hpx { namespace agas { namespace detail
{
    struct bootstrap_data_type
    { // {{{
        bootstrap_data_type()
          : primary_ns_server_()
          , locality_ns_server_(&primary_ns_server_)
          , component_ns_server_()
          , symbol_ns_server_()
        {}

        void register_counter_types()
        {
            server::locality_namespace::register_counter_types();
            server::locality_namespace::register_global_counter_types();
            server::primary_namespace::register_counter_types();
            server::primary_namespace::register_global_counter_types();
            server::component_namespace::register_counter_types();
            server::component_namespace::register_global_counter_types();
            server::symbol_namespace::register_counter_types();
            server::symbol_namespace::register_global_counter_types();
        }

        void register_server_instance(char const* servicename)
        {
            locality_ns_server_.register_server_instance(servicename);
            primary_ns_server_.register_server_instance(servicename);
            component_ns_server_.register_server_instance(servicename);
            symbol_ns_server_.register_server_instance(servicename);
        }

        void unregister_server_instance(error_code& ec)
        {
            locality_ns_server_.unregister_server_instance(ec);
            if (!ec) primary_ns_server_.unregister_server_instance(ec);
            if (!ec) component_ns_server_.unregister_server_instance(ec);
            if (!ec) symbol_ns_server_.unregister_server_instance(ec);
        }

        server::primary_namespace primary_ns_server_;
        server::locality_namespace locality_ns_server_;
        server::component_namespace component_ns_server_;
        server::symbol_namespace symbol_ns_server_;
    }; // }}}

    ///////////////////////////////////////////////////////////////////////////
    struct bootstrap_service_client : agas_service_client
    {
        ///////////////////////////////////////////////////////////////////////
        naming::address::address_type get_primary_ns_ptr() const
        {
            return reinterpret_cast<naming::address::address_type>(
                &data_.primary_ns_server_);
        }

        naming::address::address_type get_symbol_ns_ptr() const
        {
            return reinterpret_cast<naming::address::address_type>(
                &data_.symbol_ns_server_);
        }

        naming::address::address_type get_component_ns_ptr() const
        {
            return reinterpret_cast<naming::address::address_type>(
                &data_.component_ns_server_);
        }

        naming::address::address_type get_locality_ns_ptr() const
        {
            return reinterpret_cast<naming::address::address_type>(
                &data_.locality_ns_server_);
        }

        ///////////////////////////////////////////////////////////////////////
        void set_local_locality(
            naming::gid_type const& g);

        response service(
            request const& req
          , threads::thread_priority priority
          , error_code& ec);

        void register_counter_types();

        void register_server_instance(
            boost::uint32_t locality_id);

        bool unregister_server(
            request const& req
          , threads::thread_priority priority
          , error_code& ec);

        response service_primary(
            request const& req
          , error_code& ec);

        std::vector<response> service_primary_bulk(
            std::vector<request> const& reqs
          , error_code& ec);

        response service_component(
            request const& req
          , threads::thread_priority priority
          , error_code& ec);

        response service_locality(
            request const& req
          , threads::thread_priority priority
            , error_code& ec);

        response service_symbol(
            request const& req
            , threads::thread_priority priority
            , std::string const& name
            , error_code& ec);

        future<parcelset::endpoints_type> get_endpoints(
            request const& req
          , threads::thread_priority priority
          , error_code& ec);

    private:
        bootstrap_data_type data_;
    };
}}}

#endif

