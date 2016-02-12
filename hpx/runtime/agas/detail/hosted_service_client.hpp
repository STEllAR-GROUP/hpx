//  Copyright (c) 2016 Hartmut Kaiser
//  Copyright (c) 2016 Parsa Amini
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_AGAS_CLIENT_HOSTED_FEB_05_2016_114AM)
#define HPX_AGAS_CLIENT_HOSTED_FEB_05_2016_114AM

#include <hpx/runtime/agas/detail/agas_service_client.hpp>
#include <hpx/runtime/agas/component_namespace.hpp>
#include <hpx/runtime/agas/locality_namespace.hpp>
#include <hpx/runtime/agas/server/component_namespace.hpp>
#include <hpx/runtime/agas/server/locality_namespace.hpp>
#include <hpx/runtime/agas/server/primary_namespace.hpp>
#include <hpx/runtime/agas/server/symbol_namespace.hpp>

namespace hpx { namespace agas { namespace detail
{
    struct hosted_data_type
    { // {{{
        hosted_data_type()
          : primary_ns_server_()
          , symbol_ns_server_()
        {}

        void register_counter_types()
        {
            server::primary_namespace::register_counter_types();
            server::primary_namespace::register_global_counter_types();
            server::symbol_namespace::register_counter_types();
            server::symbol_namespace::register_global_counter_types();
        }

        void register_server_instance(char const* servicename
          , boost::uint32_t locality_id)
        {
            primary_ns_server_.register_server_instance(servicename, locality_id);
            symbol_ns_server_.register_server_instance(servicename, locality_id);
        }

        void unregister_server_instance(error_code& ec)
        {
            primary_ns_server_.unregister_server_instance(ec);
            if (!ec) symbol_ns_server_.unregister_server_instance(ec);
        }

        locality_namespace locality_ns_;
        component_namespace component_ns_;

        server::primary_namespace primary_ns_server_;
        server::symbol_namespace symbol_ns_server_;
    }; // }}}

    ///////////////////////////////////////////////////////////////////////////
    struct hosted_service_client : agas_service_client
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
            HPX_ASSERT(false);      // shouldn't ever be called
            return 0;
        }

        naming::address::address_type get_locality_ns_ptr() const
        {
            HPX_ASSERT(false);      // shouldn't ever be called
            return 0;
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
        hosted_data_type data_;
    };
}}}

#endif

