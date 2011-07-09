//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>

#if HPX_AGAS_VERSION <= 0x10

#include <hpx/runtime/naming/server/request.hpp>
#include <hpx/runtime/naming/server/reply.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/runtime/naming/bulk_resolver_client.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace naming 
{
    bulk_resolver_client::bulk_resolver_client(resolver_client& resolver) 
      : resolver_(resolver)
    {}

    ///////////////////////////////////////////////////////////////////////////
    int bulk_resolver_client::get_prefix(locality const& l, bool self) 
    {
        // create request
        requests_.push_back(server::request(
            self ? server::command_getprefix : server::command_getprefix_for_site, 
            l, resolver_.is_console()));
        return requests_.size()-1;
    }

    bool bulk_resolver_client::get_prefix(std::size_t index, gid_type& prefix, 
        error_code& ec) const
    {
        if (index >= responses_.size()) {
            HPX_THROWS_IF(ec, bad_parameter, 
                "bulk_resolver_client::get_prefix", "index is out of bounds");
            return false;
        }

        server::reply const& rep = responses_[index];
        if (rep.get_command() != server::command_getprefix &&
            rep.get_command() != server::command_getprefix_for_site) 
        {
            HPX_THROWS_IF(ec, bad_parameter, 
                "bulk_resolver_client::get_prefix", "server command mismatch");
            return false;
        }

        error r = rep.get_status();
        if (r != success && r != repeated_request) {
            HPX_THROWS_IF(ec, r, 
                "bulk_resolver_client::get_prefix", rep.get_error());
        }

        prefix = rep.get_prefix();
        if (&ec != &throws)
            ec = make_error_code(r, rep.get_error());
        return r == success;
    }

    ///////////////////////////////////////////////////////////////////////////
    int bulk_resolver_client::resolve(gid_type const& id) 
    {
        // create request
        requests_.push_back(server::request(server::command_resolve, id));
        return requests_.size()-1;
    }

    bool bulk_resolver_client::resolve(std::size_t index, address& addr, 
        error_code& ec) const
    {
        if (index >= responses_.size()) {
            HPX_THROWS_IF(ec, bad_parameter, 
                "bulk_resolver_client::resolve", "index is out of bounds");
            return false;
        }

        server::reply const& rep = responses_[index];
        if (rep.get_command() != server::command_resolve) {
            HPX_THROWS_IF(ec, bad_parameter, 
                "bulk_resolver_client::resolve", "server command mismatch");
            return false;
        }

        error r = rep.get_status();
        if (r != success && r != no_success) {
            HPX_THROWS_IF(ec, r, 
                "bulk_resolver_client::resolve", rep.get_error());
        }

        addr = rep.get_address();
        if (&ec != &throws)
            ec = make_error_code(r, rep.get_error());
        return r == success;
    }

    ///////////////////////////////////////////////////////////////////////////
    int bulk_resolver_client::incref(gid_type const& id, boost::uint32_t credits)
    {
        // create request
        requests_.push_back(server::request(server::command_incref, id, credits));
        return requests_.size()-1;
    }

    int bulk_resolver_client::incref(std::size_t index, error_code& ec) const
    {
        if (index >= responses_.size()) {
            HPX_THROWS_IF(ec, bad_parameter, 
                "bulk_resolver_client::incref", "index is out of bounds");
            return false;
        }

        server::reply const& rep = responses_[index];
        if (rep.get_command() != server::command_incref) {
            HPX_THROWS_IF(ec, bad_parameter, 
                "bulk_resolver_client::incref", "server command mismatch");
            return false;
        }

        error r = rep.get_status();
        if (r != success) {
            HPX_THROWS_IF(ec, r, 
                "bulk_resolver_client::incref", rep.get_error());
        }

        if (&ec != &throws)
            ec = make_success_code();
        return rep.get_refcnt();
    }

    ///////////////////////////////////////////////////////////////////////////
    bool bulk_resolver_client::execute(error_code& ec) 
    {
        responses_.clear();
        if (requests_.empty())
            return true;        // nothing to do

        return resolver_.execute(requests_, responses_, ec);
    }
}}

#endif
