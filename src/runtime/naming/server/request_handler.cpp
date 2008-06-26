//  Copyright (c) 2007-2008 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <string>
#include <boost/lexical_cast.hpp>
#include <boost/fusion/include/at.hpp>

#include <hpx/util/logging.hpp>
#include <hpx/runtime/naming/server/reply.hpp>
#include <hpx/runtime/naming/server/request.hpp>
#include <hpx/runtime/naming/server/request_handler.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace naming { namespace server 
{
    ///////////////////////////////////////////////////////////////////////////
    request_handler::request_handler()
      : totals_(command_lastcommand)
    {
    }

    request_handler::~request_handler()
    {
    }

    ///////////////////////////////////////////////////////////////////////////
    void request_handler::handle_getprefix(request const& req, reply& rep)
    {
        try {
            mutex_type::scoped_lock l(mtx_);
            site_prefix_map_type::iterator it = 
                site_prefixes_.find(req.get_site());
            if (it != site_prefixes_.end()) {
                // The real prefix has to be used as the 32 most 
                // significant bits of global id's

                // existing entry
                rep = reply(repeated_request, command_getprefix, 
                    get_id_from_prefix((*it).second.first)); 
            }
            else {
                // insert this prefix as being mapped to the given locality
                boost::uint32_t prefix = site_prefixes_.size() + 1;
                naming::id_type id = get_id_from_prefix(prefix);

                // start assigning ids with the second block of 64Bit numbers only
                naming::id_type lower_id (id.get_msb() + 1, 0);
                site_prefixes_.insert(
                    site_prefix_map_type::value_type(req.get_site(), 
                        std::make_pair(prefix, lower_id)));

                // now, bind this prefix to the locality address allowing to 
                // send parcels to the memory of a locality
                registry_type::iterator it = registry_.find(id);
                if (it != registry_.end()) {
                    // this shouldn't happen
                    rep = reply(command_getprefix, no_success, 
                        "prefix is already bound to local address");
                }
                else {
                    // the zero as last parameter means 'all' lsb gids
                    registry_.insert(registry_type::value_type(id, 
                        registry_data_type(address(req.get_site()), 1, 0)));
                }

                // The real prefix has to be used as the 32 most 
                // significant bits of global id's

                // created new entry
                rep = reply(success, command_getprefix, id);
            }
        }
        catch (std::bad_alloc) {
            rep = reply(command_getprefix, out_of_memory);
        }
        catch (...) {
            rep = reply(command_getprefix, internal_server_error);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    void request_handler::handle_getprefixes(request const& req, reply& rep)
    {
        try {
            mutex_type::scoped_lock l(mtx_);

            std::vector<boost::uint32_t> prefixes;
            prefixes.reserve(site_prefixes_.size());

            site_prefix_map_type::iterator end = site_prefixes_.end();
            for (site_prefix_map_type::iterator it = site_prefixes_.begin();
                 it != end; ++it)
            {
                prefixes.push_back((*it).second.first);
            }

            rep = reply(prefixes, prefixes.empty() ? no_success : success); 
        }
        catch (std::bad_alloc) {
            rep = reply(command_getprefixes, out_of_memory);
        }
        catch (...) {
            rep = reply(command_getprefixes, internal_server_error);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    void request_handler::handle_getidrange(request const& req, reply& rep)
    {
        try {
            mutex_type::scoped_lock l(mtx_);
            site_prefix_map_type::iterator it = 
                site_prefixes_.find(req.get_site());
            if (it != site_prefixes_.end()) {
                // The real prefix has to be used as the 32 most 
                // significant bits of global id's

                // generate the new id range
                naming::id_type lower ((*it).second.second + 1);
                naming::id_type upper (lower + (req.get_count() - 1));

                if (upper.get_msb() != lower.get_msb()) {
                    if ((lower.get_msb() & ~0xFFFFFFFF) == 0xFFFFFFFF)
                    {
                        rep = reply(command_getprefix, internal_server_error,
                            "global ids have been exhausted");
                        return;
                    }
                    lower = naming::id_type(upper.get_msb(), 0);
                    upper = lower + (req.get_count() - 1);
                }

                // store the new lower bound
                (*it).second.second = upper;

                // existing entry
                rep = reply(repeated_request, command_getidrange, lower, upper); 
            }
            else {
                // insert this prefix as being mapped to the given locality
                boost::uint32_t prefix = (boost::uint32_t)site_prefixes_.size() + 1;
                naming::id_type id = get_id_from_prefix(prefix);

                // start assigning ids with the second block of 64Bit numbers only
                naming::id_type lower_id (id.get_msb() + 1, 0);

                std::pair<site_prefix_map_type::iterator, bool> p =
                    site_prefixes_.insert(
                        site_prefix_map_type::value_type(req.get_site(), 
                            std::make_pair(prefix, lower_id)));

                // now, bind this prefix to the locality address allowing to 
                // send parcels to the memory of a locality
                registry_type::iterator it = registry_.find(id);
                if (it != registry_.end()) {
                    // this shouldn't happen
                    rep = reply(command_getprefix, no_success, 
                        "prefix is already bound to local address");
                }
                else {
                    registry_.insert(registry_type::value_type(id, 
                        registry_data_type(address(req.get_site()), 1, 0)));
                }

                // generate the new id range
                naming::id_type lower = lower_id + 1;
                naming::id_type upper = lower + (req.get_count() - 1);
                
                // store the new lower bound
                (*p.first).second.second = upper;
                
                // created new entry
                rep = reply(success, command_getidrange, lower, upper);
            }
        }
        catch (std::bad_alloc) {
            rep = reply(command_getprefix, out_of_memory);
        }
        catch (...) {
            rep = reply(command_getprefix, internal_server_error);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    void request_handler::handle_bind_range(request const& req, reply& rep)
    {
        try {
            error s = no_success;
            std::string str;
            {
                using boost::fusion::at_c;

                mutex_type::scoped_lock l(mtx_);
                registry_type::iterator it = registry_.lower_bound(req.get_id());

                if (it != registry_.end()) {
                    if ((*it).first == req.get_id()) {
                        // update existing bindings
                        if (at_c<1>((*it).second) != req.get_count()) 
                        {
                            // this is an error since we can't change block sizes 
                            s = bad_parameter;
                            str = "can't change block size or number of gids per "
                                  "object of existing binding";
                        }
                        else {
                            // store the new address and offsets
                            at_c<0>((*it).second) = req.get_address();
                            at_c<2>((*it).second) = req.get_offset();
                        }
                    }
                    else if (it != registry_.begin()) {
                        --it;
                        if ((*it).first + at_c<1>((*it).second) > req.get_id()) {
                            // the previous range covers the new id
                            s = bad_parameter;
                            str = "the new global id is contained in an existing range";
                        }
                        else {
                            // create new bindings
                            create_new_binding(req, s, str);
                        }
                    }
                    else {
                        // create new bindings, the existing ranges are larger 
                        // than the new global id
                        create_new_binding(req, s, str);
                    }
                }
                else {
                    if (!registry_.empty()) {
                        --it;
                        if ((*it).first + at_c<1>((*it).second) > req.get_id()) {
                            // the previous range covers the new id
                            s = bad_parameter;
                            str = "the new global id is contained in an existing range";
                        }
                        else {
                            // create new bindings
                            create_new_binding(req, s, str);
                        }
                    }
                    else {
                        // create new bindings
                        create_new_binding(req, s, str);
                    }
                }
            }
            rep = reply(command_bind_range, s, str.empty() ? NULL : str.c_str());
        }
        catch (std::bad_alloc) {
            rep = reply(command_bind_range, out_of_memory);
        }
        catch (...) {
            rep = reply(command_bind_range, internal_server_error);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    void request_handler::handle_unbind_range(request const& req, reply& rep)
    {
        try {
            error s = no_success;
            std::string str;
            {
                using boost::fusion::at_c;

                mutex_type::scoped_lock l(mtx_);
                registry_type::iterator it = registry_.find(req.get_id());
                if (it != registry_.end()) {
                    if (at_c<1>((*it).second) != req.get_count()) {
                        // this is an error since we can't use a different 
                        // block size while unbinding
                        s = bad_parameter;
                        str = "block sizes must match";
                    }
                    else {
                        rep = reply(command_unbind_range, at_c<0>((*it).second));
                        registry_.erase(it);
                        return;
                    }
                }
            }
            rep = reply(command_unbind_range, s, 
                str.empty() ? NULL : str.c_str());
        }
        catch (std::bad_alloc) {
            rep = reply(command_unbind_range, out_of_memory);
        }
        catch (...) {
            rep = reply(command_unbind_range, internal_server_error);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    void request_handler::handle_resolve(request const& req, reply& rep)
    {
        try {
            using boost::fusion::at_c;

            mutex_type::scoped_lock l(mtx_);
            registry_type::iterator it = registry_.lower_bound(req.get_id());
            if (it != registry_.end()) {
                if ((*it).first == req.get_id()) {
                    // found the exact match in the registry
                    rep = reply(command_resolve, at_c<0>((*it).second));
                }
                else if (it != registry_.begin()) {
                    --it;
                    if ((*it).first + at_c<1>((*it).second) > req.get_id()) {
                        // the previous range covers the given global id

                        // the only limitation while binding blocks of global 
                        // ids is that these have to have identical msb's
                        if (req.get_id().get_msb() != (*it).first.get_msb()) {
                            // no existing range covers the given global id
                            rep = reply(command_resolve, internal_server_error,
                                "msb's of global ids should match");
                        }
                        else {
                            // calculate the local address corresponding to the 
                            // given global id
                            naming::address addr (at_c<0>((*it).second));
                            boost::uint64_t gid_offset = 
                                req.get_id().get_lsb() - (*it).first.get_lsb();
                            addr.address_ += gid_offset * at_c<2>((*it).second);
                            rep = reply(command_resolve, addr);
                        }
                    }
                    else {
                        // no existing range covers the given global id
                        rep = reply(command_resolve, no_success);
                    }
                }
                else {
                    // all existing entries are larger than the given global id
                    rep = reply(command_resolve, no_success);
                }
            }
            else if (!registry_.empty()) {
                --it;
                if ((*it).first + at_c<1>((*it).second) >= req.get_id()) {
                    // the previous range covers the id to resolve
                    naming::address addr (at_c<0>((*it).second));
                    boost::uint64_t gid_offset = 
                        req.get_id().get_lsb() - (*it).first.get_lsb();
                    addr.address_ += gid_offset * at_c<2>((*it).second);
                    rep = reply(command_resolve, addr);
                }
                else {
                    rep = reply(command_resolve, no_success);
                }
            }
            else {
                rep = reply(command_resolve, no_success);
            }
        }
        catch (std::bad_alloc) {
            rep = reply(command_resolve, out_of_memory);
        }
        catch (...) {
            rep = reply(command_resolve, internal_server_error);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    void request_handler::handle_queryid(request const& req, reply& rep)
    {
        try {
            mutex_type::scoped_lock l(mtx_);
            ns_registry_type::iterator it = ns_registry_.find(req.get_name());
            if (it != ns_registry_.end()) 
                rep = reply(command_queryid, (*it).second);
            else
                rep = reply(command_queryid, no_success);
        }
        catch (std::bad_alloc) {
            rep = reply(command_queryid, out_of_memory);
        }
        catch (...) {
            rep = reply(command_queryid, internal_server_error);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    void request_handler::handle_registerid(request const& req, reply& rep)
    {
        try {
            error s = no_success;
            {
                mutex_type::scoped_lock l(mtx_);
                ns_registry_type::iterator it = ns_registry_.find(req.get_name());
                if (it != ns_registry_.end())
                    (*it).second = req.get_id();
                else {
                    ns_registry_.insert(ns_registry_type::value_type(
                        req.get_name(), req.get_id()));
                    s = success;    // created new entry
                }
            }
            rep = reply(command_registerid, s);
        }
        catch (std::bad_alloc) {
            rep = reply(command_registerid, out_of_memory);
        }
        catch (...) {
            rep = reply(command_registerid, internal_server_error);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    void request_handler::handle_unregisterid(request const& req, reply& rep)
    {
        try {
            error s = no_success;
            {
                mutex_type::scoped_lock l(mtx_);
                ns_registry_type::iterator it = ns_registry_.find(req.get_name());
                if (it != ns_registry_.end()) {
                    ns_registry_.erase(it);
                    s = success;
                }
            }
            rep = reply(command_unregisterid, s);
        }
        catch (std::bad_alloc) {
            rep = reply(command_unregisterid, out_of_memory);
        }
        catch (...) {
            rep = reply(command_unregisterid, internal_server_error);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
#if BOOST_VERSION < 103600
    double extract_count(std::pair<double, std::size_t> const& p)
    {
        return p.second;
    }
#endif
    
    void request_handler::handle_statistics_count(request const& req, reply& rep)
    {
        try {
#if BOOST_VERSION >= 103600
            rep = reply(command_statistics_count, totals_, boost::accumulators::count);
#else
            rep = reply(command_statistics_count, totals_, extract_count);
#endif
        }
        catch (std::bad_alloc) {
            rep = reply(command_statistics_count, out_of_memory);
        }
        catch (...) {
            rep = reply(command_statistics_count, internal_server_error);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
#if BOOST_VERSION < 103600
    double extract_mean(std::pair<double, std::size_t> const& p)
    {
        return (0 != p.second) ? p.first / p.second : 0.0;
    }
#endif
    
    void request_handler::handle_statistics_mean(request const& req, reply& rep)
    {
        try {
#if BOOST_VERSION >= 103600
            rep = reply(command_statistics_mean, totals_, boost::accumulators::mean);
#else
            rep = reply(command_statistics_mean, totals_, extract_mean);
#endif
        }
        catch (std::bad_alloc) {
            rep = reply(command_statistics_mean, out_of_memory);
        }
        catch (...) {
            rep = reply(command_statistics_mean, internal_server_error);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
#if BOOST_VERSION > 103600
    double request_handler::extract_moment2(accumulator_set_type const& p)
    {
        return boost::accumulators::extract::moment<2>(p);
    }
#else
    double extract_moment2(std::pair<double, std::size_t> const& p)
    {
        return 0.0;   // not implemented yet
    }
#endif

    void request_handler::handle_statistics_moment2(request const& req, reply& rep)
    {
        try {
#if BOOST_VERSION >= 103600
            rep = reply(command_statistics_moment2, totals_, 
                &request_handler::extract_moment2);
#else
            rep = reply(command_statistics_moment2, totals_, extract_moment2);
#endif
        }
        catch (std::bad_alloc) {
            rep = reply(command_statistics_moment2, out_of_memory);
        }
        catch (...) {
            rep = reply(command_statistics_moment2, internal_server_error);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    void request_handler::handle_request(request const& req, reply& rep)
    {
        LDGAS_(info) << "request: " << req;
        switch (req.get_command()) {
        case command_getprefix:
            handle_getprefix(req, rep);
            break;

        case command_getprefixes:
            handle_getprefixes(req, rep);
            break;

        case command_getidrange:
            handle_getidrange(req, rep);
            break;

        case command_bind_range:
            handle_bind_range(req, rep);
            break;

        case command_unbind_range:
            handle_unbind_range(req, rep);
            break;

        case command_resolve:
            handle_resolve(req, rep);
            break;

        case command_queryid:
            handle_queryid(req, rep);
            break;

        case command_registerid:
            handle_registerid(req, rep);
            break;

        case command_unregisterid:
            handle_unregisterid(req, rep);
            break;

        case command_statistics_count:
            handle_statistics_count(req, rep);
            break;

        case command_statistics_mean:
            handle_statistics_mean(req, rep);
            break;

        case command_statistics_moment2:
            handle_statistics_moment2(req, rep);
            break;

        default:
            rep = reply(bad_request);
            break;
        }

        if (rep.get_status() != success && rep.get_status() != repeated_request) {
            LDGAS_(error) << "response: " << rep;
        }
        else {
            LDGAS_(info) << "response: " << rep;
        }
    }

}}}  // namespace hpx::naming::server

