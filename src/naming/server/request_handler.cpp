//  Copyright (c) 2007-2008 Hartmut Kaiser
//
//  Parts of this code were taken from the Boost.Asio library
//  Copyright (c) 2003-2007 Christopher M. Kohlhoff (chris at kohlhoff dot com)
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <string>
#include <boost/lexical_cast.hpp>
#include <boost/fusion/include/at.hpp>

#include <hpx/util/dgas_logging.hpp>
#include <hpx/naming/server/reply.hpp>
#include <hpx/naming/server/request.hpp>
#include <hpx/naming/server/request_handler.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace naming { namespace server 
{
    ///////////////////////////////////////////////////////////////////////////
    //  Handle conversion to/from prefix
    inline id_type get_id_from_prefix(boost::uint32_t prefix)
    {
        return id_type(boost::uint64_t(prefix) << 32, 0);
    }
    
    inline boost::uint32_t get_prefix_from_id(id_type id)
    {
        return boost::uint32_t(id.get_msb() >> 32);
    }
    
    inline bool is_prefix_only(id_type id)
    {
        return (id.get_msb() & 0xFFFFFFFFFFFF) ? false : true;
    }
    
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
                site_prefixes_.insert(
                    site_prefix_map_type::value_type(req.get_site(), 
                        std::make_pair(prefix, id)));

                // now, bind this prefix to the locality address allowing to 
                // send parcels to a locality
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
                naming::id_type lower = (*it).second.second + 1;
                naming::id_type upper = lower + range_delta;
                
                // store the new lower bound
                (*it).second.second = upper;

                // existing entry
                rep = reply(repeated_request, command_getidrange, lower, upper); 
            }
            else {
                // insert this prefix as being mapped to the given locality
                boost::uint32_t prefix = (boost::uint32_t)site_prefixes_.size() + 1;
                naming::id_type id = get_id_from_prefix(prefix);
                std::pair<site_prefix_map_type::iterator, bool> p =
                    site_prefixes_.insert(
                        site_prefix_map_type::value_type(req.get_site(), 
                            std::make_pair(prefix, id)));

                // now, bind this prefix to the locality address allowing to 
                // send parcels to a locality
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
                naming::id_type lower = id + 1;
                naming::id_type upper = lower + range_delta;
                
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
    void request_handler::handle_bind(request const& req, reply& rep)
    {
        try {
            error s = no_success;
            {
                mutex_type::scoped_lock l(mtx_);
                registry_type::iterator it = registry_.find(req.get_id());
                if (it != registry_.end())
                    boost::fusion::at_c<0>((*it).second) = req.get_address();
                else {
                    registry_.insert(registry_type::value_type(req.get_id(), 
                        registry_data_type(req.get_address(), 1, 0)));
                    s = success;    // created new entry
                }
            }
            rep = reply(command_bind, s);
        }
        catch (std::bad_alloc) {
            rep = reply(command_bind, out_of_memory);
        }            
        catch (...) {
            rep = reply(command_bind, internal_server_error);
        }            
    }

    ///////////////////////////////////////////////////////////////////////////
    void request_handler::handle_unbind(request const& req, reply& rep)
    {
        try {
            error s = no_success;
            {
                mutex_type::scoped_lock l(mtx_);
                registry_type::iterator it = registry_.find(req.get_id());
                if (it != registry_.end()) {
                    registry_.erase(it);
                    s = success;
                }
            }
            rep = reply(command_unbind, s);
        }
        catch (std::bad_alloc) {
            rep = reply(command_unbind, out_of_memory);
        }            
        catch (...) {
            rep = reply(command_unbind, internal_server_error);
        }            
    }

    ///////////////////////////////////////////////////////////////////////////
    void request_handler::handle_bind_range(request const& req, reply& rep)
    {
        try {
            error s = no_success;
//             {
//                 mutex_type::scoped_lock l(mtx_);
//                 registry_type::iterator it = registry_.find(req.get_id());
//                 if (it != registry_.end())
//                     boost::fusion::at_c<0>((*it).second) = req.get_address();
//                 else {
//                     registry_.insert(registry_type::value_type(req.get_id(), 
//                         registry_data_type(req.get_address(), 
//                              req.get_count(), req.get_offset())));
//                     s = success;    // created new entry
//                 }
//             }
            rep = reply(command_bind, s);
        }
        catch (std::bad_alloc) {
            rep = reply(command_bind, out_of_memory);
        }            
        catch (...) {
            rep = reply(command_bind, internal_server_error);
        }            
    }

    ///////////////////////////////////////////////////////////////////////////
    void request_handler::handle_unbind_range(request const& req, reply& rep)
    {
        try {
            error s = no_success;
//             {
//                 mutex_type::scoped_lock l(mtx_);
//                 registry_type::iterator it = registry_.find(req.get_id());
//                 if (it != registry_.end()) {
//                     registry_.erase(it);
//                     s = success;
//                 }
//             }
            rep = reply(command_unbind, s);
        }
        catch (std::bad_alloc) {
            rep = reply(command_unbind, out_of_memory);
        }            
        catch (...) {
            rep = reply(command_unbind, internal_server_error);
        }            
    }

    ///////////////////////////////////////////////////////////////////////////
    void request_handler::handle_resolve(request const& req, reply& rep)
    {
        try {
            mutex_type::scoped_lock l(mtx_);
            registry_type::iterator it = registry_.find(req.get_id());
            if (it != registry_.end()) {
                rep = reply(command_resolve, boost::fusion::at_c<0>((*it).second));
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
#if BOOST_VERSION < 103600
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
                &boost::accumulators::extract::moment<2, accumulator_set_type>);
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
            
        case command_getidrange:
            handle_getidrange(req, rep);
            break;
            
        case command_bind:
            handle_bind(req, rep);
            break;
            
        case command_unbind:
            handle_unbind(req, rep);
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

