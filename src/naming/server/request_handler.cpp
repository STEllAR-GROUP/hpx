//  Copyright (c) 2007-2008 Hartmut Kaiser
//
//  Parts of this code were taken from the Boost.Asio library
//  Copyright (c) 2003-2007 Christopher M. Kohlhoff (chris at kohlhoff dot com)
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <string>
#include <boost/lexical_cast.hpp>

#include <hpx/naming/server/reply.hpp>
#include <hpx/naming/server/request.hpp>
#include <hpx/naming/server/request_handler.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace naming { namespace server 
{
    request_handler::request_handler()
    {
        memset(totaltime_, 0, sizeof(totaltime_));
        memset(totalcalls_, 0, sizeof(totalcalls_));
    }

    ///////////////////////////////////////////////////////////////////////////
    void request_handler::handle_getprefix(request const& req, reply& rep)
    {
        try {
            mutex_type::scoped_lock l(mtx_);
            site_prefix_map_type::iterator it = 
                site_prefixes_.find(req.get_site());
            if (it != site_prefixes_.end()) {
                // The real prefix has to be used as the 16 most 
                // significant bits of global id's
                
                // existing entry
                rep = reply(no_success, command_getprefix, 
                    boost::uint64_t((*it).second) << 48); 
            }
            else {
                boost::uint16_t prefix = (boost::uint16_t)site_prefixes_.size() + 1;
                site_prefixes_.insert(
                    site_prefix_map_type::value_type(req.get_site(), prefix));
                    
                // The real prefix has to be used as the 16 most 
                // significant bits of global id's
                
                // created new entry
                rep = reply(success, command_getprefix, 
                    boost::uint64_t(prefix) << 48);   
            }
        }
        catch (std::bad_alloc) {
            rep = reply(command_getprefix, out_of_memory);
        }            
        catch (...) {
            rep = reply(command_getprefix, internal_server_error);
        }            
    }

    void request_handler::handle_bind(request const& req, reply& rep)
    {
        try {
            status_type s = no_success;
            {
                mutex_type::scoped_lock l(mtx_);
                registry_type::iterator it = registry_.find(req.get_id());
                if (it != registry_.end())
                    (*it).second = req.get_address();
                else {
                    registry_.insert(registry_type::value_type(
                        req.get_id(), req.get_address()));
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

    void request_handler::handle_unbind(request const& req, reply& rep)
    {
        try {
            status_type s = no_success;
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

    void request_handler::handle_resolve(request const& req, reply& rep)
    {
        try {
            mutex_type::scoped_lock l(mtx_);
            registry_type::iterator it = registry_.find(req.get_id());
            if (it != registry_.end()) 
                rep = reply(command_resolve, (*it).second);
            else
                rep = reply(command_resolve, no_success);
        }
        catch (std::bad_alloc) {
            rep = reply(command_resolve, out_of_memory);
        }            
        catch (...) {
            rep = reply(command_resolve, internal_server_error);
        }            
    }

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

    void request_handler::handle_registerid(request const& req, reply& rep)
    {
        try {
            status_type s = no_success;
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

    void request_handler::handle_unregisterid(request const& req, reply& rep)
    {
        try {
            status_type s = no_success;
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

    void request_handler::handle_statistics(request const& req, reply& rep)
    {
        try {
            rep = reply(command_statistics, totaltime_, totalcalls_);
        }
        catch (std::bad_alloc) {
            rep = reply(command_statistics, out_of_memory);
        }            
        catch (...) {
            rep = reply(command_statistics, internal_server_error);
        }            
    }

    ///////////////////////////////////////////////////////////////////////////
    void request_handler::handle_request(request const& req, reply& rep)
    {
        switch (req.get_command()) {
        case command_getprefix:
            handle_getprefix(req, rep);
            break;
            
        case command_bind:
            handle_bind(req, rep);
            break;
            
        case command_unbind:
            handle_unbind(req, rep);
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
            
        case command_statistics:
            handle_statistics(req, rep);
            break;
            
        default:
            rep = reply(bad_request);
            break;
        }
    }

}}}  // namespace hpx::naming::server

