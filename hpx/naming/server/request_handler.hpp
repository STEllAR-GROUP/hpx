//  Copyright (c) 2007-2008 Hartmut Kaiser
//
//  Parts of this code were taken from the Boost.Asio library
//  Copyright (c) 2003-2007 Christopher M. Kohlhoff (chris at kohlhoff dot com)
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_NAMING_SERVER_REQUEST_HANDLER_MAR_24_2008_0946AM)
#define HPX_NAMING_SERVER_REQUEST_HANDLER_MAR_24_2008_0946AM

#include <string>
#include <map>

#include <boost/cstdint.hpp>
#include <boost/noncopyable.hpp>
#include <boost/thread.hpp>
#include <boost/asio/buffer.hpp>

#include <hpx/naming/name.hpp>
#include <hpx/naming/address.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace naming { namespace server 
{
    class request;

    /// The common handler for all incoming requests.
    class request_handler : private boost::noncopyable
    {
    public:
        request_handler();

        /// Handle a request and produce a reply.
        void handle_request(request const& req, reply& rep);

        void add_timing(boost::uint8_t command, double elapsed)
        {
            if (command >= command_firstcommand && 
                command < command_lastcommand)
            {
                totaltime_[command] += elapsed;
                ++totalcalls_[command];
            }
        }
        
    protected:
        void handle_getprefix(request const& req, reply& rep);
        void handle_bind(request const& req, reply& rep);
        void handle_unbind(request const& req, reply& rep);
        void handle_resolve(request const& req, reply& rep);
        void handle_queryid(request const& req, reply& rep);
        void handle_registerid(request const& req, reply& rep);
        void handle_unregisterid(request const& req, reply& rep);
        void handle_statistics(request const& req, reply& rep);

    private:
        typedef std::map<std::string, boost::uint64_t> ns_registry_type;
        typedef std::map<boost::uint64_t, naming::address> registry_type;
        typedef std::map<hpx::naming::locality, boost::uint16_t> 
            site_prefix_map_type;
        typedef boost::mutex mutex_type;
        
        mutex_type mtx_;
        ns_registry_type ns_registry_;
        registry_type registry_;
        site_prefix_map_type site_prefixes_;
        std::string msg_;

        // gathered timings and counts        
        double totaltime_[command_lastcommand];
        std::size_t totalcalls_[command_lastcommand];
    };

    // compare two entries in the site_prefix_map_type above
    inline bool operator< (
        std::pair<boost::uint32_t, boost::uint16_t> const& lhs,
        std::pair<boost::uint32_t, boost::uint16_t> const& rhs)
    {
        if (lhs.first < rhs.first)
            return true;
        if (lhs.first > rhs.first)
            return false;
        return lhs.second < rhs.second;
    }
    
}}}  // namespace hpx::naming::server

#endif 
