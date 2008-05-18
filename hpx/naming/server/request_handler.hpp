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
#if BOOST_VERSION >= 103600
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/count.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/moment.hpp>
#endif

#include <hpx/naming/name.hpp>
#include <hpx/naming/address.hpp>
#include <hpx/naming/locality.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace naming { namespace server 
{
    class request;
    class reply;

    /// The common handler for all incoming requests.
    class request_handler : private boost::noncopyable
    {
        /// size of the id range returned by command_getidrange
        /// FIXME: is this a policy?
        enum { range_delta = 1023 };
        
    public:
        request_handler();
        ~request_handler();

        /// Handle a request and produce a reply.
        void handle_request(request const& req, reply& rep);

        // collect statistics
        void add_timing(boost::uint8_t command, double elapsed)
        {
            if (command >= command_firstcommand && 
                command < command_lastcommand)
            {
#if BOOST_VERSION >= 103600
                totals_[command](elapsed);
#else
                totals_[command].first += elapsed;
                ++totals_[command].second;
#endif
            }
        }
        
    protected:
        void handle_getprefix(request const& req, reply& rep);
        void handle_getidrange(request const& req, reply& rep);
        void handle_bind(request const& req, reply& rep);
        void handle_unbind(request const& req, reply& rep);
        void handle_resolve(request const& req, reply& rep);
        void handle_queryid(request const& req, reply& rep);
        void handle_registerid(request const& req, reply& rep);
        void handle_unregisterid(request const& req, reply& rep);
        void handle_statistics_count(request const& req, reply& rep);
        void handle_statistics_mean(request const& req, reply& rep);
        void handle_statistics_moment2(request const& req, reply& rep);

    private:
        // The ns_registry_type is used to store the mappings from the 
        // global names (strings) to global ids.
        typedef std::map<std::string, naming::id_type> ns_registry_type;

        // The registry_type is used to store the mapping from the global ids
        // to the current local address of the corresponding component.
        typedef std::map<naming::id_type, naming::address> registry_type;

        // The site_prefix_map_type is used to store the assigned prefix and 
        // the current upper boundary to be used for global id assignment for
        // a particular locality.
        typedef std::pair<boost::uint32_t, naming::id_type> site_prefix_type;
        typedef std::map<hpx::naming::locality, site_prefix_type> 
            site_prefix_map_type;
        typedef site_prefix_map_type::value_type site_prefix_value_type;
            
        typedef boost::mutex mutex_type;
        
        // comparison operator for the entries stored in the site_prefix_map
        friend bool operator< (site_prefix_value_type const& lhs,
            site_prefix_value_type const& rhs);

        mutex_type mtx_;
        ns_registry_type ns_registry_;        // "name" --> global_id
        registry_type registry_;              // global_id --> local_address
        site_prefix_map_type site_prefixes_;  // locality --> prefix, upper_boundary

        // gathered timings and counts
#if BOOST_VERSION >= 103600
        typedef boost::accumulators::stats<
            boost::accumulators::tag::count, 
            boost::accumulators::tag::mean, 
            boost::accumulators::tag::moment<2> > accumulator_stats_type;
        typedef boost::accumulators::accumulator_set<
            double, accumulator_stats_type> accumulator_set_type;

    public:
        typedef std::vector<accumulator_set_type> totals_type;
#else
    public:
        typedef std::vector<std::pair<double, std::size_t> > totals_type;
#endif

    private:
        totals_type totals_;
    };

    // compare two entries in the site_prefix_map_type above
    inline bool operator< (request_handler::site_prefix_value_type const& lhs,
        request_handler::site_prefix_value_type const& rhs)
    {
        if (lhs.first < rhs.first)
            return true;
        if (lhs.first > rhs.first)
            return false;
        return lhs.second < rhs.second;
    }
    
}}}  // namespace hpx::naming::server

#endif 
