//  Copyright (c) 2007-2009 Hartmut Kaiser
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
#include <boost/fusion/include/vector.hpp>
#if BOOST_VERSION >= 103600
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/count.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/moment.hpp>
#endif

#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/naming/locality.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace naming { namespace server 
{
    class request;
    class reply;

    /// The common handler for all incoming requests.
    class request_handler : private boost::noncopyable
    {
    public:
        request_handler();
        ~request_handler();

        /// Handle a request and produce a reply.
        void handle_request(request const& req, reply& rep);

        // collect statistics
        void add_timing(boost::uint8_t command, double elapsed)
        {
            if ((unsigned)command < command_lastcommand)
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
        void handle_getconsoleprefix(request const& req, reply& rep);
        void handle_getprefixes(request const& req, reply& rep);
        void handle_get_component_id(request const& req, reply& rep);
        void handle_register_factory(request const& req, reply& rep);
        void handle_getidrange(request const& req, reply& rep);
        void handle_bind_range(request const& req, reply& rep);
        void handle_unbind_range(request const& req, reply& rep);
        void handle_resolve(request const& req, reply& rep);
        void handle_queryid(request const& req, reply& rep);
        void handle_registerid(request const& req, reply& rep);
        void handle_unregisterid(request const& req, reply& rep);
        void handle_statistics_count(request const& req, reply& rep);
        void handle_statistics_mean(request const& req, reply& rep);
        void handle_statistics_moment2(request const& req, reply& rep);

    private:
        // The registry_type is used to store the mapping from the global ids
        // to a range of local addresses of the corresponding component 
        // (defined by a base address, the count, and the offset).
        typedef boost::fusion::vector<
            naming::address, std::size_t, std::ptrdiff_t> 
        registry_data_type;

    protected:
        void create_new_binding(request const &req, error& s, std::string& str)
        {
            naming::id_type upper_bound;
            upper_bound = req.get_id() + (req.get_count() - 1);
            if (req.get_id().get_msb() != upper_bound.get_msb()) {
                s = internal_server_error;
                str = "msb's of global ids of lower and upper range bound should match";
            }
            else {
                registry_.insert(
                    registry_type::value_type(req.get_id(), 
                        registry_data_type(req.get_address(), 
                            req.get_count(), req.get_offset()
                        )));
                s = success;    // created new entry
            }
        }

    private:
        // The ns_registry_type is used to store the mappings from the 
        // global names (strings) to global ids.
        typedef std::map<std::string, naming::id_type> ns_registry_type;
        typedef std::map<naming::id_type, registry_data_type> registry_type;

        // The site_prefix_map_type is used to store the assigned prefix and 
        // the current upper boundary to be used for global id assignment for
        // a particular locality.
        typedef std::pair<boost::uint32_t, naming::id_type> site_prefix_type;
        typedef std::map<hpx::naming::locality, site_prefix_type> 
            site_prefix_map_type;
        typedef site_prefix_map_type::value_type site_prefix_value_type;

        // comparison operator for the entries stored in the site_prefix_map
        friend bool operator< (site_prefix_value_type const& lhs,
            site_prefix_value_type const& rhs);

        // Store all registered component types and registered factories
        typedef std::map<std::string, int> component_type_map;
        typedef std::multimap<int, boost::uint32_t> factory_map;

        typedef boost::mutex mutex_type;

        mutex_type ns_registry_mtx_;
        ns_registry_type ns_registry_;        // "name" --> global_id

        mutex_type registry_mtx_;
        registry_type registry_;              // global_id --> local_address
        site_prefix_map_type site_prefixes_;  // locality --> prefix, upper_boundary
        boost::uint32_t console_prefix_;      // console locality

        mutex_type component_types_mtx_;
        int component_type_;
        component_type_map component_types_;  // component names --> type id
        factory_map factories_;               // type id --> prefixes

        // gathered timings and counts
#if BOOST_VERSION >= 103600
        typedef boost::accumulators::stats<
            boost::accumulators::tag::count, 
            boost::accumulators::tag::mean, 
            boost::accumulators::tag::moment<2> > accumulator_stats_type;
        typedef boost::accumulators::accumulator_set<
            double, accumulator_stats_type> accumulator_set_type;

        static double extract_moment2(accumulator_set_type const& p);

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
