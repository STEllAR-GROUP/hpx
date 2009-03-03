//  Copyright (c) 2007-2009 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PERFORMANCE_COUNTERS_REGISTRY_MAR_01_2009_0424PM)
#define HPX_PERFORMANCE_COUNTERS_REGISTRY_MAR_01_2009_0424PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/performance_counters/counters.hpp>

#include <boost/function.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters 
{
    class registry
    {
    private:
        typedef std::map<std::string, counter_info> counter_type_map_type;

    public:
        registry(naming::resolver_client& agas_client);

        /// \brief Add a new performance counter type to the (local) registry
        counter_status add_counter_type(counter_info const& info);

        /// \brief Create a new performance counter instance based on given
        ///        counter value
        counter_status add_counter(counter_info const& info, 
            boost::int64_t* countervalue, naming::id_type& id);

        /// \brief Create a new performance counter instance based on given
        ///        function returning the counter value
        counter_status add_counter(counter_info const& info, 
            boost::function<boost::int64_t()> f, naming::id_type& id);

    private:
        naming::resolver_client& agas_client_;
        counter_type_map_type countertypes_;
    };

}}

#endif

