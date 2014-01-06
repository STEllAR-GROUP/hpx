//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_7B5783D6_FCA3_4E3D_BBCE_FBB0A6CF644B)
#define HPX_7B5783D6_FCA3_4E3D_BBCE_FBB0A6CF644B

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/interval_timer.hpp>
#include <hpx/lcos/local/mutex.hpp>
#include <hpx/include/performance_counters.hpp>

#include <string>
#include <vector>

#include <boost/cstdint.hpp>

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    class HPX_EXPORT activate_counters
    {
    public:
        activate_counters(std::vector<std::string> const& names);

        ~activate_counters();

    protected:
        void find_counters();
        bool find_counter(performance_counters::counter_info const& info,
            error_code& ec);

        void start_counters(error_code& ec = throws);
        void stop_counters(error_code& ec = throws);

    public:
        void reset_counters(error_code& ec = throws);

        typedef std::vector<unique_future<performance_counters::counter_value> >
            counter_values_type;

        counter_values_type
        evaluate_counters(bool reset = false, error_code& ec = throws); 

        std::string name(boost::uint64_t i) const
        {
            return names_[i];
        }

        naming::id_type id(boost::uint64_t i) const
        {
            return ids_[i];
        }

        std::string unit_of_measure(boost::uint64_t i) const
        {
            return uoms_[i];
        }

    private:
        std::vector<std::string> names_;      // counter instance names
        std::vector<naming::id_type> ids_;    // gids of counter instances
        std::vector<std::string> uoms_;       // units of measure
    };
}}

#endif // 7B5783D6_FCA3_4E3D_BBCE_FBB0A6CF644B

