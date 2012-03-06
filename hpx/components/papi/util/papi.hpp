//  Copyright (c) 2011-2012 Maciej Brodowicz
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PERFORMANCE_COUNTERS_PAPI_UTIL_PAPI_201112101243)
#define HPX_PERFORMANCE_COUNTERS_PAPI_UTIL_PAPI_201112101243

#include <vector>
#include <string>

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/format.hpp>

#include <papi.h>

#include <hpx/hpx.hpp>
#include <hpx/config/export_definitions.hpp>

namespace hpx { namespace performance_counters { namespace papi { namespace util
{
    using boost::program_options::options_description;
    using boost::program_options::variables_map;

    ///////////////////////////////////////////////////////////////////////////
    // PAPI call wrapper
    inline void papi_call(int rc, char const *info, char const *fname,
                          int ok = PAPI_OK)
    {
        if (rc != ok)
        {
            boost::format err("%s (%s)");
            HPX_THROW_EXCEPTION(hpx::no_success, fname,
                                boost::str(err % info % PAPI_descr_error(rc)));
        }
    }
    inline void papi_call(int rc, std::string const& info, char const *fname,
                          int ok = PAPI_OK)
    {
        papi_call(rc, info.c_str(), fname, ok);
    }


    ///////////////////////////////////////////////////////////////////////////
    // generator to retrieve information on PAPI events
    template<bool all_events = false>
    class event_info_generator
    {
        int const enum_mask_;
        int event_;
        bool active_;
        PAPI_event_info_t info_;

        bool get_info()
        { // locally available events must not have null info_.count
            return active_ &&
                   PAPI_get_event_info(event_, &info_) == PAPI_OK &&
                   (!all_events || info_.count > 0);
        }
        bool advance()
        {
            if (!active_) return false;
            return (active_ = (PAPI_enum_event(&event_, enum_mask_) == PAPI_OK));
        }

    public:
        typedef PAPI_event_info_t const *result_type;

        event_info_generator(int event_mask = PAPI_PRESET_MASK):
            enum_mask_(all_events? PAPI_ENUM_ALL: PAPI_PRESET_ENUM_AVAIL),
            event_(event_mask), active_(true)
        { // get the first event from preset flags
            PAPI_enum_event(&event_, PAPI_ENUM_FIRST);
        }

        PAPI_event_info_t const *operator()()
        {
            while (!get_info())
                if (!advance()) return 0;
            advance();
            return &info_;
        }
    };

    // map domain description to a number
    int map_domain(std::string const&);

    // command line option description for PAPI counters
    options_description get_options_description();

    // get processed command line options
    variables_map get_options();

    // quick sanity check for command line options
    bool check_options(variables_map const& vm);

    // create list of event strings from command line options that are
    // relevant to this locality
    bool get_local_events(std::vector<std::string>& ev,
                          std::vector<std::string> const& opt);

    // list locally available events with detailed information
    void list_events();

}}}}

#endif
