//  Copyright (c) 2011-2012 Maciej Brodowicz
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PERFORMANCE_COUNTERS_PAPI_UTIL_PAPI_201112101243)
#define HPX_PERFORMANCE_COUNTERS_PAPI_UTIL_PAPI_201112101243

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/format.hpp>

#include <papi.h>

#include "hpx/hpx.hpp"
#include "hpx/config/export_definitions.hpp"

namespace hpx { namespace performance_counters { namespace papi { namespace util
{
    using boost::program_options::options_description;
    using boost::program_options::variables_map;

    ///////////////////////////////////////////////////////////////////////////
    // event list delimiters
    char const EVENT_DELIMITER = ',';
    char const HOST_DELIMITER  = '@'; // ':' might be used in event names

    
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

    // PAPI library initialization
    inline void papi_init()
    {
        if (PAPI_is_initialized() == PAPI_NOT_INITED)
        {
            papi_call(PAPI_library_init(PAPI_VER_CURRENT),
                "PAPI library initialization failed (version mismatch)",
                "hpx::performance_counters::papi::util::list_events()",
                PAPI_VER_CURRENT);
        }
    }

    // map domain description to a number
    int map_domain(std::string const&);
    
    // command line option description for PAPI counters
    options_description get_options_description();

    // get processed command line options
    variables_map get_options();

    // coarse sanity check for options
    bool check_options(variables_map const& vm);

    // create list of event strings from command line options
    bool get_local_events(std::vector<std::string>& ev,
                          std::vector<std::string> const& opt);

    // list available events (true switches to long format)
    void list_events(std::string const& mode);

}}}}

#endif
