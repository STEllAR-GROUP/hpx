//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c) 2011-2012 Maciej Brodowicz
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
//#include <hpx/util/parse_command_line.hpp>

#include <boost/format.hpp>
#include <boost/foreach.hpp>
#include <boost/chrono/chrono.hpp>
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>

#include <hpx/components/papi/server/papi.hpp>
#include <hpx/components/papi/stubs/papi.hpp>
#include <hpx/components/papi/util/papi.hpp>

///////////////////////////////////////////////////////////////////////////////
// Add factory registration functionality
HPX_REGISTER_COMPONENT_MODULE();

///////////////////////////////////////////////////////////////////////////////
typedef hpx::components::managed_component<
    hpx::performance_counters::papi::server::papi_counter
> papi_counter_type;

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters { namespace papi
{
    using boost::program_options::options_description;
    using boost::program_options::variables_map;
    using boost::algorithm::to_lower;

    // startup function for PAPI counter component 
    void startup()
    {
        using hpx::performance_counters::papi::util::papi_call;

        char const *const srcloc = "hpx::performance_counters::papi::startup()";
        
        // get processed command line options
        variables_map vm = util::get_options();
        bool multiplexed = false;

        // make sure PAPI environment is initialized
        util::papi_init();
        // enable library level multiplexing if requested
        if (vm.count("papi-multiplex"))
        {
            papi_call(PAPI_multiplex_init(),
                "cannot enable counter multiplexing", srcloc);
            multiplexed = true;
            // FIXME: setting of multiplexing period if other than default
        }
        if (vm.count("papi-domain"))
        {
            std::string ds = vm["papi-domain"].as<std::string>();
            int dom = util::map_domain(ds);
            boost::format fmt("cannot switch to \"%s\" domain monitoring");
            papi_call(PAPI_set_domain(dom), boost::str(fmt % ds).c_str(), srcloc);
        }
        
        // obtain local prefix
        boost::uint32_t const prefix = hpx::get_locality_id();

        // create counter component for every monitored event
        if (vm.count("papi-events"))
        {
            std::vector<std::string> events;
            util::get_local_events(events,
                vm["papi-events"].as<std::vector<std::string> >());

            for (size_t i = 0; i < events.size(); ++i)
            {
                // convert event name to internal PAPI code
                int ecode;
                boost::format err("PAPI event %s not supported");
                papi_call(PAPI_event_name_to_code((char *)events[i].c_str(), &ecode),
                    boost::str(err % events[i]).c_str(), srcloc);
                // get short counter description
                PAPI_event_info_t einfo;
                std::string edesc = "no description";
                if (PAPI_get_event_info(ecode, &einfo) == PAPI_OK)
                { // convert the first char to lowercase
                    edesc = einfo.short_descr[0];
                    to_lower(edesc);
                    edesc += einfo.short_descr+1;
                }

                // name of the counter instance
                boost::format instance_name("/papi(locality#%d)/%s");
                // define the counter type
                raw_counter_type_data papi_counter_data;
                boost::format counter_type_name("/papi/%s"),
                    counter_type_desc("returns the current count of \"%s\"");
                papi_counter_data.name_ = boost::str(counter_type_name % events[i]);
                papi_counter_data.type_ = counter_raw;
                papi_counter_data.version_ = HPX_PERFORMANCE_COUNTER_V1;
                papi_counter_data.helptext_ = boost::str(counter_type_desc % edesc);
                // install PAPI counter type
                install_counter_types(&papi_counter_data, 1);

                // full info of the counter to create, help text and version will be
                // complemented from counter type info as specified above
                counter_info info(counter_raw,
                    boost::str(instance_name % prefix % events[i]));

                // create the PAPI performance counter component locally
                hpx::naming::id_type id(
                    hpx::components::server::create_one<papi_counter_type>(info),
                    hpx::naming::id_type::managed);

                // install the created counter, un-installation is automatic
                install_counter(id, info);

                // enforce multiplexing on event set (has to be done once and
                // before starting the count)
                if (i == 0 && vm.count("papi-multiplex"))
                    stubs::papi_counter::enable_multiplexing(id,
                        vm["papi-multiplex"].as<long>());

                // configure counter to monitor the specified event
                if (!stubs::papi_counter::set_event(id, ecode, true))
                {
                    boost::format err("failed to activate event %s");
                    HPX_THROW_EXCEPTION(hpx::no_success, srcloc,
                        boost::str(err % events[i]));
                }
            }
        }
    }

    bool check_startup(HPX_STD_FUNCTION<void()>& startup_func)
    {
        // retrieve command line
        variables_map vm = util::get_options();

        if (util::check_options(vm))
        { // perform full module startup (counters must be created)
            startup_func = startup;
            return true;
        }
        // list known events?
        if (vm.count("papi-list-events"))
            util::list_events(vm["papi-list-events"].as<std::string>());

        return false;
    }

}}}

///////////////////////////////////////////////////////////////////////////////
// register a startup function for PAPI performance counter
HPX_REGISTER_STARTUP_MODULE(hpx::performance_counters::papi::check_startup);
// register related command line options
HPX_REGISTER_COMMANDLINE_MODULE(hpx::performance_counters::papi::util::get_options_description);
