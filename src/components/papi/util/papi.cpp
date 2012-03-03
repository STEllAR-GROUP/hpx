//  Copyright (c) 2011-2012 Maciej Brodowicz
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <map>

#include <boost/format.hpp>
#include <boost/asio/ip/host_name.hpp>
#include <boost/assign/list_of.hpp>

#include <hpx/util/parse_command_line.hpp>
#include <hpx/components/papi/util/papi.hpp>

#define NS_STR "hpx::performance_counters::papi::util::"

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters { namespace papi { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    // PAPI domain description strings
    const std::map<std::string, int> papi_domain_map = boost::assign::map_list_of
        ("user",   PAPI_DOM_USER)
        ("kernel", PAPI_DOM_KERNEL)
        ("other",  PAPI_DOM_OTHER)
        ("super",  PAPI_DOM_SUPERVISOR)
        ("all",    PAPI_DOM_ALL)
        ("min",    PAPI_DOM_MIN)
        ("max",    PAPI_DOM_MAX)
        ;

    int map_domain(std::string const& s)
    {
        std::map<std::string, int>::const_iterator it = papi_domain_map.find(s);
        if (it == papi_domain_map.end())
        {
            boost::format fmt("invalid argument %s for --papi-domain");
            HPX_THROW_EXCEPTION(hpx::commandline_option_error,
                                NS_STR "check_options()", str(fmt % s));
        }
        return it->second;
    }
    
    ///////////////////////////////////////////////////////////////////////////
    options_description get_options_description()
    {
        using boost::program_options::value;

        // description of PAPI counter specific options
        options_description papi_opts("PAPI counter options");
        papi_opts.add_options()
            ("papi-events", value<std::vector<std::string> >()->composing(),
             "list of monitored PAPI events in the following format: "
             "[host@]event1[,event2...]. If the host name is specified and "
             "matches the local machine name, the associated event list is "
             "used, otherwise events from the host-less lists are applied.")
            ("papi-domain", value<std::string>()->default_value("user"),
             "sets monitoring scope to one of:\n"
             "  user   - user context only,\n"
             "  kernel - kernel/OS context only,\n"
             "  super  - supervisor or hypervisor context,\n"
             "  other  - exception and transient mode,\n"
             "  all    - all above contexts,\n"
             "  min    - smallest available context,\n"
             "  max    - largest available context.")
            ("papi-multiplex", value<long>()->implicit_value(0),
             "enable low level counter multiplexing.")
            ("papi-list-events", value<std::string>()->implicit_value("short"),
             "list events available on local host (mutually exclusive with "
             "--papi-events); the optional argument is one of:\n"
             "  full  - for detailed event description,\n"
             "  short - for minimal description.")
            ;
        return papi_opts;
    }

    variables_map get_options()
    {
        using hpx::util::retrieve_commandline_arguments;

        variables_map vm;
        if (!retrieve_commandline_arguments(get_options_description(), vm))
        {
            HPX_THROW_EXCEPTION(hpx::not_implemented,
                NS_STR "get_options()",
                "Failed to handle command line options");
        }
        return vm;
    }

    bool check_options(variables_map const& vm)
    {
        if (vm.count("papi-events") && vm.count("papi-list-events"))
            HPX_THROW_EXCEPTION(hpx::commandline_option_error,
                NS_STR "check_options()",
                "--papi-events and --papi-event-list are mutually exclusive");
        bool needed = false;
        if (vm.count("papi-events")) needed = true;
        if (vm.count("papi-list-events"))
        {
            std::string v = vm["papi-list-events"].as<std::string>();
            if (v != "short" && v != "full")
                HPX_THROW_EXCEPTION(hpx::commandline_option_error,
                    NS_STR"check_options()",
                    "invalid option to --papi-list-events");
        }
        if (vm.count("papi-domain"))
        {
            std::string v = vm["papi-domain"].as<std::string>();
            map_domain(v); // throws if not found
        }
        if (vm.count("papi-multiplex") && vm["papi-multiplex"].as<long>() < 0)
            HPX_THROW_EXCEPTION(hpx::commandline_option_error, NS_STR "check_options()",
                "argument to --papi-multiplex must be positive");
        return needed;
    }

    ///////////////////////////////////////////////////////////////////////////
    enum LocalMode
    {
        NON_LOCAL,      // host name specified and doesn't match
        STRICTLY_LOCAL, // host name specified and matches
        DEFAULT_LOCAL   // no host name
    };

    LocalMode local_mode(std::string& opt, std::string& host)
    {
        size_t i = opt.find(HOST_DELIMITER);
        if (i == 0)
        {
            boost::format efmt("Missing host name in \"%s\"");
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                                NS_STR "local_mode()", str(efmt % opt));
        }
        if (i == std::string::npos) return DEFAULT_LOCAL;

        LocalMode lm = (host == opt.substr(0, i))? STRICTLY_LOCAL: NON_LOCAL;
        // remove host name from option string
        opt.erase(0, i+1);
        return lm;
    }

    bool append_unique(std::vector<std::string>& vs, std::string s)
    {
        std::vector<std::string>::const_iterator it;
        for (it = vs.begin(); it != vs.end(); it++)
            if (*it == s) return false;
        vs.push_back(s);
        return true;
    }

    bool get_local_events(std::vector<std::string>& evstr,
                          std::vector<std::string> const& opt)
    {
        using boost::asio::ip::host_name;

        std::vector<std::string>::const_iterator it;
        std::vector<std::string> defev, locev, *evp = &defev;
        std::string hostname(host_name());
        for (it = opt.begin(); it != opt.end(); ++it)
        {
            std::string opt = *it;
            switch (local_mode(opt, hostname))
            {
            case NON_LOCAL:
                continue;
            case STRICTLY_LOCAL:
                evp = &locev;
                break;
            case DEFAULT_LOCAL:
                if (!locev.empty()) continue;
                evp = &defev;
                break;
            }
            std::string::size_type start = 0, end;
            // split event list at delimiters
            while ((end = opt.find(EVENT_DELIMITER, start)) != std::string::npos)
            {
                append_unique(*evp, opt.substr(start, end-start));
                start = end+1;
            }
            append_unique(*evp, opt.substr(start, opt.length()-start));
        }
        if (!locev.empty())
        {
            evstr = locev;
            return true;
        }
        else if (!defev.empty())
        {
            evstr = defev;
            return true;
        }

        return false;
    }

    std::string decode_type(unsigned t)
    {
        std::string s;
        if (t & PAPI_PRESET_BIT_MSC) s += ":OTHER";
        if (t & PAPI_PRESET_BIT_INS) s += ":INS";
        if (t & PAPI_PRESET_BIT_IDL) s += ":IDLE";
        if (t & PAPI_PRESET_BIT_BR)  s += ":BRANCH";
        if (t & PAPI_PRESET_BIT_CND) s += ":COND";
        if (t & PAPI_PRESET_BIT_MEM) s += ":MEM";
        if (t & PAPI_PRESET_ENUM_CACH) s+= ":CACHE";
        if (t & PAPI_PRESET_BIT_L1)  s += ":L1";
        if (t & PAPI_PRESET_BIT_L2)  s += ":L2";
        if (t & PAPI_PRESET_BIT_L3)  s += ":L3";
        if (t & PAPI_PRESET_BIT_TLB) s += ":TLB";
        if (t & PAPI_PRESET_BIT_FP)  s += ":FP";
        if (!s.empty()) return s.substr(1);
        s = "-";
        return s;
    }

    std::string dependencies(event_info& info)
    {
        if (info.count <= 1) return std::string("none");
        std::string s;
        for (size_t i = 0; i < info.count; ++i)
        {
            if (i > 0) s += ",";
            s += info.name[i];
        }
        return s;
    }

    void list_events(std::string const& mode)
    {
        using boost::asio::ip::host_name;

        bool full = true;  // produce full information listing
        boost::format fmt; // per-event output format
        if (mode == "short")
        {
            full = false;
            fmt = boost::format("%-15s: %s\n");
        }
        else
        {
            fmt = boost::format(
                      std::string("Event        : %s\n"
                                  "Type         : %s\n"
                                  "Derived from : %s\n"
                                  "Description  : %s\n")+
                                  std::string(79, '-')+'\n');
        }

        // PAPI library is likely not initialized at this point
        papi_init();

        // print header
        std::string host(host_name()), hdr("Available PAPI events on ");
        if (host.empty()) hdr += "unknown host";
        else hdr += host;
        hdr += ":";
        std::cout << hdr << std::endl
                  << std::string(hdr.length(), '=') << std::endl;

        // collect available events and print their descriptions
        int ev = PAPI_PRESET_MASK;
        PAPI_enum_event(&ev, PAPI_ENUM_FIRST);
        do
        {
            PAPI_event_info_t info;
            if (PAPI_get_event_info(ev, &info) == PAPI_OK && info.count)
            {
                if (full)
                    std::cout << fmt % info.symbol
                                     % decode_type(info.event_type)
                                     % dependencies(info)
                                     % info.long_descr;
                else
                    std::cout << fmt % info.symbol % info.short_descr;
            }
        } while (PAPI_enum_event(&ev, PAPI_ENUM_ALL/*PRESET_ENUM_AVAIL*/) == PAPI_OK);
    }

}}}}
