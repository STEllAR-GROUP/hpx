//  Copyright (c) 2011-2012 Maciej Brodowicz
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_PAPI)

#include <map>

#include <boost/format.hpp>
#include <boost/asio/ip/host_name.hpp>
#include <boost/assign/list_of.hpp>
#include <boost/generator_iterator.hpp>

#include <hpx/exception.hpp>
#include <hpx/util/parse_command_line.hpp>
#include <hpx/util/thread_mapper.hpp>
#include <hpx/components/papi/util/papi.hpp>

#define NS_STR "hpx::performance_counters::papi::util::"

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters { namespace papi { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    // PAPI domain handling

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

    // map domain string into PAPI handle
    int map_domain(std::string const& s)
    {
        std::map<std::string, int>::const_iterator it = papi_domain_map.find(s);
        if (it == papi_domain_map.end())
            HPX_THROW_EXCEPTION(hpx::commandline_option_error,
                NS_STR "check_options()",
                "invalid argument "+s+" to --papi-domain");
        return it->second;
    }

    ///////////////////////////////////////////////////////////////////////////
    // option handling

    // return options_description object for PAPI counters
    options_description get_options_description()
    {
        using boost::program_options::value;

        // description of PAPI counter specific options
        options_description papi_opts("PAPI counter options");
        papi_opts.add_options()
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
            ("papi-event-info", value<std::string>()->implicit_value("preset"),
             "display detailed information about events available on local host;"
             " the optional argument is one of:\n"
             "  preset - show available predefined events,\n"
             "  native - show available native events,\n"
             "  all    - show all avilable events.")
            ;
        return papi_opts;
    }

    // retrieve command line options
    variables_map get_options()
    {
        using hpx::util::retrieve_commandline_arguments;

        variables_map vm;
        if (!retrieve_commandline_arguments(get_options_description(), vm))
        {
            HPX_THROW_EXCEPTION(hpx::commandline_option_error,
                NS_STR "get_options()",
                "failed to handle command line options");
        }
        return vm;
    }

    // scan through performance counters to see if PAPI is needed
    bool need_papi_component(variables_map const& vm)
    {
        if (vm.count("hpx:print-counter"))
        {
            std::vector<std::string> names =
                vm["hpx:print-counter"].as<std::vector<std::string> >();
            std::vector<std::string>::iterator it;
            for (it = names.begin(); it != names.end(); it++)
                if (it->substr(0, 6) == "/papi{")
                    return true;
        }
        return false;
    }

    // checks options for sanity; returns true if papi component is needed
    bool check_options(variables_map const& vm)
    {
        bool needed = need_papi_component(vm);
        if (vm.count("papi-event-info"))
        {
            std::string v = vm["papi-event-info"].as<std::string>();
            if (v != "preset" && v != "native" && v != "all")
                HPX_THROW_EXCEPTION(hpx::commandline_option_error,
                    NS_STR "check_options()",
                    "unsupported mode "+v+" in --papi-event-info");
        }
        if (vm.count("papi-domain"))
        {
            std::string v = vm["papi-domain"].as<std::string>();
            int dom = map_domain(v); // throws if not found
            papi_call(PAPI_set_domain(dom),
                "could not switch to \""+v+"\" domain monitoring",
                NS_STR "check_options()");
            needed = true;
        }
        // FIXME: implement multiplexing properly and uncomment below when done
        if (vm.count("papi-multiplex"))
            HPX_THROW_EXCEPTION(hpx::not_implemented,
                NS_STR "check_options()",
                "counter multiplexing is currently not supported");
#if 0
        if (vm.count("papi-multiplex") && vm["papi-multiplex"].as<long>() < 0)
            HPX_THROW_EXCEPTION(hpx::commandline_option_error,
                NS_STR "check_options()",
                "argument to --papi-multiplex must be positive");
#endif
        return needed;
    }

    ///////////////////////////////////////////////////////////////////////////
    // event listing options

    // decipher event type flags
    std::string decode_preset_type(unsigned t)
    {
        std::string s;
        if (t & PAPI_PRESET_BIT_MSC) s += ",OTHER";
        if (t & PAPI_PRESET_BIT_INS) s += ",INS";
        if (t & PAPI_PRESET_BIT_IDL) s += ",IDLE";
        if (t & PAPI_PRESET_BIT_BR)  s += ",BRANCH";
        if (t & PAPI_PRESET_BIT_CND) s += ",COND";
        if (t & PAPI_PRESET_BIT_MEM) s += ",MEM";
        if (t & PAPI_PRESET_ENUM_CACH) s+= ",CACHE";
        if (t & PAPI_PRESET_BIT_L1)  s += ",L1";
        if (t & PAPI_PRESET_BIT_L2)  s += ",L2";
        if (t & PAPI_PRESET_BIT_L3)  s += ",L3";
        if (t & PAPI_PRESET_BIT_TLB) s += ",TLB";
        if (t & PAPI_PRESET_BIT_FP)  s += ",FP";
        if (!s.empty()) return s.substr(1);
        s = "-";
        return s;
    }

    // determine depndencies of a derived event
    std::string dependencies(event_info const& info)
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

    void list_presets()
    {
        boost::format fmt = boost::format(
            std::string("Event        : %s\n"
                        "Type         : %s\n"
                        "Code         : 0x%x\n"
                        "Derived from : %s\n"
                        "Description  : %s\n"
                        "%s")+                 // optional note
                            std::string(79, '-')+'\n');


        // collect available events and print their descriptions
        avail_preset_info_gen gen;
        boost::generator_iterator_generator<avail_preset_info_gen>::type it;
        for (it = boost::make_generator_iterator(gen); *it != 0; ++it)
        {
            std::string note;
            if (strlen((*it)->note) > 0)
            { // is this actually provided anywhere??
                note = "Note:\n";
                note += (*it)->note;
                note += "\n";
            }
            std::cout << fmt % (*it)->symbol
                             % decode_preset_type((*it)->event_type)
                             % (*it)->event_code
                             % dependencies(**it)
                             % (*it)->long_descr
                             % note;
        }
    }

    std::string registers(PAPI_event_info_t const& info)
    {
        if (!info.count) return "-";
        std::string regs;
        for (unsigned i = 0; i < info.count; ++i)
        {
            if (info.name[i][0])
            {
                if (!regs.empty()) regs += std::string(15, ' ');
                boost::format fmt("reg[%d] name: %-20s value: 0x%16x\n");
                regs += boost::str(fmt % i % info.name[i] % info.code[i]);
            }
        }
        return regs;
    }

    void print_native_info(PAPI_event_info_t const& info)
    {
        boost::format fmt = boost::format(
            std::string("Event        : %s\n"
                        "Type         : native\n"
                        "Code         : 0x%x\n"
                        "Registers    : %s\n"
                        "Description  : %s\n"
                        "%s")+                 // optional note
                            std::string(79, '-')+'\n');
        std::string note;
        if (strlen(info.note) > 0)
        {
            note = "Note:\n";
            note += info.note;
            note += "\n";
        }
        std::cout << fmt % info.symbol
                         % info.event_code
                         % registers(info)
                         % info.long_descr
                         % note;
    }

    void list_native()
    {
        // list available events for each PAPI component
        for (int ci = 0; ci < PAPI_num_components(); ++ci)
        {
            native_info_gen gen(ci);
            boost::generator_iterator_generator<native_info_gen>::type it =
                boost::make_generator_iterator(gen);
            for ( ; *it != 0; ++it)
                print_native_info(**it);
        }
    }

    // list available events with descriptions
    void list_events(std::string const& scope)
    {
        using boost::asio::ip::host_name;

        std::string host(host_name());
        // print header
        std::string hdr("PAPI events available on ");
        if (!host.empty())
        {
            boost::format fmt("%s (locality %d):");
            hdr += boost::str(fmt % host % hpx::get_locality_id());
        }
        else
        {
            boost::format fmt("locality %d:");
            hdr += boost::str(fmt % hpx::get_locality_id());
        }
        std::cout << hdr << std::endl
                  << std::string(hdr.length(), '=') << std::endl;
        if (scope == "preset" || scope == "all") list_presets();
        if (scope == "native" || scope == "all") list_native();
    }

    boost::uint32_t get_counter_thread(counter_path_elements const& cpe,
                                       std::string& label)
    {
        hpx::util::thread_mapper& tm = get_runtime().get_thread_mapper();
        if (cpe.instanceindex_ < 0)
        {
            label = cpe.instancename_;
        }
        else
        {
            boost::format lab("%s#%d");
            label = boost::str(lab % cpe.instancename_ % cpe.instanceindex_);
        }
        return tm.get_thread_index(label);
    }

}}}}

#endif
