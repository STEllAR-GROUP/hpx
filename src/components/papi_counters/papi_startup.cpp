//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c) 2011-2012 Maciej Brodowicz
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <cctype>

#include <boost/format.hpp>
#include <boost/foreach.hpp>
#include <boost/program_options.hpp>

#include <hpx/hpx.hpp>
#include <hpx/util/thread_mapper.hpp>
#include <hpx/components/papi/server/papi.hpp>
#include <hpx/components/papi/util/papi.hpp>
#include <hpx/exception.hpp>

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
    using hpx::performance_counters::counter_info;
    using hpx::performance_counters::counter_path_elements;
    using hpx::util::thread_mapper;
    using util::papi_call;

#define NS_STR "hpx::performance_counters::papi::"

    // create PAPI counter
    hpx::naming::gid_type create_papi_counter(counter_info const& info,
                                              hpx::error_code& ec)
    {
        // verify the validity of the counter instance name
        hpx::performance_counters::counter_path_elements paths;
        get_counter_path_elements(info.fullname_, paths, ec);
        if (ec) return hpx::naming::invalid_gid;

        if (paths.parentinstance_is_basename_)
        {
            HPX_THROWS_IF(ec, hpx::bad_parameter,
                NS_STR "create_papi_counter()",
                "unsupported counter instance parent name: "+
                    paths.parentinstancename_);
            return hpx::naming::invalid_gid;
        }

        // validate thread label taken from counter name
        thread_mapper& tm = hpx::get_runtime().get_thread_mapper();
        boost::format label("%s-%d");
        boost::uint32_t tix = tm.get_thread_index(
            str(label % paths.instancename_ % paths.instanceindex_));
        if (tix == thread_mapper::invalid_index)
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                NS_STR "create_papi_counter()",
                "cannot find thread specified in "+info.fullname_);
        // create a local PAPI counter component
        hpx::naming::gid_type id;
        try
        {
            id = hpx::components::server::create_one<papi_counter_type>(info);
        }
        catch (hpx::exception const& e)
        {
            if (&ec == &hpx::throws)
                throw;
            ec = make_error_code(e.get_error(), e.what());
            return hpx::naming::invalid_gid;
        }

        if (&ec != &hpx::throws) ec = hpx::make_success_code();
        return id;
    }

    // bool is true when group is enumerated
    typedef std::pair<std::string, bool> thread_category;

    // find thread categories
    void find_thread_groups(std::vector<std::string> const& labels,
                            std::vector<std::string>& tdesc)
    {
        std::vector<std::string>::const_iterator it;
        std::set<thread_category> cats;
        for (it = labels.begin(); it != labels.end(); it++)
        {
            size_t pos = it->find_last_of('-');
            if (pos == it->npos)
                cats.insert(std::make_pair(*it, false));
            else
            {
                char const *p = it->c_str();
                char *end;
                (void)strtol(p+pos+1, &end, 0);
                if (*end == 0)
                    cats.insert(std::make_pair(it->substr(0, pos), true));
                else
                    cats.insert(std::make_pair(*it, false));
            }
        }
        std::set<thread_category>::iterator si;
        for (si = cats.begin(); si != cats.end(); si++)
            tdesc.push_back(si->second? si->first+"#<*>": si->first);
    }

    template<class T>
    bool discover_events(counter_path_elements& cpe,
        std::vector<std::string> const& tdesc,
        hpx::performance_counters::counter_info& info,
        T& gen,
        HPX_STD_FUNCTION<hpx::performance_counters::discover_counter_func> const& f,
        hpx::error_code& ec)
    {
        typename boost::generator_iterator_generator<T>::type gi =
            boost::make_generator_iterator(gen);
        for ( ; *gi != 0; ++gi)
        {
            std::vector<std::string>::const_iterator it;
            // iterate over known thread names
            for (it = tdesc.begin(); it != tdesc.end(); ++it)
            {
                cpe.instancename_ = *it;
                cpe.countername_ = (*gi)->symbol;

                hpx::performance_counters::counter_status status =
                    get_counter_name(cpe, info.fullname_, ec);
                if (!status_is_valid(status)) return false;
                std::string evstr((*gi)->long_descr);
                info.helptext_ = "returns the count of occurrences of \""+
                    evstr+"\" in "+(*it)+" instance";
                if (!f(info, ec) || ec) return false;
            }
        }
        return true;
    }

    // discover available PAPI counters
    bool discover_papi_counters(
        hpx::performance_counters::counter_info const& info,
        HPX_STD_FUNCTION<hpx::performance_counters::discover_counter_func> const& f,
        hpx::error_code& ec)
    {
        hpx::performance_counters::counter_info cnt_info = info;

        // decompose the counter name
        counter_path_elements p;
        hpx::performance_counters::counter_status status =
            get_counter_path_elements(info.fullname_, p, ec);
        if (!status_is_valid(status)) return false;

        // obtain known OS thread labels
        std::vector<std::string> labels, tdesc;
        hpx::util::thread_mapper& tm = get_runtime().get_thread_mapper();
        tm.get_registered_labels(labels);
        find_thread_groups(labels, tdesc);

        // fill in common path segments for all counters
        counter_path_elements cpe;
        cpe.objectname_ = p.objectname_;
        cpe.parentinstancename_ = "locality#<*>";
        cpe.parentinstanceindex_ = -1;
        cpe.instanceindex_ = -1;

        // enumerate PAPI presets
        {
            util::all_preset_info_gen gen;
            if (!discover_events(cpe, tdesc, cnt_info, gen, f, ec))
                return false;
        }
        // enumerate PAPI native events
        for (int ci = 0; ci < PAPI_num_components(); ++ci)
        {
            util::native_info_gen gen(ci);
            if (!discover_events(cpe, tdesc, cnt_info, gen, f, ec))
                return false;
        }

        if (&ec != &hpx::throws)
            ec = hpx::make_success_code();
        return true;
    }

    // enable multiplexing with specified period for thread tix
    bool enable_multiplexing(int evset, int interval)
    {
        switch (PAPI_get_multiplex(evset))
        {
        case PAPI_OK:              // not yet multiplexed
            if (PAPI_assign_eventset_component(evset, 0) != PAPI_OK)
                return false;
            if (!interval)
            { // enable multiplexing with default interval
                return PAPI_set_multiplex(evset) != PAPI_OK;
            }
            break;
        case PAPI_EINVAL:          // already multiplexed
            if (!interval) return true;
            break; // still need to change interval
        default:                   // error
            return false;
        }
        // force the requested interval
        PAPI_option_t popt;
        popt.multiplex.eventset = evset;
        popt.multiplex.ns = interval;
        popt.multiplex.flags = PAPI_MULTIPLEX_DEFAULT;
        return PAPI_set_opt(PAPI_MULTIPLEX, &popt) == PAPI_OK;
    }

    // startup function for PAPI counter component
    void startup()
    {
        using namespace hpx::performance_counters;

        // define & install generic PAPI counter type
        generic_counter_type_data const papi_cnt_type =
        {
            "/papi",
            counter_raw,
            "the current count of occurrences of a specific PAPI event",
            HPX_PERFORMANCE_COUNTER_V1,
            &create_papi_counter,
            &discover_papi_counters
        };
        install_counter_types(&papi_cnt_type, 1);

        // deferred options
        variables_map vm = util::get_options();
        if (vm.count("papi-event-info"))
        {
            std::string v = vm["papi-event-info"].as<std::string>();
            util::list_events(v);
        }
    }

    bool check_startup(HPX_STD_FUNCTION<void()>& startup_func, bool& pre_startup)
    {
        // PAPI initialization
        if (PAPI_is_initialized() == PAPI_NOT_INITED)
        {
            if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT)
            {
                HPX_THROW_EXCEPTION(hpx::no_success,
                    "hpx::performance_counters::papi::check_startup()",
                    "PAPI library initialization failed (version mismatch)");
            }
        }
        // retrieve command line
        variables_map vm = util::get_options();

        if (util::check_options(vm))
        { // perform full module startup (counters will be used)
            startup_func = startup;
            pre_startup = true;
            return true;
        }

        return false;
    }

}}}

///////////////////////////////////////////////////////////////////////////////
// register a startup function for PAPI performance counter
HPX_REGISTER_STARTUP_MODULE(hpx::performance_counters::papi::check_startup);
// register related command line options
HPX_REGISTER_COMMANDLINE_MODULE(hpx::performance_counters::papi::util::get_options_description);
