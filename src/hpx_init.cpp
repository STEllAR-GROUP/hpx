//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c)      2017 Shoshana Jakobovits
//  Copyright (c) 2010-2011 Phillip LeBlanc, Dylan Stark
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>

#include <hpx/hpx_user_main_config.hpp>
#include <hpx/apply.hpp>
#include <hpx/async.hpp>
#include <hpx/compat/mutex.hpp>
#include <hpx/runtime_impl.hpp>
#include <hpx/runtime/agas/addressing_service.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/components/runtime_support.hpp>
#include <hpx/runtime/config_entry.hpp>
#include <hpx/runtime/find_localities.hpp>
#include <hpx/runtime/shutdown_function.hpp>
#include <hpx/runtime/startup_function.hpp>
#include <hpx/runtime/threads/policies/schedulers.hpp>
#include <hpx/util/apex.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/bind_action.hpp>
#include <hpx/util/command_line_handling.hpp>
#include <hpx/util/function.hpp>
#include <hpx/util/init_logging.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/query_counters.hpp>

#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

#if defined(HPX_NATIVE_MIC) || defined(__bgq__)
#  include <cstdlib>
#endif

#include <cmath>
#include <cstddef>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <new>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#if !defined(HPX_WINDOWS)
#  include <signal.h>
#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    void set_error_handlers();
}

///////////////////////////////////////////////////////////////////////////////
namespace hpx_startup
{
    std::vector<std::string> (*user_main_config_function)(
        std::vector<std::string> const&) = nullptr;
}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace detail
{
    // forward declarations only
    void console_print(std::string const&);
    void list_symbolic_name(std::string const&, naming::gid_type const&);
    void list_component_type(std::string const&, components::component_type);
}}

HPX_PLAIN_ACTION_ID(hpx::detail::console_print,
    console_print_action, hpx::actions::console_print_action_id)
HPX_PLAIN_ACTION_ID(hpx::detail::list_symbolic_name,
    list_symbolic_name_action, hpx::actions::list_symbolic_name_action_id)
HPX_PLAIN_ACTION_ID(hpx::detail::list_component_type,
    list_component_type_action, hpx::actions::list_component_type_action_id)

typedef
    hpx::util::detail::bound_action<
        list_symbolic_name_action
      , hpx::util::tuple<
            hpx::naming::id_type
          , hpx::util::detail::placeholder<1>
          , hpx::util::detail::placeholder<2>
        >
    >
    bound_list_symbolic_name_action;

HPX_UTIL_REGISTER_FUNCTION_DECLARATION(
    void(std::string const&, const hpx::naming::gid_type&)
  , bound_list_symbolic_name_action
  , list_symbolic_name_function)

HPX_UTIL_REGISTER_FUNCTION(
    void(std::string const&, const hpx::naming::gid_type&)
  , bound_list_symbolic_name_action
  , list_symbolic_name_function)

typedef
    hpx::util::detail::bound_action<
        list_component_type_action
      , hpx::util::tuple<
            hpx::naming::id_type
          , hpx::util::detail::placeholder<1>
          , hpx::util::detail::placeholder<2>
        >
    >
    bound_list_component_type_action;

HPX_UTIL_REGISTER_FUNCTION_DECLARATION(
    void(std::string const&, hpx::components::component_type)
  , bound_list_component_type_action
  , list_component_type_function)

HPX_UTIL_REGISTER_FUNCTION(
    void(std::string const&, hpx::components::component_type)
  , bound_list_component_type_action
  , list_component_type_function)

namespace hpx { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    // print string on the console
    void console_print(std::string const& name)
    {
        std::cout << name << std::endl;
    }

    inline void print(std::string const& name, error_code& ec = throws)
    {
        naming::id_type console(agas::get_console_locality(ec));
        if (ec) return;

        hpx::async<console_print_action>(console, name).get(ec);
        if (ec) return;

        if (&ec != &throws)
            ec = make_success_code();
    }

    ///////////////////////////////////////////////////////////////////////////
    // redirect the printing of the given counter name to the console
    bool list_counter(performance_counters::counter_info const& info,
        error_code& ec)
    {
        print(info.fullname_, ec);
        return true;
    }

    // List the names of all registered performance counters.
    void list_counter_names_header(bool skeleton)
    {
        // print header
        print("List of available counter instances");
        if (skeleton)
            print("(replace '*' below with the appropriate sequence number)");
        print(std::string(78, '-'));
    }

    void list_counter_names_minimal()
    {
        // list all counter names
        list_counter_names_header(true);
        performance_counters::discover_counter_types(&list_counter,
            performance_counters::discover_counters_minimal);
    }

    void list_counter_names_full()
    {
        // list all counter names
        list_counter_names_header(false);
        performance_counters::discover_counter_types(&list_counter,
            performance_counters::discover_counters_full);
    }

    ///////////////////////////////////////////////////////////////////////////
    // redirect the printing of the full counter info to the console
    bool list_counter_info(performance_counters::counter_info const& info,
        error_code& ec)
    {
        // compose the information to be printed for each of the counters
        std::ostringstream strm;

        strm << std::string(78, '-') << '\n';
        strm << "fullname: " << info.fullname_ << '\n';
        strm << "helptext: " << info.helptext_ << '\n';
        strm << "type:     "
              << performance_counters::get_counter_type_name(info.type_)
              << '\n';

        boost::format fmt("%d.%d.%d\n");
        strm << "version:  "        // 0xMMmmrrrr
              << boost::str(fmt % (info.version_ / 0x1000000) %
                    (info.version_ / 0x10000 % 0x100) %
                    (info.version_ % 0x10000));
        strm << std::string(78, '-') << '\n';

        print(strm.str(), ec);

        if (&ec != &throws)
            ec = make_success_code();
        return true;
    }

    // List the names of all registered performance counters.
    void list_counter_infos_header(bool skeleton)
    {
        // print header
        print("Information about available counter instances");
        if (skeleton)
            print("(replace '*' below with the appropriate sequence number)");
        print("--------------------------------------------------------");
    }

    void list_counter_infos_minimal()
    {
        // list all counter information
        list_counter_infos_header(true);
        performance_counters::discover_counter_types(&list_counter_info,
            performance_counters::discover_counters_minimal);
    }

    void list_counter_infos_full()
    {
        // list all counter information
        list_counter_infos_header(false);
        performance_counters::discover_counter_types(&list_counter_info,
            performance_counters::discover_counters_full);
    }

    ///////////////////////////////////////////////////////////////////////////
    void list_symbolic_name(std::string const& name, naming::gid_type const& gid)
    {
        std::ostringstream strm;

        strm << name << ", " << gid << ", "
             << (naming::detail::has_credits(gid) ? "managed" : "unmanaged");

        print(strm.str());
    }

    void list_symbolic_names()
    {
        print(std::string("List of all registered symbolic names:"));
        print(std::string("--------------------------------------"));

        using hpx::util::placeholders::_1;
        using hpx::util::placeholders::_2;

        naming::id_type console(agas::get_console_locality());
        naming::get_agas_client().iterate_ids(
            hpx::util::bind<list_symbolic_name_action>(console, _1, _2));
    }

    ///////////////////////////////////////////////////////////////////////////
    void list_component_type(std::string const& name,
        components::component_type ctype)
    {
        print(boost::str(boost::format("%1%, %|40t|%2%") %
            name % components::get_component_type_name(ctype)));
    }

    void list_component_types()
    {
        print(std::string("List of all registered component types:"));
        print(std::string("---------------------------------------"));

        using hpx::util::placeholders::_1;
        using hpx::util::placeholders::_2;

        naming::id_type console(agas::get_console_locality());
        naming::get_agas_client().iterate_types(
            hpx::util::bind<list_component_type_action>(console, _1, _2));
    }

    ///////////////////////////////////////////////////////////////////////////
    void start_counters(std::shared_ptr<util::query_counters> const& qc)
    {
        try {
            HPX_ASSERT(qc);
            qc->start();
        }
        catch (...) {
            std::cerr << hpx::diagnostic_information(std::current_exception())
                << std::flush;
            hpx::terminate();
        }
    }
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    // Print stack trace and exit.
#if defined(HPX_WINDOWS)
    extern BOOL WINAPI termination_handler(DWORD ctrl_type);
#else
    extern void termination_handler(int signum);
#endif

    extern void new_handler();

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        struct dump_config
        {
            dump_config(hpx::runtime const& rt) : rt_(std::cref(rt)) {}

            void operator()() const
            {
                std::cout << "Configuration after runtime start:\n";
                std::cout << "----------------------------------\n";
                rt_.get().get_config().dump(0, std::cout);
                std::cout << "----------------------------------\n";
            }

            std::reference_wrapper<hpx::runtime const> rt_;
        };

        ///////////////////////////////////////////////////////////////////////
        void handle_list_and_print_options(hpx::runtime& rt,
            boost::program_options::variables_map& vm,
            bool print_counters_locally)
        {
            if (vm.count("hpx:list-counters")) {
                // Print the names of all registered performance counters.
                std::string option(vm["hpx:list-counters"].as<std::string>());
                if (0 == std::string("minimal").find(option))
                    rt.add_startup_function(&list_counter_names_minimal);
                else if (0 == std::string("full").find(option))
                    rt.add_startup_function(&list_counter_names_full);
                else {
                    std::string msg ("Invalid command line option value"
                        "for --hpx:list-counters: ");
                    msg += option;
                    msg += ", allowed values are 'minimal' and 'full'";
                    throw detail::command_line_error(msg.c_str());
                }
            }
            if (vm.count("hpx:list-counter-infos")) {
                // Print info about all registered performance counters.
                std::string option(vm["hpx:list-counter-infos"].as<std::string>());
                if (0 == std::string("minimal").find(option))
                    rt.add_startup_function(&list_counter_infos_minimal);
                else if (0 == std::string("full").find(option))
                    rt.add_startup_function(&list_counter_infos_full);
                else {
                    std::string msg ("Invalid command line option value"
                        "for --hpx:list-counter-infos: ");
                    msg += option;
                    msg += ", allowed values are 'minimal' and 'full'";
                    throw detail::command_line_error(msg.c_str());
                }
            }
            if (vm.count("hpx:list-symbolic-names")) {
                // Print all registered symbolic names.
                rt.add_startup_function(&list_symbolic_names);
            }
            if (vm.count("hpx:list-component-types")) {
                // Print all registered component types.
                rt.add_startup_function(&list_component_types);
            }

            if (vm.count("hpx:print-counter") || vm.count("hpx:print-counter-reset"))
            {
                std::size_t interval = 0;
                if (vm.count("hpx:print-counter-interval"))
                    interval = vm["hpx:print-counter-interval"].as<std::size_t>();

                std::vector<std::string> counters;
                if (vm.count("hpx:print-counter"))
                {
                    counters = vm["hpx:print-counter"]
                        .as<std::vector<std::string> >();
                }
                std::vector<std::string> reset_counters;
                if (vm.count("hpx:print-counter-reset"))
                {
                    reset_counters = vm["hpx:print-counter-reset"]
                        .as<std::vector<std::string> >();
                }

                std::vector<std::string> counter_shortnames;
                std::string counter_format("normal");
                if (vm.count("hpx:print-counter-format")) {
                    counter_format = vm["hpx:print-counter-format"].as<std::string>();
                    if (counter_format == "csv-short"){
                        for (std::size_t i = 0; i != counters.size() ; ++i) {
                            std::vector<std::string> entry;
                            boost::algorithm::split(entry, counters[i],
                                boost::algorithm::is_any_of(","),
                                boost::algorithm::token_compress_on);

                            if (entry.size() != 2)
                            {
                                throw detail::command_line_error(
                                    "Invalid format for command line option "
                                    "--hpx:print-counter-format=csv-short");
                            }

                            counter_shortnames.push_back(entry[0]);
                            counters[i] = entry[1];
                        }
                    }
                }

                bool csv_header = true;
                if(vm.count("hpx:no-csv-header"))
                    csv_header = false;

                std::string destination("cout");
                if (vm.count("hpx:print-counter-destination"))
                    destination = vm["hpx:print-counter-destination"].as<std::string>();

                // schedule the query function at startup, which will schedule
                // itself to run after the given interval
                std::shared_ptr<util::query_counters> qc =
                    std::make_shared<util::query_counters>(
                        std::ref(counters), std::ref(reset_counters), interval,
                        destination, counter_format, counter_shortnames, csv_header,
                        print_counters_locally);

                // schedule to print counters at shutdown, if requested
                if (get_config_entry("hpx.print_counter.shutdown", "0") == "1")
                {
                    // schedule to run at shutdown
                    rt.add_pre_shutdown_function(
                        util::bind(&util::query_counters::evaluate, qc));
                }

                // schedule to start all counters
                rt.add_startup_function(util::bind(&start_counters, qc));

                // register the query_counters object with the runtime system
                rt.register_query_counters(qc);
            }
            else if (vm.count("hpx:print-counter-interval")) {
                throw detail::command_line_error("Invalid command line option "
                    "--hpx:print-counter-interval, valid in conjunction with "
                    "--hpx:print-counter only");
            }
            else if (vm.count("hpx:print-counter-destination")) {
                throw detail::command_line_error("Invalid command line option "
                    "--hpx:print-counter-destination, valid in conjunction with "
                    "--hpx:print-counter only");
            }
            else if (vm.count("hpx:print-counter-format")) {
                throw detail::command_line_error("Invalid command line option "
                    "--hpx:print-counter-format, valid in conjunction with "
                    "--hpx:print-counter only");
            }
            else if (vm.count("hpx:print-counter-at")) {
                throw detail::command_line_error("Invalid command line option "
                    "--hpx:print-counter-at, valid in conjunction with "
                    "--hpx:print-counter only");
            }
            else if (vm.count("hpx:reset-counters")) {
                throw detail::command_line_error("Invalid command line option "
                    "--hpx:reset-counters, valid in conjunction with "
                    "--hpx:print-counter only");
            }
        }

        void add_startup_functions(hpx::runtime& rt,
            boost::program_options::variables_map& vm, runtime_mode mode,
            startup_function_type startup, shutdown_function_type shutdown)
        {
            if (vm.count("hpx:app-config"))
            {
                std::string config(vm["hpx:app-config"].as<std::string>());
                rt.get_config().load_application_configuration(config.c_str());
            }

            if (!!startup)
                rt.add_startup_function(std::move(startup));

            if (!!shutdown)
                rt.add_shutdown_function(std::move(shutdown));

            // Add startup function related to listing counter names or counter
            // infos (on console only).
            bool print_counters_locally =
                vm.count("hpx:print-counters-locally") != 0;
            if (mode == runtime_mode_console || print_counters_locally)
                handle_list_and_print_options(rt, vm, print_counters_locally);

            // Dump the configuration before all components have been loaded.
            if (vm.count("hpx:dump-config-initial")) {
                std::cout << "Configuration after runtime construction:\n";
                std::cout << "-----------------------------------------\n";
                rt.get_config().dump(0, std::cout);
                std::cout << "-----------------------------------------\n";
            }

            // Dump the configuration after all components have been loaded.
            if (vm.count("hpx:dump-config"))
                rt.add_startup_function(dump_config(rt));
        }

        ///////////////////////////////////////////////////////////////////////
        int run(hpx::runtime& rt,
            util::function_nonser<int(boost::program_options::variables_map& vm)>
                const& f,
            boost::program_options::variables_map& vm, runtime_mode mode,
            startup_function_type startup, shutdown_function_type shutdown)
        {
            LPROGRESS_;

            add_startup_functions(rt, vm, mode, std::move(startup),
                std::move(shutdown));

            // Run this runtime instance using the given function f.
            if (!f.empty())
                return rt.run(util::bind(f, vm));

            // Run this runtime instance without an hpx_main
            return rt.run();
        }

        int start(hpx::runtime& rt,
            util::function_nonser<int(boost::program_options::variables_map& vm)>
                const& f,
            boost::program_options::variables_map& vm, runtime_mode mode,
            startup_function_type startup, shutdown_function_type shutdown)
        {
            LPROGRESS_;

            add_startup_functions(rt, vm, mode, std::move(startup),
                std::move(shutdown));

            if (!f.empty()) {
                // Run this runtime instance using the given function f.
                return rt.start(util::bind(f, vm));
            }

            // Run this runtime instance without an hpx_main
            return rt.start();
        }

        int run_or_start(bool blocking, std::unique_ptr<hpx::runtime> rt,
            util::command_line_handling& cfg,
            startup_function_type startup, shutdown_function_type shutdown)
        {
            if (blocking) {
                return run(*rt, cfg.hpx_main_f_, cfg.vm_, cfg.mode_,
                    std::move(startup), std::move(shutdown));
            }

            // non-blocking version
            start(*rt, cfg.hpx_main_f_, cfg.vm_, cfg.mode_, std::move(startup),
                std::move(shutdown));

            rt.release();          // pointer to runtime is stored in TLS
            return 0;
        }



        ///////////////////////////////////////////////////////////////////////

        HPX_EXPORT int run_or_start(
            util::function_nonser<
                int(boost::program_options::variables_map& vm)
            > const& f,
            boost::program_options::options_description const& desc_cmdline,
            int argc, char** argv, std::vector<std::string> && ini_config,
            startup_function_type startup,
            shutdown_function_type shutdown, hpx::runtime_mode mode,
            bool blocking)
        {
#if !defined(HPX_HAVE_DISABLED_SIGNAL_EXCEPTION_HANDLERS)
            set_error_handlers();
#endif

#if defined(HPX_NATIVE_MIC) || defined(__bgq__) || defined(__bgqion__)
            unsetenv("LANG");
            unsetenv("LC_CTYPE");
            unsetenv("LC_NUMERIC");
            unsetenv("LC_TIME");
            unsetenv("LC_COLLATE");
            unsetenv("LC_MONETARY");
            unsetenv("LC_MESSAGES");
            unsetenv("LC_PAPER");
            unsetenv("LC_NAME");
            unsetenv("LC_ADDRESS");
            unsetenv("LC_TELEPHONE");
            unsetenv("LC_MEASUREMENT");
            unsetenv("LC_IDENTIFICATION");
            unsetenv("LC_ALL");
#endif

            int result = 0;
            try {
                // make sure the runtime system is not active yet
                if (get_runtime_ptr() != nullptr)
                {
                    std::cerr << "hpx::init: can't initialize runtime system "
                        "more than once! Exiting...\n";
                    return -1;
                }

                // handle all common command line switches
                util::command_line_handling cfg(
                    mode, f, std::move(ini_config), argv[0]);

                result = cfg.call(desc_cmdline, argc, argv);

                if (result != 0) {
                    if (result > 0)
                        result = 0;     // --hpx:help
                    return result;
                }

                // Setup all parameters of the resource_partitioner
                auto &rp = hpx::get_resource_partitioner();
                rp.init_resources(cfg);

                // initialize logging
                util::detail::init_logging(cfg.rtcfg_,
                    cfg.mode_ == runtime_mode_console);

                util::apex_wrapper_init apex(argc, argv);

                // Initialize and start the HPX runtime.
                LPROGRESS_ << "run_local: create runtime";

                // Build and configure this runtime instance.
                typedef hpx::runtime_impl runtime_type;
                std::unique_ptr<hpx::runtime> rt(
                        new runtime_type(cfg.rtcfg_, cfg, cfg.mode_, cfg.num_threads_));
                //! FIXME give only the parts of cfg that are needed for the initialization of the scheduler

                result = run_or_start(blocking, std::move(rt), cfg,
                                    std::move(startup), std::move(shutdown));
            }
            catch (detail::command_line_error const& e) {
                std::cerr << "{env}: " << hpx::detail::get_execution_environment();
                std::cerr << "hpx::init: std::exception caught: " << e.what() << "\n";
                return -1;
            }
            return result;
        }

        template <typename T>
        inline T
        get_option(std::string const& config, T default_ = T())
        {
            if (!config.empty()) {
                try {
                    return boost::lexical_cast<T>(
                        get_runtime().get_config().get_entry(config, default_));
                }
                catch (boost::bad_lexical_cast const&) {
                    ;   // do nothing
                }
            }
            return default_;
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    int finalize(double shutdown_timeout, double localwait, error_code& ec)
    {
        if (!threads::get_self_ptr()) {
            HPX_THROWS_IF(ec, invalid_status, "hpx::finalize",
                "this function can be called from an HPX thread only");
            return -1;
        }

        if (!is_running()) {
            HPX_THROWS_IF(ec, invalid_status, "hpx::finalize",
                "the runtime system is not active (did you already "
                "call finalize?)");
            return -1;
        }

        if (&ec != &throws)
            ec = make_success_code();

        if (std::abs(localwait + 1.0) < 1e-16)
            localwait = detail::get_option("hpx.finalize_wait_time", -1.0);

        {
            hpx::util::high_resolution_timer t;
            double start_time = t.elapsed();
            double current = 0.0;
            do {
                current = t.elapsed();
            } while (current - start_time < localwait * 1e-6);
        }

        if (std::abs(shutdown_timeout + 1.0) < 1e-16)
            shutdown_timeout = detail::get_option("hpx.shutdown_timeout", -1.0);

        // tell main locality to start application exit, duplicated requests
        // will be ignored
        apply<components::server::runtime_support::shutdown_all_action>(
            hpx::find_root_locality(), shutdown_timeout);

        //util::apex_finalize();
        return 0;
    }

    ///////////////////////////////////////////////////////////////////////////
    int disconnect(double shutdown_timeout, double localwait, error_code& ec)
    {
        if (!threads::get_self_ptr()) {
            HPX_THROWS_IF(ec, invalid_status, "hpx::disconnect",
                "this function can be called from an HPX thread only");
            return -1;
        }

        if (!is_running()) {
            HPX_THROWS_IF(ec, invalid_status, "hpx::disconnect",
                "the runtime system is not active (did you already "
                "call finalize?)");
            return -1;
        }

        if (&ec != &throws)
            ec = make_success_code();

        if (std::abs(localwait + 1.0) < 1e-16)
            localwait = detail::get_option("hpx.finalize_wait_time", -1.0);

        {
            hpx::util::high_resolution_timer t;
            double start_time = t.elapsed();
            double current = 0.0;
            do {
                current = t.elapsed();
            } while (current - start_time < localwait * 1e-6);
        }

        if (std::abs(shutdown_timeout + 1.0) < 1e-16)
            shutdown_timeout = detail::get_option("hpx.shutdown_timeout", -1.0);

        //util::apex_finalize();

        components::server::runtime_support* p =
            reinterpret_cast<components::server::runtime_support*>(
                  get_runtime().get_runtime_support_lva());

        if (nullptr == p) {
            HPX_THROWS_IF(ec, invalid_status, "hpx::disconnect",
                "the runtime system is not active (did you already "
                "call finalize?)");
            return -1;
        }

        p->call_shutdown_functions(true);
        p->call_shutdown_functions(false);

        p->stop(shutdown_timeout, naming::invalid_id, true);

        return 0;
    }

    ///////////////////////////////////////////////////////////////////////////
    void terminate()
    {
        if (!threads::get_self_ptr()) {
            // hpx::terminate shouldn't be called from a non-HPX thread
            std::terminate();
        }

        components::server::runtime_support* p =
            reinterpret_cast<components::server::runtime_support*>(
                  get_runtime().get_runtime_support_lva());

        if (nullptr == p) {
            // the runtime system is not running, just terminate
            std::terminate();
        }

        p->terminate_all();
    }

    ///////////////////////////////////////////////////////////////////////////
    int stop(error_code& ec)
    {
        if (threads::get_self_ptr()) {
            HPX_THROWS_IF(ec, invalid_status, "hpx::stop",
                "this function cannot be called from an HPX thread");
            return -1;
        }

        std::unique_ptr<runtime> rt(get_runtime_ptr());    // take ownership!
        if (nullptr == rt.get()) {
            HPX_THROWS_IF(ec, invalid_status, "hpx::stop",
                "the runtime system is not active (did you already "
                "call hpx::stop?)");
            return -1;
        }

        int result = rt->wait();

        rt->stop();
        rt->rethrow_exception();

        return result;
    }

    namespace detail
    {
        HPX_EXPORT int init_helper(
            boost::program_options::variables_map& /*vm*/,
            util::function_nonser<int(int, char**)> const& f)
        {
            std::string cmdline(hpx::get_config_entry("hpx.reconstructed_cmd_line", ""));

            using namespace boost::program_options;
#if defined(HPX_WINDOWS)
            std::vector<std::string> args = split_winmain(cmdline);
#else
            std::vector<std::string> args = split_unix(cmdline);
#endif

            // Copy all arguments which are not hpx related to a temporary array
            std::vector<char*> argv(args.size()+1);
            std::size_t argcount = 0;
            for (std::size_t i = 0; i != args.size(); ++i)
            {
                if (0 != args[i].find("--hpx:")) {
                    argv[argcount++] = const_cast<char*>(args[i].data());
                }
                else if (6 == args[i].find("positional", 6)) {
                    std::string::size_type p = args[i].find_first_of("=");
                    if (p != std::string::npos) {
                        args[i] = args[i].substr(p+1);
                        argv[argcount++] = const_cast<char*>(args[i].data());
                    }
                }
            }

            // add a single nullptr in the end as some application rely on that
            argv[argcount] = nullptr;

            // Invoke custom startup functions
            return f(static_cast<int>(argcount), argv.data());
        }
    }
}
