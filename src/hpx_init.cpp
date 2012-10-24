//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2010-2011 Phillip LeBlanc, Dylan Stark
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/config.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/util/command_line_handling.hpp>
#include <hpx/util/bind_action.hpp>

#include <hpx/include/async.hpp>
#include <hpx/runtime/components/runtime_support.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/runtime/threads/policies/schedulers.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime_impl.hpp>
#include <hpx/util/query_counters.hpp>
#include <hpx/util/stringstream.hpp>
#include <hpx/util/function.hpp>

#if !defined(BOOST_WINDOWS)
#  include <signal.h>
#endif

#include <iostream>
#include <vector>
#include <new>

#include <boost/shared_ptr.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>
#include <boost/assign/std/vector.hpp>
#include <boost/foreach.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace detail
{
    // forward declarations only
    void console_print(std::string const&);
    void list_symbolic_name(std::string const&, naming::gid_type const&);
    void list_component_type(std::string const&, components::component_type);
}}

HPX_PLAIN_ACTION(hpx::detail::console_print,
    console_print_action, hpx::components::factory_enabled)
HPX_PLAIN_ACTION(hpx::detail::list_symbolic_name,
    list_symbolic_name_action, hpx::components::factory_enabled)
HPX_PLAIN_ACTION(hpx::detail::list_component_type,
    list_component_type_action, hpx::components::factory_enabled)

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
    void list_counter_names()
    {
        // print header
        print("List of available counter instances");
        print("(replace <*> below with the appropriate sequence number)");
        print(std::string(78, '-'));

        // list all counter names
        performance_counters::discover_counter_types(&list_counter);
    }

    ///////////////////////////////////////////////////////////////////////////
    // redirect the printing of the full counter info to the console
    bool list_counter_info(performance_counters::counter_info const& info,
        error_code& ec)
    {
        // compose the information to be printed for each of the counters
        util::osstream strm;

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
    void list_counter_infos()
    {
        // print header
        print("Information about available counter instances");
        print("(replace <*> below with the appropriate sequence number)");

        // list all counter information
        performance_counters::discover_counter_types(&list_counter_info);
    }

    ///////////////////////////////////////////////////////////////////////////
    void list_symbolic_name(std::string const& name, naming::gid_type const& gid)
    {
        util::osstream strm;

        strm << name << ", " << gid << ", "
             << (naming::get_credit_from_gid(gid) ? "managed" : "unmanaged");

        print(strm.str());
    }

    void list_symbolic_names()
    {
        print(std::string("List of all registered symbolic names:"));
        print(std::string(78, '-'));

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
        print(std::string(78, '-'));

        using hpx::util::placeholders::_1;
        using hpx::util::placeholders::_2;

        naming::id_type console(agas::get_console_locality());
        naming::get_agas_client().iterate_types(
            hpx::util::bind<list_component_type_action>(console, _1, _2));
    }

    ///////////////////////////////////////////////////////////////////////////
    void print_counters(boost::shared_ptr<util::query_counters> const& qc)
    {
        BOOST_ASSERT(qc);
        qc->start();
    }
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    // Print stack trace and exit.
#if defined(BOOST_WINDOWS)
    extern BOOL termination_handler(DWORD ctrl_type);
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
            dump_config(hpx::runtime const& rt) : rt_(boost::cref(rt)) {}

            void operator()() const
            {
                std::cout << "Configuration after runtime start:\n";
                std::cout << "----------------------------------\n";
                rt_.get().get_config().dump(0, std::cout);
                std::cout << "----------------------------------\n";
            }

            boost::reference_wrapper<hpx::runtime const> rt_;
        };

        ///////////////////////////////////////////////////////////////////////
        void handle_list_and_print_options(hpx::runtime& rt,
            boost::program_options::variables_map& vm)
        {
            if (vm.count("hpx:list-counters")) {
                // Print the names of all registered performance counters.
                rt.add_startup_function(&list_counter_names);
            }
            if (vm.count("hpx:list-counter-infos")) {
                // Print info about all registered performance counters.
                rt.add_startup_function(&list_counter_infos);
            }
            if (vm.count("hpx:list-symbolic-names")) {
                // Print all registered symbolic names.
                rt.add_startup_function(&list_symbolic_names);
            }
            if (vm.count("hpx:list-component-types")) {
                // Print all registered component types.
                rt.add_startup_function(&list_component_types);
            }

            if (vm.count("hpx:print-counter")) {
                std::size_t interval = 0;
                if (vm.count("hpx:print-counter-interval"))
                    interval = vm["hpx:print-counter-interval"].as<std::size_t>();

                std::vector<std::string> counters =
                    vm["hpx:print-counter"].as<std::vector<std::string> >();

                std::string destination("cout");
                if (vm.count("hpx:print-counter-destination"))
                    destination = vm["hpx:print-counter-destination"].as<std::string>();

                // schedule the query function at startup, which will schedule
                // itself to run after the given interval
                boost::shared_ptr<util::query_counters> qc =
                    boost::make_shared<util::query_counters>(
                        boost::ref(counters), interval, destination);

                // schedule to run at shutdown
                rt.add_pre_shutdown_function(
                    boost::bind(&util::query_counters::evaluate, qc));

                // schedule to start all counters
                rt.add_startup_function(boost::bind(&print_counters, qc));
            }
            else if (vm.count("hpx:print-counter-interval")) {
                throw std::logic_error("Invalid command line option "
                    "--hpx:print-counter-interval, valid in conjunction with "
                    "--hpx:print-counter only");
            }
            else if (vm.count("hpx:print-counter-destination")) {
                throw std::logic_error("Invalid command line option "
                    "--hpx:print-counter-destination, valid in conjunction with "
                    "--hpx:print-counter only");
            }
        }

        void add_startup_functions(hpx::runtime& rt, 
            boost::program_options::variables_map& vm, runtime_mode mode,
            startup_function_type const& startup,
            shutdown_function_type const& shutdown)
        {
            if (vm.count("hpx:app-config"))
            {
                std::string config(vm["hpx:app-config"].as<std::string>());
                rt.get_config().load_application_configuration(config.c_str());
            }

            if (!!startup)
                rt.add_startup_function(startup);

            if (!!shutdown)
                rt.add_shutdown_function(shutdown);

            // Add startup function related to listing counter names or counter
            // infos (on console only).
            if (mode == runtime_mode_console)
                handle_list_and_print_options(rt, vm);

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

        int run(hpx::runtime& rt, hpx_main_type f,
            boost::program_options::variables_map& vm, runtime_mode mode,
            startup_function_type const& startup,
            shutdown_function_type const& shutdown, std::size_t num_threads,
            std::size_t num_localities)
        {
            add_startup_functions(rt, vm, mode, startup, shutdown);

            // Run this runtime instance using the given function f.
            if (0 != f)
                return rt.run(boost::bind(f, vm), num_threads, num_localities);

            // Run this runtime instance without an hpx_main
            return rt.run(num_threads, num_localities);
        }

        // helper function testing option compatibility
        void ensure_queuing_option_compatibility(
            boost::program_options::variables_map const& vm)
        {
            if (vm.count("hpx:high-priority-threads")) {
                throw std::logic_error("Invalid command line option "
                    "--hpx:high-priority-threads, valid for "
                    "--hpx:queuing=priority_local only");
            }
            if (vm.count("hpx:numa-sensitive")) {
                throw std::logic_error("Invalid command line option "
                    "--hpx:numa-sensitive, valid for "
                    "--hpx:queuing=priority_local or priority_abp only");
            }
            if (vm.count("hpx:hierarchy-arity")) {
                throw std::logic_error("Invalid command line option "
                    "--hpx:hierarchy-arity, valid for --hpx:queuing=hierarchy only.");
            }
        }

        void ensure_hwloc_compatibility(
            boost::program_options::variables_map const& vm)
        {
#if defined(HPX_HAVE_HWLOC)
            if (vm.count("hpx:pu-offset")) {
                throw std::logic_error("Invalid command line option "
                    "--hpx:pu-offset, valid for --hpx:queuing=priority_local only.");
            }
            if (vm.count("hpx:pu-step")) {
                throw std::logic_error("Invalid command line option "
                    "--hpx:pu-step, valid for --hpx:queuing=priority_local only.");
            }
#endif
        }

#if defined(HPX_GLOBAL_SCHEDULER)
        ///////////////////////////////////////////////////////////////////////
        // global scheduler (one queue for all OS threads)
        int run_global(util::runtime_configuration& rtcfg,
            hpx_main_type f, boost::program_options::variables_map& vm,
            runtime_mode mode, startup_function_type const& startup,
            shutdown_function_type const& shutdown, std::size_t num_threads,
            std::size_t num_localities)
        {
            ensure_queuing_option_compatibility(vm);
            ensure_hwloc_compatibility(vm);

            // scheduling policy
            typedef hpx::threads::policies::global_queue_scheduler
                global_queue_policy;

            global_queue_policy::init_parameter_type init;

            // Build and configure this runtime instance.
            typedef hpx::runtime_impl<global_queue_policy> runtime_type;
            runtime_type rt(rtcfg, mode, init);

            return run(rt, f, vm, mode, startup, shutdown, num_threads,
                num_localities);
        }
#endif

#if defined(HPX_LOCAL_SCHEDULER)
        ///////////////////////////////////////////////////////////////////////
        // local scheduler (one queue for each OS threads)
        int run_local(util::runtime_configuration& rtcfg,
            hpx_main_type f, boost::program_options::variables_map& vm,
            runtime_mode mode, startup_function_type const& startup,
            shutdown_function_type const& shutdown, std::size_t num_threads,
            std::size_t num_localities)
        {
            ensure_queuing_option_compatibility(vm);

            // scheduling policy
            typedef hpx::threads::policies::local_queue_scheduler
                local_queue_policy;
            local_queue_policy::init_parameter_type init(num_threads, 1000);

            // Build and configure this runtime instance.
            typedef hpx::runtime_impl<local_queue_policy> runtime_type;
            runtime_type rt(rtcfg, mode, init);

            return run(rt, f, vm, mode, startup, shutdown, num_threads,
                num_localities);
        }
#endif

        ///////////////////////////////////////////////////////////////////////
        // local scheduler with priority queue (one queue for each OS threads
        // plus one separate queue for high priority PX-threads)
        int run_priority_local(util::runtime_configuration& rtcfg,
            hpx_main_type f, boost::program_options::variables_map& vm,
            runtime_mode mode, startup_function_type const& startup,
            shutdown_function_type const& shutdown, std::size_t num_threads,
            std::size_t num_localities)
        {
            std::size_t num_high_priority_queues = num_threads;
            if (vm.count("hpx:high-priority-threads")) {
                num_high_priority_queues =
                    vm["hpx:high-priority-threads"].as<std::size_t>();
            }
            bool numa_sensitive = false;
            if (vm.count("hpx:numa-sensitive"))
                numa_sensitive = true;

            if (vm.count("hpx:hierarchy-arity")) {
                throw std::logic_error("Invalid command line option "
                    "--hpx:hierarchy-arity, valid for --hpx:queuing=hierarchy only.");
            }

            std::size_t pu_offset = 0;
            std::size_t pu_step = 1;
#if defined(HPX_HAVE_HWLOC) || defined(BOOST_WINDOWS)
            if (vm.count("hpx:pu-offset")) {
                pu_offset = vm["hpx:pu-offset"].as<std::size_t>();
                if (pu_offset >= hpx::threads::hardware_concurrency()) {
                    throw std::logic_error("Invalid command line option "
                        "--hpx:pu-offset, value must be smaller than number of "
                        "available processing units.");
                }
            }

            if (vm.count("hpx:pu-step")) {
                pu_step = vm["hpx:pu-step"].as<std::size_t>();
                if (pu_step == 0 || pu_step >= hpx::threads::hardware_concurrency()) {
                    throw std::logic_error("Invalid command line option "
                        "--hpx:pu-step, value must be non-zero smaller than number of "
                        "available processing units.");
                }
            }
#endif
            // scheduling policy
            typedef hpx::threads::policies::local_priority_queue_scheduler
                local_queue_policy;
            local_queue_policy::init_parameter_type init(
                num_threads, num_high_priority_queues, 1000, numa_sensitive,
                pu_offset, pu_step);

            // Build and configure this runtime instance.
            typedef hpx::runtime_impl<local_queue_policy> runtime_type;
            runtime_type rt(rtcfg, mode, init);

            return run(rt, f, vm, mode, startup, shutdown, num_threads,
                num_localities);
        }

#if defined(HPX_ABP_SCHEDULER)
        ///////////////////////////////////////////////////////////////////////
        // abp scheduler: local deques for each OS thread, with work
        // stealing from the "bottom" of each.
        int run_abp(util::runtime_configuration& rtcfg,
            hpx_main_type f, boost::program_options::variables_map& vm,
            runtime_mode mode, startup_function_type const& startup,
            shutdown_function_type const& shutdown, std::size_t num_threads,
            std::size_t num_localities)
        {
            ensure_queuing_option_compatibility(vm);
            ensure_hwloc_compatibility(vm);

            // scheduling policy
            typedef hpx::threads::policies::abp_queue_scheduler
                abp_queue_policy;
            abp_queue_policy::init_parameter_type init(num_threads, 1000);

            // Build and configure this runtime instance.
            typedef hpx::runtime_impl<abp_queue_policy> runtime_type;
            runtime_type rt(rtcfg, mode, init);

            return run(rt, f, vm, mode, startup, shutdown, num_threads,
                num_localities);
        }
#endif

#if defined(HPX_ABP_PRIORITY_SCHEDULER)
        ///////////////////////////////////////////////////////////////////////
        // priority abp scheduler: local priority deques for each OS thread,
        // with work stealing from the "bottom" of each.
        int run_priority_abp(util::runtime_configuration& rtcfg,
            hpx_main_type f, boost::program_options::variables_map& vm,
            runtime_mode mode, startup_function_type const& startup,
            shutdown_function_type const& shutdown, std::size_t num_threads,
            std::size_t num_localities)
        {
            std::size_t num_high_priority_queues = num_threads;
            if (vm.count("hpx:high-priority-threads")) {
                num_high_priority_queues =
                    vm["hpx:high-priority-threads"].as<std::size_t>();
            }
            bool numa_sensitive = false;
            if (vm.count("hpx:numa-sensitive"))
                numa_sensitive = true;
            if (vm.count("hpx:hierarchy-arity")) {
                throw std::logic_error("Invalid command line option "
                    "--hpx:hierarchy-arity, valid for --hpx:queuing=hierarchy only.");
            }

            ensure_hwloc_compatibility(vm);

            // scheduling policy
            typedef hpx::threads::policies::abp_priority_queue_scheduler
                abp_priority_queue_policy;
            abp_priority_queue_policy::init_parameter_type init(
                num_threads, num_high_priority_queues, 1000, numa_sensitive);

            // Build and configure this runtime instance.
            typedef hpx::runtime_impl<abp_priority_queue_policy> runtime_type;
            runtime_type rt(rtcfg, mode, init);

            return run(rt, f, vm, mode, startup, shutdown, num_threads,
                num_localities);
        }
#endif

#if defined(HPX_HIERARCHY_SCHEDULER)
        ///////////////////////////////////////////////////////////////////////
        // hierarchical scheduler: The thread queues are built up hierarchically
        // this avoids contention during work stealing
        int run_hierarchy(util::runtime_configuration& rtcfg,
            hpx_main_type f, boost::program_options::variables_map& vm,
            runtime_mode mode, startup_function_type const& startup,
            shutdown_function_type const& shutdown, std::size_t num_threads,
            std::size_t num_localities)
        {
            if (vm.count("hpx:high-priority-threads")) {
                throw std::logic_error("Invalid command line option "
                    "--hpx:high-priority-threads, valid for "
                    "--hpx:queuing=priority_local only.");
            }
            if (vm.count("hpx:numa-sensitive")) {
                throw std::logic_error("Invalid command line option "
                    "--hpx:numa-sensitive, valid for "
                    "--hpx:queuing=priority_local or priority_abp only");
            }

            ensure_hwloc_compatibility(vm);

            // scheduling policy
            typedef hpx::threads::policies::hierarchy_scheduler queue_policy;
            std::size_t arity = 2;
            if (vm.count("hpx:hierarchy-arity"))
                arity = vm["hpx:hierarchy-arity"].as<std::size_t>();
            queue_policy::init_parameter_type init(num_threads, arity, 1000);

            // Build and configure this runtime instance.
            typedef hpx::runtime_impl<queue_policy> runtime_type;
            runtime_type rt(rtcfg, mode, init);

            return run(rt, f, vm, mode, startup, shutdown, num_threads,
                num_localities);
        }
#endif

#if defined(HPX_PERIODIC_PRIORITY_SCHEDULER)
        ///////////////////////////////////////////////////////////////////////
        // hierarchical scheduler: The thread queues are built up hierarchically
        // this avoids contention during work stealing
        int run_periodic(util::runtime_configuration& rtcfg,
            hpx_main_type f, boost::program_options::variables_map& vm,
            runtime_mode mode, startup_function_type const& startup,
            shutdown_function_type const& shutdown, std::size_t num_threads,
            std::size_t num_localities)
        {
            std::size_t num_high_priority_queues = num_threads;
            if (vm.count("hpx:high-priority-threads")) {
                num_high_priority_queues =
                    vm["hpx:high-priority-threads"].as<std::size_t>();
            }
            bool numa_sensitive = false;
            if (vm.count("hpx:numa-sensitive"))
                numa_sensitive = true;
            if (vm.count("hpx:hierarchy-arity")) {
                throw std::logic_error("Invalid command line option "
                    "--hpx:hierarchy-arity, valid for --hpx:queuing=hierarchy only.");
            }

            ensure_hwloc_compatibility(vm);

            // scheduling policy
            typedef hpx::threads::policies::local_periodic_priority_scheduler
                local_queue_policy;
            local_queue_policy::init_parameter_type init(
                num_threads, num_high_priority_queues, 1000, numa_sensitive);

            // Build and configure this runtime instance.
            typedef hpx::runtime_impl<local_queue_policy> runtime_type;
            runtime_type rt(rtcfg, mode, init);

            return run(rt, f, vm, mode, startup, shutdown, num_threads,
                num_localities);
        }
#endif

        ///////////////////////////////////////////////////////////////////////
        void set_error_handlers()
        {
#if defined(BOOST_WINDOWS)
            // Set console control handler to allow server to be stopped.
            SetConsoleCtrlHandler(hpx::termination_handler, TRUE);
#else
            struct sigaction new_action;
            new_action.sa_handler = hpx::termination_handler;
            sigemptyset(&new_action.sa_mask);
            new_action.sa_flags = 0;

            sigaction(SIGINT, &new_action, NULL);  // Interrupted 
            sigaction(SIGBUS, &new_action, NULL);  // Bus error
            sigaction(SIGFPE, &new_action, NULL);  // Floating point exception
            sigaction(SIGILL, &new_action, NULL);  // Illegal instruction
            sigaction(SIGPIPE, &new_action, NULL); // Bad pipe
            sigaction(SIGSEGV, &new_action, NULL); // Segmentation fault
            sigaction(SIGSYS, &new_action, NULL);  // Bad syscall
#endif

            std::set_new_handler(hpx::new_handler);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    int init(hpx_main_type f,
        boost::program_options::options_description& desc_cmdline,
        int argc, char* argv[], std::vector<std::string> const& ini_config,
        startup_function_type const& startup, 
        shutdown_function_type const& shutdown, hpx::runtime_mode mode)
    {
        int result = 0;
        detail::set_error_handlers();

        try {
            // load basic ini configuration information to allow for command-
            // line option aliases
            util::runtime_configuration rtcfg;

            // handle all common command line switches
            std::size_t num_threads = 1;
            std::size_t num_localities = 1;
            std::string queuing;
            boost::program_options::variables_map vm;

            result = util::command_line_handling(desc_cmdline, argc, argv, 
                ini_config, mode, 
                f, vm, rtcfg, num_threads, num_localities, queuing);
            if (result) return result;

            // Initialize and start the HPX runtime.
            if (0 == std::string("global").find(queuing)) {
#if defined(HPX_GLOBAL_SCHEDULER)
                result = detail::run_global(rtcfg, f, vm, mode,
                    startup, shutdown, num_threads, num_localities);
#else
                throw std::logic_error("Command line option --hpx:queuing=global "
                    "is not configured in this build. Please rebuild with "
                    "'cmake -DHPX_GLOBAL_SCHEDULER=ON'.");
#endif
            }
            else if (0 == std::string("local").find(queuing)) {
#if defined(HPX_LOCAL_SCHEDULER)
                result = detail::run_local(rtcfg, f, vm, mode,
                    startup, shutdown, num_threads, num_localities);
#else
                throw std::logic_error("Command line option --hpx:queuing=local "
                    "is not configured in this build. Please rebuild with "
                    "'cmake -DHPX_LOCAL_SCHEDULER=ON'.");
#endif
            }
            else if (0 == std::string("priority_local").find(queuing)) {
                // local scheduler with priority queue (one queue for each OS threads
                // plus one separate queue for high priority PX-threads)
                result = detail::run_priority_local(rtcfg, f, vm, mode,
                    startup, shutdown, num_threads, num_localities);
            }
            else if (0 == std::string("abp").find(queuing)) {
                // abp scheduler: local dequeues for each OS thread, with work
                // stealing from the "bottom" of each.
#if defined(HPX_ABP_SCHEDULER)
                result = detail::run_abp(rtcfg, f, vm, mode,
                    startup, shutdown, num_threads, num_localities);
#else
                throw std::logic_error("Command line option --hpx:queuing=abp "
                    "is not configured in this build. Please rebuild with "
                    "'cmake -DHPX_ABP_SCHEDULER=ON'.");
#endif
            }
            else if (0 == std::string("priority_abp").find(queuing)) {
                // priority abp scheduler: local priority dequeues for each
                // OS thread, with work stealing from the "bottom" of each.
#if defined(HPX_ABP_PRIORITY_SCHEDULER)
                result = detail::run_priority_abp(rtcfg, f, vm, mode,
                    startup, shutdown, num_threads, num_localities);
#else
                throw std::logic_error("Command line option --hpx:queuing=priority_abp "
                    "is not configured in this build. Please rebuild with "
                    "'cmake -DHPX_ABP_PRIORITY_SCHEDULER=ON'.");
#endif
            }
            else if (0 == std::string("hierarchy").find(queuing)) {
#if defined(HPX_HIERARCHY_SCHEDULER)
                // hierarchy scheduler: tree of queues, with work
                // stealing from the parent queue in that tree.
                result = detail::run_hierarchy(rtcfg, f, vm, mode,
                    startup, shutdown, num_threads, num_localities);
#else
                throw std::logic_error("Command line option --hpx:queuing=hierarchy "
                    "is not configured in this build. Please rebuild with "
                    "'cmake -DHPX_HIERARCHY_SCHEDULER=ON'.");
#endif
            }
            else if (0 == std::string("periodic").find(queuing)) {
#if defined(HPX_PERIODIC_PRIORITY_SCHEDULER)
                result = detail::run_periodic(rtcfg, f, vm, mode,
                    startup, shutdown, num_threads, num_localities);
#else
                throw std::logic_error("Command line option --hpx:queuing=periodic "
                    "is not configured in this build. Please rebuild with "
                    "'cmake -DHPX_PERIODIC_PRIORITY_SCHEDULER=ON'.");
#endif
            }
            else {
                throw std::logic_error("Bad value for command line option "
                    "--hpx:queuing");
            }
        }
        catch (std::exception& e) {
            std::cerr << "hpx::init: std::exception caught: " << e.what()
                      << "\n";
            return -1;
        }
        catch (...) {
            std::cerr << "hpx::init: unexpected exception caught\n";
            return -1;
        }
        return result;
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
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
    int finalize(double shutdown_timeout, double localwait)
    {
        if (std::abs(localwait - 1.0) < 1e-16)
            localwait = detail::get_option("hpx.finalize_wait_time", -1.0);
        else
        {
            hpx::util::high_resolution_timer t;
            double start_time = t.elapsed();
            double current = 0.0;
            do {
                current = t.elapsed();
            } while (current - start_time < localwait * 1e-6);
        }

        if (std::abs(shutdown_timeout - 1.0) < 1e-16)
            shutdown_timeout = detail::get_option("hpx.shutdown_timeout", -1.0);

        components::stubs::runtime_support::shutdown_all(
            naming::get_id_from_locality_id(HPX_AGAS_BOOTSTRAP_PREFIX),
            shutdown_timeout);

        return 0;
    }

    ///////////////////////////////////////////////////////////////////////////
    int disconnect(double shutdown_timeout, double localwait)
    {
        if (std::abs(localwait - 1.0) < 1e-16)
            localwait = detail::get_option("hpx.finalize_wait_time", -1.0);
        else
        {
            hpx::util::high_resolution_timer t;
            double start_time = t.elapsed();
            double current = 0.0;
            do {
                current = t.elapsed();
            } while (current - start_time < localwait * 1e-6);
        }

        if (std::abs(shutdown_timeout - 1.0) < 1e-16)
            shutdown_timeout = detail::get_option("hpx.shutdown_timeout", -1.0);

        components::server::runtime_support* p =
            reinterpret_cast<components::server::runtime_support*>(
                  get_runtime().get_runtime_support_lva());

        p->call_shutdown_functions(true);
        p->call_shutdown_functions(false);
        p->stop(shutdown_timeout, naming::invalid_id, true);

        return 0;
    }

    ///////////////////////////////////////////////////////////////////////////
    void terminate()
    {
        components::stubs::runtime_support::terminate_all(
            naming::get_id_from_locality_id(HPX_AGAS_BOOTSTRAP_PREFIX));
    }
}

