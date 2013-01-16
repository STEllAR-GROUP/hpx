//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2010-2011 Phillip LeBlanc, Dylan Stark
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/config.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/hpx_start.hpp>
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
namespace hpx
{
    void set_error_handlers();
}

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

typedef
    hpx::util::detail::bound_action3<
        hpx::actions::plain_action2<
            std::string const&
          , const hpx::naming::gid_type&
          , hpx::detail::list_symbolic_name
          , hpx::actions::detail::this_type
        >
      , hpx::naming::id_type
      , hpx::util::placeholders::arg<0>
      , hpx::util::placeholders::arg<1>
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
    hpx::util::detail::bound_action3<
        hpx::actions::plain_action2<
            std::string const&
          , int
          , hpx::detail::list_component_type
          , hpx::actions::detail::this_type
        >
      , hpx::naming::id_type
      , hpx::util::placeholders::arg<0>
      , hpx::util::placeholders::arg<1>
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
    void list_counter_infos_header(bool skeleton)
    {
        // print header
        print("Information about available counter instances");
        if (skeleton)
            print("(replace '*' below with the appropriate sequence number)");
        print(std::string(78, '-'));
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
                    throw std::logic_error(msg.c_str());
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
                    throw std::logic_error(msg.c_str());
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

        // helper functions testing option compatibility
        void ensure_high_priority_compatibility(
            boost::program_options::variables_map const& vm)
        {
            if (vm.count("hpx:high-priority-threads")) {
                throw std::logic_error("Invalid command line option "
                    "--hpx:high-priority-threads, valid for "
                    "--hpx:queuing=priority_local only");
            }
        }

        void ensure_numa_sensitivity_compatibility(
            boost::program_options::variables_map const& vm)
        {
            if (vm.count("hpx:numa-sensitive")) {
                throw std::logic_error("Invalid command line option "
                    "--hpx:numa-sensitive, valid for "
                    "--hpx:queuing=local, priority_local, or priority_abp only");
            }
        }

        void ensure_hierarchy_arity_compatibility(
            boost::program_options::variables_map const& vm)
        {
            if (vm.count("hpx:hierarchy-arity")) {
                throw std::logic_error("Invalid command line option "
                    "--hpx:hierarchy-arity, valid for --hpx:queuing=hierarchy only.");
            }
        }

        void ensure_queuing_option_compatibility(
            boost::program_options::variables_map const& vm)
        {
            ensure_high_priority_compatibility(vm);
            ensure_numa_sensitivity_compatibility(vm);
            ensure_hierarchy_arity_compatibility(vm);
        }

        void ensure_hwloc_compatibility(
            boost::program_options::variables_map const& vm)
        {
#if defined(HPX_HAVE_HWLOC) || defined(BOOST_WINDOWS)
            // pu control is available for HWLOC and Windows only
            if (vm.count("hpx:pu-offset")) {
                throw std::logic_error("Invalid command line option "
                    "--hpx:pu-offset, valid for --hpx:queuing=priority_local only.");
            }
            if (vm.count("hpx:pu-step")) {
                throw std::logic_error("Invalid command line option "
                    "--hpx:pu-step, valid for --hpx:queuing=priority_local only.");
            }
#endif
#if defined(HPX_HAVE_HWLOC)
            // affinity control is available for HWLOC only
            if (vm.count("hpx:affinity")) {
                throw std::logic_error("Invalid command line option "
                    "--hpx:affinity, valid for --hpx:queuing=priority_local only.");
            }
#endif
        }

        ///////////////////////////////////////////////////////////////////////
        int run(hpx::runtime& rt, hpx_main_type f,
            boost::program_options::variables_map& vm, runtime_mode mode,
            startup_function_type const& startup,
            shutdown_function_type const& shutdown)
        {
            add_startup_functions(rt, vm, mode, startup, shutdown);

            // Run this runtime instance using the given function f.
            if (0 != f)
                return rt.run(boost::bind(f, vm));

            // Run this runtime instance without an hpx_main
            return rt.run();
        }

        int start(hpx::runtime& rt, hpx_main_type f,
            boost::program_options::variables_map& vm, runtime_mode mode,
            startup_function_type const& startup,
            shutdown_function_type const& shutdown)
        {
            add_startup_functions(rt, vm, mode, startup, shutdown);

            if (0 != f) {
                // Run this runtime instance using the given function f.
                return rt.start(boost::bind(f, vm));
            }

            // Run this runtime instance without an hpx_main
            return rt.start();
        }

#if defined(HPX_GLOBAL_SCHEDULER)
        ///////////////////////////////////////////////////////////////////////
        // global scheduler (one queue for all OS threads)
        int run_global(startup_function_type const& startup,
            shutdown_function_type const& shutdown,
            util::command_line_handling& cfg, bool blocking)
        {
            ensure_queuing_option_compatibility(cfg.vm_);
            ensure_hwloc_compatibility(cfg.vm_);

            // scheduling policy
            typedef hpx::threads::policies::global_queue_scheduler
                global_queue_policy;

            global_queue_policy::init_parameter_type init;

            // Build and configure this runtime instance.
            typedef hpx::runtime_impl<global_queue_policy> runtime_type;
            HPX_STD_UNIQUE_PTR<hpx::runtime> rt(
                new runtime_type(cfg.rtcfg_, cfg.mode_, cfg.num_threads_, init));

            if (blocking) {
                return run(*rt, cfg.hpx_main_f_, cfg.vm_, cfg.mode_, startup,
                    shutdown);
            }

            // non-blocking version
            start(*rt, cfg.hpx_main_f_, cfg.vm_, cfg.mode_, startup, shutdown);

            rt.release();          // pointer to runtime is stored in TLS
            return 0;
        }
#endif

#if defined(HPX_LOCAL_SCHEDULER)
        ///////////////////////////////////////////////////////////////////////
        // local scheduler (one queue for each OS threads)
        int run_local(startup_function_type const& startup,
            shutdown_function_type const& shutdown,
            util::command_line_handling& cfg, bool blocking)
        {
            ensure_high_priority_compatibility(cfg.vm_);
            ensure_hierarchy_arity_compatibility(cfg.vm_);

            bool numa_sensitive = false;
            if (cfg.vm_.count("hpx:numa-sensitive"))
                numa_sensitive = true;

            std::size_t pu_offset = 0;
            std::size_t pu_step = 1;
            std::string affinity("pu");
#if defined(HPX_HAVE_HWLOC) || defined(BOOST_WINDOWS)
            if (cfg.vm_.count("hpx:pu-offset")) {
                pu_offset = cfg.vm_["hpx:pu-offset"].as<std::size_t>();
                if (pu_offset >= hpx::threads::hardware_concurrency()) {
                    throw std::logic_error("Invalid command line option "
                        "--hpx:pu-offset, value must be smaller than number of "
                        "available processing units.");
                }
            }

            if (cfg.vm_.count("hpx:pu-step")) {
                pu_step = cfg.vm_["hpx:pu-step"].as<std::size_t>();
                if (pu_step == 0 || pu_step >= hpx::threads::hardware_concurrency()) {
                    throw std::logic_error("Invalid command line option "
                        "--hpx:pu-step, value must be non-zero smaller than number of "
                        "available processing units.");
                }
            }
#endif
#if defined(HPX_HAVE_HWLOC)
            if (cfg.vm_.count("hpx:affinity")) {
                affinity = cfg.vm_["hpx:affinity"].as<std::string>();
                if (0 != std::string("pu").find(affinity) &&
                    0 != std::string("core").find(affinity) &&
                    0 != std::string("numa").find(affinity) &&
                    0 != std::string("machine").find(affinity))
                {
                    throw std::logic_error("Invalid command line option "
                        "--hpx:affinity, value must be one of: pu, core, numa, "
                        "or machine.");
                }
            }
#endif

            // scheduling policy
            typedef hpx::threads::policies::local_queue_scheduler
                local_queue_policy;
            local_queue_policy::init_parameter_type init(
                cfg.num_threads_, 1000, numa_sensitive, pu_offset, pu_step,
                affinity);

            // Build and configure this runtime instance.
            typedef hpx::runtime_impl<local_queue_policy> runtime_type;
            HPX_STD_UNIQUE_PTR<hpx::runtime> rt(
                new runtime_type(cfg.rtcfg_, cfg.mode_, cfg.num_threads_, init));

            if (blocking) {
                return run(*rt, cfg.hpx_main_f_, cfg.vm_, cfg.mode_, startup,
                    shutdown);
            }

            // non-blocking version
            start(*rt, cfg.hpx_main_f_, cfg.vm_, cfg.mode_, startup, shutdown);

            rt.release();          // pointer to runtime is stored in TLS
            return 0;
        }
#endif

        ///////////////////////////////////////////////////////////////////////
        // local scheduler with priority queue (one queue for each OS threads
        // plus one separate queue for high priority PX-threads)
        int run_priority_local(startup_function_type const& startup,
            shutdown_function_type const& shutdown,
            util::command_line_handling& cfg, bool blocking)
        {
            ensure_hierarchy_arity_compatibility(cfg.vm_);

            std::size_t num_high_priority_queues = cfg.num_threads_;
            if (cfg.vm_.count("hpx:high-priority-threads")) {
                num_high_priority_queues =
                    cfg.vm_["hpx:high-priority-threads"].as<std::size_t>();
            }

            bool numa_sensitive = false;
            if (cfg.vm_.count("hpx:numa-sensitive"))
                numa_sensitive = true;

            std::size_t pu_offset = 0;
            std::size_t pu_step = 1;
            std::string affinity("pu");
#if defined(HPX_HAVE_HWLOC) || defined(BOOST_WINDOWS)
            if (cfg.vm_.count("hpx:pu-offset")) {
                pu_offset = cfg.vm_["hpx:pu-offset"].as<std::size_t>();
                if (pu_offset >= hpx::threads::hardware_concurrency()) {
                    throw std::logic_error("Invalid command line option "
                        "--hpx:pu-offset, value must be smaller than number of "
                        "available processing units.");
                }
            }

            if (cfg.vm_.count("hpx:pu-step")) {
                pu_step = cfg.vm_["hpx:pu-step"].as<std::size_t>();
                if (pu_step == 0 || pu_step >= hpx::threads::hardware_concurrency()) {
                    throw std::logic_error("Invalid command line option "
                        "--hpx:pu-step, value must be non-zero smaller than number of "
                        "available processing units.");
                }
            }
#endif
#if defined(HPX_HAVE_HWLOC)
            if (cfg.vm_.count("hpx:affinity")) {
                affinity = cfg.vm_["hpx:affinity"].as<std::string>();
                if (0 != std::string("pu").find(affinity) &&
                    0 != std::string("core").find(affinity) &&
                    0 != std::string("numa").find(affinity) &&
                    0 != std::string("machine").find(affinity))
                {
                    throw std::logic_error("Invalid command line option "
                        "--hpx:affinity, value must be one of: pu, core, numa, "
                        "or machine.");
                }
            }
#endif
            // scheduling policy
            typedef hpx::threads::policies::local_priority_queue_scheduler
                local_queue_policy;
            local_queue_policy::init_parameter_type init(
                cfg.num_threads_, num_high_priority_queues, 1000,
                numa_sensitive, pu_offset, pu_step, affinity);

            // Build and configure this runtime instance.
            typedef hpx::runtime_impl<local_queue_policy> runtime_type;
            HPX_STD_UNIQUE_PTR<hpx::runtime> rt(
                new runtime_type(cfg.rtcfg_, cfg.mode_, cfg.num_threads_, init));

            if (blocking) {
                return run(*rt, cfg.hpx_main_f_, cfg.vm_, cfg.mode_, startup,
                    shutdown);
            }

            // non-blocking version
            start(*rt, cfg.hpx_main_f_, cfg.vm_, cfg.mode_, startup, shutdown);

            rt.release();          // pointer to runtime is stored in TLS
            return 0;
        }

#if defined(HPX_ABP_SCHEDULER)
        ///////////////////////////////////////////////////////////////////////
        // abp scheduler: local deques for each OS thread, with work
        // stealing from the "bottom" of each.
        int run_abp(startup_function_type const& startup,
            shutdown_function_type const& shutdown,
            util::command_line_handling& cfg, bool blocking)
        {
            ensure_queuing_option_compatibility(cfg.vm_);
            ensure_hwloc_compatibility(cfg.vm_);

            // scheduling policy
            typedef hpx::threads::policies::abp_queue_scheduler
                abp_queue_policy;
            abp_queue_policy::init_parameter_type init(cfg.num_threads_, 1000);

            // Build and configure this runtime instance.
            typedef hpx::runtime_impl<abp_queue_policy> runtime_type;
            HPX_STD_UNIQUE_PTR<hpx::runtime> rt(
                new runtime_type(cfg.rtcfg_, cfg.mode_, cfg.num_threads_, init));

            if (blocking) {
                return run(*rt, cfg.hpx_main_f_, cfg.vm_, cfg.mode_, startup,
                    shutdown);
            }

            // non-blocking version
            start(*rt, cfg.hpx_main_f_, cfg.vm_, cfg.mode_, startup, shutdown);

            rt.release();          // pointer to runtime is stored in TLS
            return 0;
        }
#endif

#if defined(HPX_ABP_PRIORITY_SCHEDULER)
        ///////////////////////////////////////////////////////////////////////
        // priority abp scheduler: local priority deques for each OS thread,
        // with work stealing from the "bottom" of each.
        int run_priority_abp(startup_function_type const& startup,
            shutdown_function_type const& shutdown,
            util::command_line_handling& cfg, bool blocking)
        {
            ensure_hierarchy_arity_compatibility(cfg.vm_);
            ensure_hwloc_compatibility(cfg.vm_);

            std::size_t num_high_priority_queues = cfg.num_threads_;
            if (cfg.vm_.count("hpx:high-priority-threads")) {
                num_high_priority_queues =
                    cfg.vm_["hpx:high-priority-threads"].as<std::size_t>();
            }

            bool numa_sensitive = false;
            if (cfg.vm_.count("hpx:numa-sensitive"))
                numa_sensitive = true;

            // scheduling policy
            typedef hpx::threads::policies::abp_priority_queue_scheduler
                abp_priority_queue_policy;
            abp_priority_queue_policy::init_parameter_type init(
                cfg.num_threads_, num_high_priority_queues, 1000,
                numa_sensitive);

            // Build and configure this runtime instance.
            typedef hpx::runtime_impl<abp_priority_queue_policy> runtime_type;
            HPX_STD_UNIQUE_PTR<hpx::runtime> rt(
                new runtime_type(cfg.rtcfg_, cfg.mode_, cfg.num_threads_, init));

            if (blocking) {
                return run(*rt, cfg.hpx_main_f_, cfg.vm_, cfg.mode_, startup,
                    shutdown);
            }

            // non-blocking version
            start(*rt, cfg.hpx_main_f_, cfg.vm_, cfg.mode_, startup, shutdown);

            rt.release();          // pointer to runtime is stored in TLS
            return 0;
        }
#endif

#if defined(HPX_HIERARCHY_SCHEDULER)
        ///////////////////////////////////////////////////////////////////////
        // hierarchical scheduler: The thread queues are built up hierarchically
        // this avoids contention during work stealing
        int run_hierarchy(startup_function_type const& startup,
            shutdown_function_type const& shutdown,
            util::command_line_handling& cfg, bool blocking)
        {
            ensure_high_priority_compatibility(cfg.vm_);
            ensure_numa_sensitivity_compatibility(cfg.vm_);
            ensure_hwloc_compatibility(cfg.vm_);

            // scheduling policy
            typedef hpx::threads::policies::hierarchy_scheduler queue_policy;
            std::size_t arity = 2;
            if (cfg.vm_.count("hpx:hierarchy-arity"))
                arity = cfg.vm_["hpx:hierarchy-arity"].as<std::size_t>();

            queue_policy::init_parameter_type init(cfg.num_threads_, arity, 1000);

            // Build and configure this runtime instance.
            typedef hpx::runtime_impl<queue_policy> runtime_type;
            HPX_STD_UNIQUE_PTR<hpx::runtime> rt(
                new runtime_type(cfg.rtcfg_, cfg.mode_, cfg.num_threads_, init));

            if (blocking) {
                return run(*rt, cfg.hpx_main_f_, cfg.vm_, cfg.mode_, startup,
                    shutdown);
            }

            // non-blocking version
            start(*rt, cfg.hpx_main_f_, cfg.vm_, cfg.mode_, startup, shutdown);

            rt.release();          // pointer to runtime is stored in TLS
            return 0;
        }
#endif

#if defined(HPX_PERIODIC_PRIORITY_SCHEDULER)
        ///////////////////////////////////////////////////////////////////////
        // hierarchical scheduler: The thread queues are built up hierarchically
        // this avoids contention during work stealing
        int run_periodic(startup_function_type const& startup,
            shutdown_function_type const& shutdown,
            util::command_line_handling& cfg, bool blocking)
        {
            ensure_hierarchy_arity_compatibility(cfg.vm_);
            ensure_hwloc_compatibility(cfg.vm_);

            std::size_t num_high_priority_queues = cfg.num_threads_;
            if (cfg.vm_.count("hpx:high-priority-threads")) {
                num_high_priority_queues =
                    cfg.vm_["hpx:high-priority-threads"].as<std::size_t>();
            }

            bool numa_sensitive = false;
            if (cfg.vm_.count("hpx:numa-sensitive"))
                numa_sensitive = true;

            // scheduling policy
            typedef hpx::threads::policies::local_periodic_priority_scheduler
                local_queue_policy;
            local_queue_policy::init_parameter_type init(cfg.num_threads_,
                num_high_priority_queues, 1000, numa_sensitive);

            // Build and configure this runtime instance.
            typedef hpx::runtime_impl<local_queue_policy> runtime_type;
            HPX_STD_UNIQUE_PTR<hpx::runtime> rt(
                new runtime_type(cfg.rtcfg_, cfg.mode_, cfg.num_threads_, init));

            if (blocking) {
                return run(*rt, cfg.hpx_main_f_, cfg.vm_, cfg.mode_, startup,
                    shutdown);
            }

            // non-blocking version
            start(*rt, cfg.hpx_main_f_, cfg.vm_, cfg.mode_, startup, shutdown);

            rt.release();          // pointer to runtime is stored in TLS
            return 0;
        }
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    int run_or_start(hpx_main_type f,
        boost::program_options::options_description& desc_cmdline,
        int argc, char* argv[], std::vector<std::string> const& ini_config,
        startup_function_type const& startup,
        shutdown_function_type const& shutdown, hpx::runtime_mode mode,
        bool blocking)
    {
        int result = 0;
        set_error_handlers();

        try {
            // handle all common command line switches
            util::command_line_handling cfg(mode, f, ini_config);

            result = cfg.call(desc_cmdline, argc, argv);
            if (result != 0) {
                if (result > 0)
                    result = 0;     // --hpx:help
                return result;
            }

            // Initialize and start the HPX runtime.
            if (0 == std::string("global").find(cfg.queuing_)) {
#if defined(HPX_GLOBAL_SCHEDULER)
                result = detail::run_global(startup, shutdown, cfg, blocking);
#else
                throw std::logic_error("Command line option --hpx:queuing=global "
                    "is not configured in this build. Please rebuild with "
                    "'cmake -DHPX_GLOBAL_SCHEDULER=ON'.");
#endif
            }
            else if (0 == std::string("local").find(cfg.queuing_)) {
#if defined(HPX_LOCAL_SCHEDULER)
                result = detail::run_local(startup, shutdown, cfg, blocking);
#else
                throw std::logic_error("Command line option --hpx:queuing=local "
                    "is not configured in this build. Please rebuild with "
                    "'cmake -DHPX_LOCAL_SCHEDULER=ON'.");
#endif
            }
            else if (0 == std::string("priority_local").find(cfg.queuing_)) {
                // local scheduler with priority queue (one queue for each OS threads
                // plus one separate queue for high priority PX-threads)
                result = detail::run_priority_local(startup, shutdown, cfg, blocking);
            }
            else if (0 == std::string("abp").find(cfg.queuing_)) {
                // abp scheduler: local dequeues for each OS thread, with work
                // stealing from the "bottom" of each.
#if defined(HPX_ABP_SCHEDULER)
                result = detail::run_abp(startup, shutdown, cfg, blocking);
#else
                throw std::logic_error("Command line option --hpx:queuing=abp "
                    "is not configured in this build. Please rebuild with "
                    "'cmake -DHPX_ABP_SCHEDULER=ON'.");
#endif
            }
            else if (0 == std::string("priority_abp").find(cfg.queuing_)) {
                // priority abp scheduler: local priority dequeues for each
                // OS thread, with work stealing from the "bottom" of each.
#if defined(HPX_ABP_PRIORITY_SCHEDULER)
                result = detail::run_priority_abp(startup, shutdown, cfg, blocking);
#else
                throw std::logic_error("Command line option --hpx:queuing=priority_abp "
                    "is not configured in this build. Please rebuild with "
                    "'cmake -DHPX_ABP_PRIORITY_SCHEDULER=ON'.");
#endif
            }
            else if (0 == std::string("hierarchy").find(cfg.queuing_)) {
#if defined(HPX_HIERARCHY_SCHEDULER)
                // hierarchy scheduler: tree of queues, with work
                // stealing from the parent queue in that tree.
                result = detail::run_hierarchy(startup, shutdown, cfg, blocking);
#else
                throw std::logic_error("Command line option --hpx:queuing=hierarchy "
                    "is not configured in this build. Please rebuild with "
                    "'cmake -DHPX_HIERARCHY_SCHEDULER=ON'.");
#endif
            }
            else if (0 == std::string("periodic").find(cfg.queuing_)) {
#if defined(HPX_PERIODIC_PRIORITY_SCHEDULER)
                result = detail::run_periodic(startup, shutdown, cfg, blocking);
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
    int init(hpx_main_type f,
        boost::program_options::options_description& desc_cmdline,
        int argc, char* argv[], std::vector<std::string> const& ini_config,
        startup_function_type const& startup,
        shutdown_function_type const& shutdown, hpx::runtime_mode mode)
    {
        return run_or_start(f, desc_cmdline, argc, argv, ini_config,
            startup, shutdown, mode, true);
    }

    int start(hpx_main_type f,
        boost::program_options::options_description& desc_cmdline,
        int argc, char* argv[], std::vector<std::string> const& ini_config,
        startup_function_type const& startup,
        shutdown_function_type const& shutdown, hpx::runtime_mode mode)
    {
        return run_or_start(f, desc_cmdline, argc, argv, ini_config,
            startup, shutdown, mode, false);
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
    int finalize(double shutdown_timeout, double localwait, error_code& ec)
    {
        if (!is_running()) {
            HPX_THROWS_IF(ec, invalid_status, "hpx::finalize",
                "the runtime system is not active (did you already "
                "call finalize?)");
            return -1;
        }

        if (&ec != &throws)
            ec = make_success_code();

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
    int disconnect(double shutdown_timeout, double localwait, error_code& ec)
    {
        if (!is_running()) {
            HPX_THROWS_IF(ec, invalid_status, "hpx::finalize",
                "the runtime system is not active (did you already "
                "call finalize?)");
            return -1;
        }

        if (&ec != &throws)
            ec = make_success_code();

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

    ///////////////////////////////////////////////////////////////////////////
    int stop()
    {
        HPX_STD_UNIQUE_PTR<runtime> rt(get_runtime_ptr());    // take ownership!

        int result = rt->wait();
        rt->stop();

        return result;
    }
}

