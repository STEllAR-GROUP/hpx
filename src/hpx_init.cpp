//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2010-2011 Phillip LeBlanc, Dylan Stark
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/config.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/util/asio_util.hpp>
#include <hpx/util/pbs_environment.hpp>
#include <hpx/util/map_hostnames.hpp>
#include <hpx/util/sed_transform.hpp>
#include <hpx/util/parse_command_line.hpp>

#include <hpx/lcos/eager_future.hpp>
#include <hpx/runtime/components/runtime_support.hpp>
#include <hpx/runtime/actions/function.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/runtime/threads/policies/schedulers.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/util/query_counters.hpp>
#include <hpx/util/stringstream.hpp>

#if !defined(BOOST_WINDOWS)
#  include <signal.h>
#endif

#include <iostream>
#include <vector>

#include <boost/shared_ptr.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>
#include <boost/assign/std/vector.hpp>
#include <boost/foreach.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    typedef int (*hpx_main_func)(boost::program_options::variables_map& vm);
}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace detail
{
    // forward declarations only
    void console_print(std::string const& name);
    void list_symbolic_name(std::string const& name, naming::gid_type const& gid);
    void list_component_type(std::string const& name, components::component_type);
}}

typedef hpx::actions::plain_action1<
    std::string const&, hpx::detail::console_print
> console_print_action;
HPX_REGISTER_PLAIN_ACTION_EX2(console_print_action, console_print_action, true);

typedef hpx::actions::plain_action2<
    std::string const&, hpx::naming::gid_type const&,
    hpx::detail::list_symbolic_name
> list_symbolic_name_action;
HPX_REGISTER_PLAIN_ACTION_EX2(list_symbolic_name_action, list_symbolic_name_action, true);

typedef hpx::actions::plain_action2<
    std::string const&, hpx::components::component_type,
    hpx::detail::list_component_type
> list_component_type_action;
HPX_REGISTER_PLAIN_ACTION_EX2(list_component_type_action, list_component_type_action, true);

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
        naming::gid_type console;
        naming::get_agas_client().get_console_prefix(console, ec);
        if (ec) return;

        typedef lcos::eager_future<console_print_action> future_type;
        future_type(console, name).get(ec);
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
        print(std::string(72, '-'));

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
        naming::gid_type console;
        naming::get_agas_client().get_console_prefix(console);

        print(std::string("registered symbolic names"));

        typedef void iter_func(std::string const&, naming::gid_type const&);
        hpx::actions::function<iter_func> cb(new list_symbolic_name_action);
        naming::get_agas_client().iterateids(cb);
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
        naming::gid_type console;
        naming::get_agas_client().get_console_prefix(console);

        print(std::string("registered component types"));

        typedef void iter_func(std::string const&, components::component_type);
        hpx::actions::function<iter_func> cb(new list_component_type_action);
        naming::get_agas_client().iterate_types(cb);
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
    BOOL termination_handler(DWORD ctrl_type);
#else
    void termination_handler(int signum);
#endif

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        int print_version(std::ostream& out)
        {
            out << std::endl << hpx::copyright() << std::endl;
            out << hpx::complete_version() << std::endl;
            return 0;
        }


        ///////////////////////////////////////////////////////////////////////
        template <typename Runtime>
        struct dump_config
        {
            dump_config(Runtime const& rt) : rt_(boost::cref(rt)) {}

            void operator()() const
            {
                std::cout << "Configuration after runtime start:\n";
                std::cout << "----------------------------------\n";
                rt_.get().get_config().dump(0, std::cout);
                std::cout << "----------------------------------\n";
            }

            boost::reference_wrapper<Runtime const> rt_;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Runtime>
        void handle_list_and_print_options(Runtime& rt,
            boost::program_options::variables_map& vm)
        {
            if (vm.count("list-counters")) {
                // Print the names of all registered performance counters.
                rt.add_startup_function(&list_counter_names);
            }
            if (vm.count("list-counter-infos")) {
                // Print info about all registered performance counters.
                rt.add_startup_function(&list_counter_infos);
            }
            if (vm.count("list-symbolic-names")) {
                // Print all registered symbolic names.
                rt.add_startup_function(&list_symbolic_names);
            }
            if (vm.count("list-component-types")) {
                // Print all registered component types.
                rt.add_startup_function(&list_component_types);
            }

            if (vm.count("print-counter")) {
                std::size_t interval = 0;
                if (vm.count("print-counter-interval"))
                    interval = vm["print-counter-interval"].as<std::size_t>();

                std::vector<std::string> counters =
                    vm["print-counter"].as<std::vector<std::string> >();

                std::string destination("cout");
                if (vm.count("print-counter-destination"))
                    destination = vm["print-counter-destination"].as<std::string>();

                // schedule the query function at startup, which will schedule
                // itself to run after the given interval
                boost::shared_ptr<util::query_counters> qc =
                    boost::make_shared<util::query_counters>(
                        boost::ref(counters), interval, destination);

                // schedule to run at shutdown
                rt.add_pre_shutdown_function(
                    boost::bind(&util::query_counters::evaluate, qc));

                if (0 != interval)
                    rt.add_startup_function(boost::bind(&print_counters, qc));
            }
            else if (vm.count("print-counter-interval")) {
                throw std::logic_error("Invalid command line option "
                    "--print-counter-interval, valid in conjunction with "
                    "--print-counter only");
            }
            else if (vm.count("print-counter-destination")) {
                throw std::logic_error("Invalid command line option "
                    "--print-counter-destination, valid in conjunction with "
                    "--print-counter only");
            }
        }

        template <typename Runtime>
        int run(Runtime& rt, hpx_main_func f,
            boost::program_options::variables_map& vm, runtime_mode mode,
            startup_function_type const& startup,
            shutdown_function_type const& shutdown, std::size_t num_threads,
            std::size_t num_localities)
        {
            if (vm.count("app-config"))
            {
                std::string config(vm["app-config"].as<std::string>());
                rt.get_config().load_application_configuration(config.c_str());
            }

            if (!!startup)
                rt.add_startup_function(startup);

            if (!!shutdown)
                rt.add_shutdown_function(shutdown);

            // add startup function related to listing counter names or counter
            // infos
            handle_list_and_print_options(rt, vm);

            // Dump the configuration before all components have been loaded.
            if (vm.count("dump-config-initial")) {
                std::cout << "Configuration after runtime construction:\n";
                std::cout << "-----------------------------------------\n";
                rt.get_config().dump(0, std::cout);
                std::cout << "-----------------------------------------\n";
            }

            // Dump the configuration after all components have been loaded.
            if (vm.count("dump-config"))
                rt.add_startup_function(dump_config<Runtime>(rt));

            // Run this runtime instance using the given function f.
            if (0 != f)
                return rt.run(boost::bind(f, vm), num_threads, num_localities);

            // Run this runtime instance without an hpx_main
            return rt.run(num_threads, num_localities);
        }

#if defined(HPX_GLOBAL_SCHEDULER)
        ///////////////////////////////////////////////////////////////////////
        // global scheduler (one queue for all OS threads)
        int run_global(hpx_main_func f, boost::program_options::variables_map& vm,
            runtime_mode mode, std::vector<std::string> const& ini_config,
            startup_function_type const& startup,
            shutdown_function_type const& shutdown, std::size_t num_threads,
            std::size_t num_localities)
        {
            if (vm.count("high-priority-threads")) {
                throw std::logic_error("Invalid command line option "
                    "--high-priority-threads, valid for "
                    "--queueing=priority_local only");
            }
            if (vm.count("numa-sensitive")) {
                throw std::logic_error("Invalid command line option "
                    "--numa-sensitive, valid for "
                    "--queueing=priority_local  or priority_abp only");
            }
            if (vm.count("hierarchy-arity")) {
                throw std::logic_error("Invalid command line option "
                    "--hierarchy-arity, valid for --queueing=hierarchy only.");
            }
#if defined(HPX_HAVE_HWLOC)
            if (vm.count("pu-offset")) {
                throw std::logic_error("Invalid command line option "
                    "--pu-offset, valid for --queueing=priority_local only.");
            }
            if (vm.count("pu-step")) {
                throw std::logic_error("Invalid command line option "
                    "--pu-step, valid for --queueing=priority_local only.");
            }
#endif
            // scheduling policy
            typedef hpx::threads::policies::global_queue_scheduler
                global_queue_policy;

            // Build and configure this runtime instance.
            typedef hpx::runtime_impl<global_queue_policy> runtime_type;
            runtime_type rt(mode, global_queue_policy::init_parameter_type(),
                vm["hpx-config"].as<std::string>(), ini_config);

            return run(rt, f, vm, mode, startup, shutdown, num_threads,
                num_localities);
        }
#endif

#if defined(HPX_LOCAL_SCHEDULER)
        ///////////////////////////////////////////////////////////////////////
        // local scheduler (one queue for each OS threads)
        int run_local(hpx_main_func f, boost::program_options::variables_map& vm,
            runtime_mode mode, std::vector<std::string> const& ini_config,
            startup_function_type const& startup,
            shutdown_function_type const& shutdown, std::size_t num_threads,
            std::size_t num_localities)
        {
            if (vm.count("high-priority-threads")) {
                throw std::logic_error("Invalid command line option "
                    "--high-priority-threads, valid for "
                    "--queueing=priority_local only.");
            }
            if (vm.count("numa-sensitive")) {
                throw std::logic_error("Invalid command line option "
                    "--numa-sensitive, valid for "
                    "--queueing=priority_local or priority_abp only");
            }
            if (vm.count("hierarchy-arity")) {
                throw std::logic_error("Invalid command line option "
                    "--hierarchy-arity, valid for --queueing=hierarchy only.");
            }
#if defined(HPX_HAVE_HWLOC)
            if (vm.count("pu-offset")) {
                throw std::logic_error("Invalid command line option "
                    "--pu-offset, valid for --queueing=priority_local only.");
            }
            if (vm.count("pu-step")) {
                throw std::logic_error("Invalid command line option "
                    "--pu-step, valid for --queueing=priority_local only.");
            }
#endif
            // scheduling policy
            typedef hpx::threads::policies::local_queue_scheduler
                local_queue_policy;
            local_queue_policy::init_parameter_type init(num_threads, 1000);

            // Build and configure this runtime instance.
            typedef hpx::runtime_impl<local_queue_policy> runtime_type;
            runtime_type rt(mode, init, vm["hpx-config"].as<std::string>(),
                ini_config);

            return run(rt, f, vm, mode, startup, shutdown, num_threads,
                num_localities);
        }
#endif

        ///////////////////////////////////////////////////////////////////////
        // local scheduler with priority queue (one queue for each OS threads
        // plus one separate queue for high priority PX-threads)
        int run_priority_local(hpx_main_func f, boost::program_options::variables_map& vm,
            runtime_mode mode, std::vector<std::string> const& ini_config,
            startup_function_type const& startup,
            shutdown_function_type const& shutdown, std::size_t num_threads,
            std::size_t num_localities)
        {
            std::size_t num_high_priority_queues = num_threads;
            if (vm.count("high-priority-threads")) {
                num_high_priority_queues =
                    vm["high-priority-threads"].as<std::size_t>();
            }
            bool numa_sensitive = false;
            if (vm.count("numa-sensitive"))
                numa_sensitive = true;

            if (vm.count("hierarchy-arity")) {
                throw std::logic_error("Invalid command line option "
                    "--hierarchy-arity, valid for --queueing=hierarchy only.");
            }

            std::size_t pu_offset = 0;
            std::size_t pu_step = 1;
#if defined(HPX_HAVE_HWLOC)
            if (vm.count("pu-offset")) {
                pu_offset = vm["pu-offset"].as<std::size_t>();
                if (pu_offset >= hpx::threads::hardware_concurrency()) {
                    throw std::logic_error("Invalid command line option "
                        "--pu-offset, value must be smaller than number of "
                        "available processing units.");
                }
            }

            if (vm.count("pu-step")) {
                pu_step = vm["pu-step"].as<std::size_t>();
                if (pu_step == 0 || pu_step >= hpx::threads::hardware_concurrency()) {
                    throw std::logic_error("Invalid command line option "
                        "--pu-step, value must be non-zero smaller than number of "
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
            runtime_type rt(mode, init, vm["hpx-config"].as<std::string>(),
                ini_config);

            return run(rt, f, vm, mode, startup, shutdown, num_threads,
                num_localities);
        }

#if defined(HPX_ABP_SCHEDULER)
        ///////////////////////////////////////////////////////////////////////
        // abp scheduler: local deques for each OS thread, with work
        // stealing from the "bottom" of each.
        int run_abp(hpx_main_func f, boost::program_options::variables_map& vm,
            runtime_mode mode, std::vector<std::string> const& ini_config,
            startup_function_type const& startup,
            shutdown_function_type const& shutdown, std::size_t num_threads,
            std::size_t num_localities)
        {
            if (vm.count("high-priority-threads")) {
                throw std::logic_error("Invalid command line option "
                    "--high-priority-threads, valid for "
                    "--queueing=priority_local only.");
            }
            if (vm.count("numa-sensitive")) {
                throw std::logic_error("Invalid command line option "
                    "--numa-sensitive, valid for "
                    "--queueing=priority_local or priority_abp only");
            }
            if (vm.count("hierarchy-arity")) {
                throw std::logic_error("Invalid command line option "
                    "--hierarchy-arity, valid for --queueing=hierarchy only.");
            }
#if defined(HPX_HAVE_HWLOC)
            if (vm.count("pu-offset")) {
                throw std::logic_error("Invalid command line option "
                    "--pu-offset, valid for --queueing=priority_local only.");
            }
            if (vm.count("pu-step")) {
                throw std::logic_error("Invalid command line option "
                    "--pu-step, valid for --queueing=priority_local only.");
            }
#endif
            // scheduling policy
            typedef hpx::threads::policies::abp_queue_scheduler
                abp_queue_policy;
            abp_queue_policy::init_parameter_type init(num_threads, 1000);

            // Build and configure this runtime instance.
            typedef hpx::runtime_impl<abp_queue_policy> runtime_type;
            runtime_type rt(mode, init, vm["hpx-config"].as<std::string>(),
                ini_config);

            return run(rt, f, vm, mode, startup, shutdown, num_threads,
                num_localities);
        }
#endif

#if defined(HPX_ABP_PRIORITY_SCHEDULER)
        ///////////////////////////////////////////////////////////////////////
        // priority abp scheduler: local priority deques for each OS thread,
        // with work stealing from the "bottom" of each.
        int run_priority_abp(
            hpx_main_func f, boost::program_options::variables_map& vm,
            runtime_mode mode, std::vector<std::string> const& ini_config,
            startup_function_type const& startup,
            shutdown_function_type const& shutdown, std::size_t num_threads,
            std::size_t num_localities)
        {
            std::size_t num_high_priority_queues = num_threads;
            if (vm.count("high-priority-threads")) {
                num_high_priority_queues =
                    vm["high-priority-threads"].as<std::size_t>();
            }
            bool numa_sensitive = false;
            if (vm.count("numa-sensitive"))
                numa_sensitive = true;
            if (vm.count("hierarchy-arity")) {
                throw std::logic_error("Invalid command line option "
                    "--hierarchy-arity, valid for --queueing=hierarchy only.");
            }
#if defined(HPX_HAVE_HWLOC)
            if (vm.count("pu-offset")) {
                throw std::logic_error("Invalid command line option "
                    "--pu-offset, valid for --queueing=priority_local only.");
            }
            if (vm.count("pu-step")) {
                throw std::logic_error("Invalid command line option "
                    "--pu-step, valid for --queueing=priority_local only.");
            }
#endif
            // scheduling policy
            typedef hpx::threads::policies::abp_priority_queue_scheduler
                abp_priority_queue_policy;
            abp_priority_queue_policy::init_parameter_type init(
                num_threads, num_high_priority_queues, 1000, numa_sensitive);

            // Build and configure this runtime instance.
            typedef hpx::runtime_impl<abp_priority_queue_policy> runtime_type;
            runtime_type rt(mode, init, vm["hpx-config"].as<std::string>(),
                ini_config);

            return run(rt, f, vm, mode, startup, shutdown, num_threads,
                num_localities);
        }
#endif

#if defined(HPX_HIERARCHY_SCHEDULER)
        ///////////////////////////////////////////////////////////////////////
        // hierarchical scheduler: The thread queues are built up hierarchically
        // this avoids contention during work stealing
        int run_hierarchy(hpx_main_func f, boost::program_options::variables_map& vm,
            runtime_mode mode, std::vector<std::string> const& ini_config,
            startup_function_type const& startup,
            shutdown_function_type const& shutdown, std::size_t num_threads,
            std::size_t num_localities)
        {
            if (vm.count("high-priority-threads")) {
                throw std::logic_error("Invalid command line option "
                    "--high-priority-threads, valid for "
                    "--queueing=priority_local only.");
            }
            if (vm.count("numa-sensitive")) {
                throw std::logic_error("Invalid command line option "
                    "--numa-sensitive, valid for "
                    "--queueing=priority_local or priority_abp only");
            }
#if defined(HPX_HAVE_HWLOC)
            if (vm.count("pu-offset")) {
                throw std::logic_error("Invalid command line option "
                    "--pu-offset, valid for --queueing=priority_local only.");
            }
            if (vm.count("pu-step")) {
                throw std::logic_error("Invalid command line option "
                    "--pu-step, valid for --queueing=priority_local only.");
            }
#endif
            // scheduling policy
            typedef hpx::threads::policies::hierarchy_scheduler queue_policy;
            std::size_t arity = 2;
            if (vm.count("hierarchy-arity"))
                arity = vm["hierarchy-arity"].as<std::size_t>();
            queue_policy::init_parameter_type init(num_threads, arity, 1000);

            // Build and configure this runtime instance.
            typedef hpx::runtime_impl<queue_policy> runtime_type;
            runtime_type rt(mode, init, vm["hpx-config"].as<std::string>(),
                ini_config);

            return run(rt, f, vm, mode, startup, shutdown, num_threads,
                num_localities);
        }
#endif

#if defined(HPX_PERIODIC_PRIORITY_SCHEDULER)
        ///////////////////////////////////////////////////////////////////////
        // hierarchical scheduler: The thread queues are built up hierarchically
        // this avoids contention during work stealing
        int run_periodic(hpx_main_func f, boost::program_options::variables_map& vm,
            runtime_mode mode, std::vector<std::string> const& ini_config,
            startup_function_type const& startup,
            shutdown_function_type const& shutdown, std::size_t num_threads,
            std::size_t num_localities)
        {
            std::size_t num_high_priority_queues = num_threads;
            if (vm.count("high-priority-threads")) {
                num_high_priority_queues =
                    vm["high-priority-threads"].as<std::size_t>();
            }
            bool numa_sensitive = false;
            if (vm.count("numa-sensitive"))
                numa_sensitive = true;
            if (vm.count("hierarchy-arity")) {
                throw std::logic_error("Invalid command line option "
                    "--hierarchy-arity, valid for --queueing=hierarchy only.");
            }
#if defined(HPX_HAVE_HWLOC)
            if (vm.count("pu-offset")) {
                throw std::logic_error("Invalid command line option "
                    "--pu-offset, valid for --queueing=priority_local only.");
            }
            if (vm.count("pu-step")) {
                throw std::logic_error("Invalid command line option "
                    "--pu-step, valid for --queueing=priority_local only.");
            }
#endif
            // scheduling policy
            typedef hpx::threads::policies::local_periodic_priority_scheduler
                local_queue_policy;
            local_queue_policy::init_parameter_type init(
                num_threads, num_high_priority_queues, 1000, numa_sensitive);

            // Build and configure this runtime instance.
            typedef hpx::runtime_impl<local_queue_policy> runtime_type;
            runtime_type rt(mode, init, vm["hpx-config"].as<std::string>(),
                ini_config);

            return run(rt, f, vm, mode, startup, shutdown, num_threads,
                num_localities);
        }
#endif

        ///////////////////////////////////////////////////////////////////////
        void set_signal_handlers()
        {
#if defined(BOOST_WINDOWS)
            // Set console control handler to allow server to be stopped.
            SetConsoleCtrlHandler(hpx::termination_handler, TRUE);
#else
            struct sigaction new_action;
            new_action.sa_handler = hpx::termination_handler;
            sigemptyset(&new_action.sa_mask);
            new_action.sa_flags = 0;

            sigaction(SIGBUS, &new_action, NULL);  // Bus error
            sigaction(SIGFPE, &new_action, NULL);  // Floating point exception
            sigaction(SIGILL, &new_action, NULL);  // Illegal instruction
            sigaction(SIGPIPE, &new_action, NULL); // Bad pipe
            sigaction(SIGSEGV, &new_action, NULL); // Segmentation fault
            sigaction(SIGSYS, &new_action, NULL);  // Bad syscall
#endif
        }

        ///////////////////////////////////////////////////////////////////////
        inline void encode (std::string &str, char s, char const *r)
        {
            std::string::size_type pos = 0;
            while ((pos = str.find_first_of(s, pos)) != std::string::npos)
            {
                str.replace (pos, 1, r);
                ++pos;
            }
        }

        inline std::string encode_string(std::string str)
        {
            encode(str, '\n', "\\n");
            return str;
        }

        ///////////////////////////////////////////////////////////////////////
        inline std::string enquote(std::string const& arg)
        {
            if (arg.find_first_of(" \t") != std::string::npos)
                return std::string("\"") + arg + "\"";
            return arg;
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    int init(hpx_main_func f,
        boost::program_options::options_description& desc_cmdline,
        int argc, char* argv[], std::vector<std::string> ini_config,
        startup_function_type startup, shutdown_function_type shutdown,
        hpx::runtime_mode mode)
    {
        int result = 0;
        detail::set_signal_handlers();

        try {
            using boost::program_options::variables_map;
            using namespace boost::assign;

            // Analyze the command line.
            variables_map vm;
            boost::program_options::options_description help;
            std::vector<std::string> unregistered_options;
            bool cmd_result = util::parse_commandline(desc_cmdline, argc, argv,
                vm, util::allow_unregistered, mode, &help, &unregistered_options);
            if (!cmd_result)
                return -1;

            //  store unregistered command line arguments
            if (!unregistered_options.empty()) {
                typedef std::vector<std::string>::const_iterator iterator_type;
                std::string options;
                iterator_type  end = unregistered_options.end();
                for (iterator_type  it = unregistered_options.begin(); it != end; ++it)
                    options += " " + detail::enquote(*it);
                ini_config += "hpx.unknown_cmd_line=" +
                    detail::enquote(argv[0]) + options;
            }

            // print version/copyright information
            if (vm.count("version"))
                return detail::print_version(std::cout);

            if (vm.count("help")) {
                std::string help_option(vm["help"].as<std::string>());
                if (0 == std::string("minimal").find(help_option))
                {
                    std::cout << help << std::endl;
                    return 0;
                }
                else {
                    hpx::util::osstream strm;
                    strm << help << std::endl;
                    ini_config += "hpx.cmd_line_help=" +
                        detail::encode_string(strm.str());
                    ini_config += "hpx.cmd_line_help_option=" + help_option;
                }
            }

            bool debug_clp = vm.count("debug-clp") ? true : false;

            // create host name mapping
            util::map_hostnames mapnames(debug_clp);
            if (vm.count("ifsuffix"))
                mapnames.use_suffix(vm["ifsuffix"].as<std::string>());
            if (vm.count("ifprefix"))
                mapnames.use_prefix(vm["ifprefix"].as<std::string>());

            // The AGAS host name and port number are pre-initialized from
            //the command line
            std::string agas_host;
            boost::uint16_t agas_port = HPX_INITIAL_IP_PORT;
            if (vm.count("agas"))
                util::split_ip_address(vm["agas"].as<std::string>(), agas_host, agas_port);

            // Check command line arguments.
            util::pbs_environment env(debug_clp);

            if (vm.count("iftransform")) {
                util::sed_transform iftransform(vm["iftransform"].as<std::string>());

                // Check for parsing failures
                if (!iftransform) {
                    throw std::logic_error(boost::str(boost::format(
                        "Could not parse --iftransform argument '%1%'") %
                        vm["iftransform"].as<std::string>()));
                }

                typedef util::pbs_environment::transform_function_type
                    transform_function_type;
                env.use_transform(transform_function_type(iftransform));
            }

            if (vm.count("nodefile")) {
                if (vm.count("nodes")) {
                    throw std::logic_error("Ambiguous command line options. "
                        "Do not specify more than one of the --nodefile and "
                        "--nodes options at the same time.");
                }
                ini_config += "hpx.nodefile=" +
                    env.init_from_file(vm["nodefile"].as<std::string>(), agas_host);
            }
            else if (vm.count("nodes")) {
                ini_config += "hpx.nodes=" + env.init_from_nodelist(
                    vm["nodes"].as<std::vector<std::string> >(), agas_host);
            }

            // let the PBS environment decide about the AGAS host
            agas_host = env.agas_host_name(
                agas_host.empty() ? HPX_INITIAL_IP_ADDRESS : agas_host);

            std::string hpx_host(env.host_name(HPX_INITIAL_IP_ADDRESS));
            boost::uint16_t hpx_port = HPX_INITIAL_IP_PORT;
            std::size_t num_threads = env.retrieve_number_of_threads();
            bool run_agas_server = vm.count("run-agas-server") ? true : false;

            std::size_t num_localities = env.retrieve_number_of_localities();
            if (vm.count("localities"))
                num_localities = vm["localities"].as<std::size_t>();

            std::size_t node = env.retrieve_node_number();

            // we initialize certain settings if --node is specified (or data
            // has been retrieved from the environment)
            if (node != std::size_t(-1) || vm.count("node")) {
                // command line overwrites the environment
                if (vm.count("node")) {
                    if (vm.count("agas")) {
                        throw std::logic_error("Command line option --node "
                            "is not compatible with --agas/-a");
                    }
                    node = vm["node"].as<std::size_t>();
                    if (1 == num_localities && !vm.count("localities")) {
                        throw std::logic_error("Command line option --node "
                            "requires to specify the number of localities as "
                            "well (for instance by using --localities/-l)");
                    }
                }
                if (env.agas_node() == node) {
                    // console node, by default runs AGAS
                    run_agas_server = true;
                    mode = hpx::runtime_mode_console;
                }
                else {
                    hpx_port += static_cast<boost::uint16_t>(node);         // each node gets an unique port
                    mode = hpx::runtime_mode_worker;

                    // do not execute any explicit hpx_main except if asked
                    // otherwise
                    if (!vm.count("run-hpx-main"))
                        f = 0;
                }
                // store node number in configuration
                ini_config += "hpx.locality=" +
                    boost::lexical_cast<std::string>(node + 1);
            }

            if (vm.count("ini")) {
                std::vector<std::string> cfg =
                    vm["ini"].as<std::vector<std::string> >();
                std::copy(cfg.begin(), cfg.end(), std::back_inserter(ini_config));
            }

            if (vm.count("hpx"))
                util::split_ip_address(vm["hpx"].as<std::string>(), hpx_host, hpx_port);

            if (vm.count("threads")) {
                if (env.run_with_pbs()) {
                    std::cerr  << "hpx::init: command line warning: --threads/-t "
                        "used used when running with PBS, the application might"
                        "not run properly." << std::endl;
                }
                num_threads = vm["threads"].as<std::size_t>();
            }

            std::string queueing("priority_local");
            if (vm.count("queueing"))
                queueing = vm["queueing"].as<std::string>();

            // If the user has not specified an explicit runtime mode we
            // retrieve it from the command line.
            if (hpx::runtime_mode_default == mode) {
                // The default mode is console, i.e. all workers need to be
                // started with --worker/-w.
                mode = hpx::runtime_mode_console;
                if (vm.count("console") + vm.count("worker") + vm.count("connect") > 1) {
                    throw std::logic_error("Ambiguous command line options. "
                        "Do not specify more than one of --console/-c, "
                        "--worker/-w, or --connect");
                }

                // In these cases we default to executing with an empty
                // hpx_main, except if specified otherwise.
                if (vm.count("worker")) {
                    mode = hpx::runtime_mode_worker;

                    // do not execute any explicit hpx_main except if asked
                    // otherwise
                    if (!vm.count("run-hpx-main"))
                        f = 0;
                }
                else if (vm.count("connect")) {
                    mode = hpx::runtime_mode_connect;

                    // do not execute any explicit hpx_main except if asked
                    // otherwise
                    if (!vm.count("run-hpx-main"))
                        f = 0;
                }
            }

            // map host names to ip addresses, if requested
            hpx_host = mapnames.map(hpx_host, hpx_port);
            agas_host = mapnames.map(agas_host, agas_port);

            // sanity checks
            if (num_localities == 1 && !vm.count("agas") && !vm.count("node")) {
                // We assume we have to run the AGAS server if the number of
                // localities to run on is not specified (or is '1')
                // and no additional option (--agas or --node) has been
                // specified. That simplifies running small standalone
                // applications on one locality.
                run_agas_server = true;
            }

            if (hpx_host == agas_host && hpx_port == agas_port) {
                // we assume that we need to run the agas server if the user
                // asked for the same network addresses for HPX and AGAS
                run_agas_server = true;
            }
            else if (run_agas_server) {
                // otherwise, if the user instructed us to run the AGAS server,
                // we set the AGAS network address to the same value as the HPX
                // network address
                agas_host = hpx_host;
                agas_port = hpx_port;
            }
            else if (env.run_with_pbs()) {
                // in PBS mode, if the network addresses are different and we
                // should not run the AGAS server we assume to be in worker mode
                mode = hpx::runtime_mode_worker;

                // do not execute any explicit hpx_main except if asked
                // otherwise
                if (!vm.count("run-hpx-main"))
                    f = 0;
            }

            // write HPX and AGAS network parameters to the proper ini-file entries
            ini_config += "hpx.address=" + hpx_host;
            ini_config += "hpx.port=" + boost::lexical_cast<std::string>(hpx_port);
            ini_config += "hpx.agas.address=" + agas_host;
            ini_config += "hpx.agas.port=" + boost::lexical_cast<std::string>(agas_port);

            if (run_agas_server) {
                ini_config += "hpx.agas.service_mode=bootstrap";
                if (vm.count("run-agas-server-only"))
                    ini_config += "hpx.components.load_external=0";
            }
            else if (vm.count("run-agas-server-only") && !env.run_with_pbs()) {
                throw std::logic_error("Command line option --run-agas-server-only "
                    "can be specified only for the node running the AGAS server.");
            }
            if (1 == num_localities && vm.count("run-agas-server-only")) {
                std::cerr  << "hpx::init: command line warning: --run-agas-server-only "
                    "used for single locality execution, application might "
                    "not run properly." << std::endl;
            }

            // Set whether the AGAS server is running as a dedicated runtime.
            // This decides whether the AGAS actions are executed with normal
            // priority (if dedicated) or with high priority (non-dedicated)
            if (vm.count("run-agas-server-only"))
                ini_config += "hpx.agas.dedicated_server=1";

            if (vm.count("debug-hpx-log")) {
                ini_config += "hpx.logging.console.destination=" +
                    vm["debug-hpx-log"].as<std::string>();
                ini_config += "hpx.logging.destination=" +
                    vm["debug-hpx-log"].as<std::string>();
                ini_config += "hpx.logging.console.level=5";
                ini_config += "hpx.logging.level=5";
            }

            if (vm.count("debug-agas-log")) {
                ini_config += "hpx.logging.console.agas.destination=" +
                    vm["debug-agas-log"].as<std::string>();
                ini_config += "hpx.logging.agas.destination=" +
                    vm["debug-agas-log"].as<std::string>();
                ini_config += "hpx.logging.console.agas.level=5";
                ini_config += "hpx.logging.agas.level=5";
            }

            // Collect the command line for diagnostic purposes.
            std::string cmd_line;
            for (int i = 0; i < argc; ++i)
            {
                // quote only if it contains whitespace
                std::string arg(argv[i]);
                cmd_line += detail::enquote(arg);

                if ((i + 1) != argc)
                    cmd_line += " ";
            }

            // Store the program name and the command line.
            ini_config += "hpx.program_name=" + std::string(argv[0]);
            ini_config += "hpx.cmd_line=" + cmd_line;

            // Set number of OS threads in configuration.
            ini_config += "hpx.os_threads=" +
                boost::lexical_cast<std::string>(num_threads);

            // Set number of localities in configuration (do it everywhere,
            // even if this information is only used by the AGAS server).
            ini_config += "hpx.localities=" +
                boost::lexical_cast<std::string>(num_localities);

//             // If we are running on one locality we should not throttle the
//             // number of concurrent AGAS resolve requests.
//             if (1 == num_localities) {
//                 ini_config += "hpx.agas.max_resolve_requests=" +
//                     boost::lexical_cast<std::string>(
//                         (std::numeric_limits<int>::max)());
//             }

            // FIXME: AGAS V2: if a locality is supposed to run the AGAS
            //        service only and requests to use 'priority_local' as the
            //        scheduler, switch to the 'local' scheduler instead.
            ini_config += std::string("hpx.runtime_mode=") +
                get_runtime_mode_name(mode);

            if (debug_clp) {
                std::cerr << "Configuration before runtime start:\n";
                std::cerr << "-----------------------------------\n";
                BOOST_FOREACH(std::string const& s, ini_config) {
                    std::cerr << s << std::endl;
                }
                std::cerr << "-----------------------------------\n";
            }

            // Initialize and start the HPX runtime.
            if (0 == std::string("global").find(queueing)) {
#if defined(HPX_GLOBAL_SCHEDULER)
                result = detail::run_global(f, vm, mode, ini_config,
                    startup, shutdown, num_threads, num_localities);
#else
                throw std::logic_error("Command line option --queueing=global "
                    "is not configured in this build. Please rebuild with "
                    "'cmake -DHPX_GLOBAL_SCHEDULER=ON'.");
#endif
            }
            else if (0 == std::string("local").find(queueing)) {
#if defined(HPX_LOCAL_SCHEDULER)
                result = detail::run_local(f, vm, mode, ini_config,
                    startup, shutdown, num_threads, num_localities);
#else
                throw std::logic_error("Command line option --queueing=local "
                    "is not configured in this build. Please rebuild with "
                    "'cmake -DHPX_LOCAL_SCHEDULER=ON'.");
#endif
            }
            else if (0 == std::string("priority_local").find(queueing)) {
                // local scheduler with priority queue (one queue for each OS threads
                // plus one separate queue for high priority PX-threads)
                result = detail::run_priority_local(f, vm, mode, ini_config,
                    startup, shutdown, num_threads, num_localities);
            }
            else if (0 == std::string("abp").find(queueing)) {
                // abp scheduler: local dequeues for each OS thread, with work
                // stealing from the "bottom" of each.
#if defined(HPX_ABP_SCHEDULER)
                result = detail::run_abp(f, vm, mode, ini_config,
                    startup, shutdown, num_threads, num_localities);
#else
                throw std::logic_error("Command line option --queueing=abp "
                    "is not configured in this build. Please rebuild with "
                    "'cmake -DHPX_ABP_SCHEDULER=ON'.");
#endif
            }
            else if (0 == std::string("priority_abp").find(queueing)) {
                // priority abp scheduler: local priority dequeues for each
                // OS thread, with work stealing from the "bottom" of each.
#if defined(HPX_ABP_PRIORITY_SCHEDULER)
                result = detail::run_priority_abp(f, vm, mode, ini_config,
                    startup, shutdown, num_threads, num_localities);
#else
                throw std::logic_error("Command line option --queueing=priority_abp "
                    "is not configured in this build. Please rebuild with "
                    "'cmake -DHPX_ABP_PRIORITY_SCHEDULER=ON'.");
#endif
            }
            else if (0 == std::string("hierarchy").find(queueing)) {
#if defined(HPX_HIERARCHY_SCHEDULER)
                // hierarchy scheduler: tree of queues, with work
                // stealing from the parent queue in that tree.
                result = detail::run_hierarchy(f, vm, mode, ini_config,
                    startup, shutdown, num_threads, num_localities);
#else
                throw std::logic_error("Command line option --queueing=hierarchy "
                    "is not configured in this build. Please rebuild with "
                    "'cmake -DHPX_HIERARCHY_SCHEDULER=ON'.");
#endif
            }
            else if (0 == std::string("periodic").find(queueing)) {
#if defined(HPX_PERIODIC_PRIORITY_SCHEDULER)
                result = detail::run_periodic(f, vm, mode, ini_config,
                    startup, shutdown, num_threads, num_localities);
#else
                throw std::logic_error("Command line option --queueing=periodic "
                    "is not configured in this build. Please rebuild with "
                    "'cmake -DHPX_PERIODIC_PRIORITY_SCHEDULER=ON'.");
#endif
            }
            else {
                throw std::logic_error("Bad value for command line option "
                    "--queueing/-q");
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
    void finalize(double shutdown_timeout, double localwait)
    {
        if (localwait == -1.0)
            localwait = detail::get_option("hpx.finalize_wait_time", -1.0);

        if (localwait != -1.0) {
            hpx::util::high_resolution_timer t;
            double start_time = t.elapsed();
            double current = 0.0;
            do {
                current = t.elapsed();
            } while (current - start_time < localwait * 1e-6);
        }

        if (shutdown_timeout == -1.0)
            shutdown_timeout = detail::get_option("hpx.shutdown_timeout", -1.0);

        components::stubs::runtime_support::shutdown_all(
            naming::get_id_from_locality_id(HPX_AGAS_BOOTSTRAP_PREFIX),
            shutdown_timeout);
    }

    ///////////////////////////////////////////////////////////////////////////
    void disconnect(double shutdown_timeout, double localwait)
    {
        if (localwait == -1.0)
            localwait = detail::get_option("hpx.finalize_wait_time", -1.0);

        if (localwait != -1.0) {
            hpx::util::high_resolution_timer t;
            double start_time = t.elapsed();
            double current = 0.0;
            do {
                current = t.elapsed();
            } while (current - start_time < localwait * 1e-6);
        }

        if (shutdown_timeout == -1.0)
            shutdown_timeout = detail::get_option("hpx.shutdown_timeout", -1.0);

        components::server::runtime_support* p =
            reinterpret_cast<components::server::runtime_support*>(
                  get_runtime().get_runtime_support_lva());

        p->call_shutdown_functions(true);
        p->call_shutdown_functions(false);
        p->shutdown(shutdown_timeout);
    }

    ///////////////////////////////////////////////////////////////////////////
    void terminate()
    {
        components::stubs::runtime_support::terminate_all(
            naming::get_id_from_locality_id(HPX_AGAS_BOOTSTRAP_PREFIX));
    }
}

