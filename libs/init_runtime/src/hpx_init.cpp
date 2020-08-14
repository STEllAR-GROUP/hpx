//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c)      2017 Shoshana Jakobovits
//  Copyright (c) 2010-2011 Phillip LeBlanc, Dylan Stark
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>

#include <hpx/assert.hpp>
#include <hpx/command_line_handling/command_line_handling.hpp>
#include <hpx/coroutines/detail/context_impl.hpp>
#include <hpx/execution/detail/execution_parameter_callbacks.hpp>
#include <hpx/execution_base/register_locks.hpp>
#include <hpx/executors/exception_list.hpp>
#include <hpx/functional/bind_front.hpp>
#include <hpx/functional/function.hpp>
#include <hpx/futures/detail/future_data.hpp>
#include <hpx/hpx_finalize.hpp>
#include <hpx/hpx_suspend.hpp>
#include <hpx/hpx_user_main_config.hpp>
#include <hpx/init_runtime/detail/run_or_start.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/filesystem.hpp>
#include <hpx/modules/format.hpp>
#include <hpx/modules/logging.hpp>
#ifdef HPX_HAVE_LIB_MPI_BASE
#include <hpx/modules/mpi_base.hpp>
#endif
#include <hpx/modules/schedulers.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/modules/timing.hpp>
#include <hpx/parallel/util/detail/handle_exception_termination_handler.hpp>
#include <hpx/program_options/options_description.hpp>
#include <hpx/program_options/parsers.hpp>
#include <hpx/program_options/variables_map.hpp>
#include <hpx/resource_partitioner/partitioner.hpp>
#include <hpx/runtime/naming_fwd.hpp>
#include <hpx/runtime/runtime_fwd.hpp>
#include <hpx/runtime_local/config_entry.hpp>
#include <hpx/runtime_local/custom_exception_info.hpp>
#include <hpx/runtime_local/debugging.hpp>
#include <hpx/runtime_local/get_locality_id.hpp>
#include <hpx/runtime_local/runtime_handlers.hpp>
#include <hpx/runtime_local/runtime_local.hpp>
#include <hpx/runtime_local/shutdown_function.hpp>
#include <hpx/runtime_local/startup_function.hpp>
#include <hpx/string_util/classification.hpp>
#include <hpx/string_util/split.hpp>
#include <hpx/threading/thread.hpp>
#include <hpx/type_support/pack.hpp>
#include <hpx/util/from_string.hpp>

#include <hpx/program_options/options_description.hpp>
#include <hpx/program_options/parsers.hpp>
#include <hpx/program_options/variables_map.hpp>

#if defined(HPX_HAVE_DISTRIBUTED_RUNTIME)
#include <hpx/actions_base/plain_action.hpp>
#include <hpx/modules/async_distributed.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/components/runtime_support.hpp>
#include <hpx/runtime/find_localities.hpp>
#include <hpx/runtime_distributed.hpp>
#include <hpx/util/bind_action.hpp>
#include <hpx/util/init_logging.hpp>
#include <hpx/util/query_counters.hpp>
#include <hpx/util/register_locks_globally.hpp>
#endif

#if defined(HPX_NATIVE_MIC) || defined(__bgq__)
#include <cstdlib>
#endif

#include <cmath>
#include <cstddef>
#include <exception>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <new>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#if !defined(HPX_WINDOWS)
#include <signal.h>
#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx {
    void set_error_handlers();
}

///////////////////////////////////////////////////////////////////////////////
namespace hpx_startup {
    std::vector<std::string> (*user_main_config_function)(
        std::vector<std::string> const&) = nullptr;
}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace detail {
    // forward declarations only
    void console_print(std::string const&);
    void list_symbolic_name(std::string const&, hpx::id_type const&);
    void list_component_type(std::string const&, components::component_type);
}}    // namespace hpx::detail

#if defined(HPX_HAVE_DISTRIBUTED_RUNTIME)
HPX_PLAIN_ACTION_ID(hpx::detail::console_print, console_print_action,
    hpx::actions::console_print_action_id)
HPX_PLAIN_ACTION_ID(hpx::detail::list_component_type,
    list_component_type_action, hpx::actions::list_component_type_action_id)

typedef hpx::util::detail::bound_action<list_component_type_action,
    hpx::util::index_pack<0, 1, 2>, hpx::naming::id_type,
    hpx::util::detail::placeholder<1>, hpx::util::detail::placeholder<2>>
    bound_list_component_type_action;

HPX_UTIL_REGISTER_FUNCTION_DECLARATION(
    void(std::string const&, hpx::components::component_type),
    bound_list_component_type_action, list_component_type_function)

HPX_UTIL_REGISTER_FUNCTION(
    void(std::string const&, hpx::components::component_type),
    bound_list_component_type_action, list_component_type_function)
#endif

namespace hpx { namespace detail {
    ///////////////////////////////////////////////////////////////////////////
    // print string on the console
    void console_print(std::string const& name)
    {
        std::cout << name << std::endl;
    }

    inline void print(std::string const& name, error_code& ec = throws)
    {
#if defined(HPX_HAVE_DISTRIBUTED_RUNTIME)
        naming::id_type console(agas::get_console_locality(ec));
        if (ec)
            return;

        hpx::async<console_print_action>(console, name).get(ec);
        if (ec)
            return;
#else
        console_print(name);
#endif
        if (&ec != &throws)
            ec = make_success_code();
    }

#if defined(HPX_HAVE_DISTRIBUTED_RUNTIME)
    ///////////////////////////////////////////////////////////////////////////
    // redirect the printing of the given counter name to the console
    bool list_counter(
        performance_counters::counter_info const& info, error_code& ec)
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
        performance_counters::discover_counter_types(
            &list_counter, performance_counters::discover_counters_minimal);
    }

    void list_counter_names_full()
    {
        // list all counter names
        list_counter_names_header(false);
        performance_counters::discover_counter_types(
            &list_counter, performance_counters::discover_counters_full);
    }

    ///////////////////////////////////////////////////////////////////////////
    // redirect the printing of the full counter info to the console
    bool list_counter_info(
        performance_counters::counter_info const& info, error_code& ec)
    {
        // compose the information to be printed for each of the counters
        std::ostringstream strm;

        strm << std::string(78, '-') << '\n';
        strm << "fullname: " << info.fullname_ << '\n';
        strm << "helptext: " << info.helptext_ << '\n';
        strm << "type:     "
             << performance_counters::get_counter_type_name(info.type_) << '\n';

        strm << "version:  ";    // 0xMMmmrrrr
        hpx::util::format_to(strm, "{}.{}.{}\n", info.version_ / 0x1000000,
            info.version_ / 0x10000 % 0x100, info.version_ % 0x10000);
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
        performance_counters::discover_counter_types(
            &list_counter_info, performance_counters::discover_counters_full);
    }

    ///////////////////////////////////////////////////////////////////////////
    void list_symbolic_name(std::string const& name, hpx::id_type const& id)
    {
        std::ostringstream strm;

        strm << name << ", " << id << ", "
             << (id.get_management_type() == id_type::managed ? "managed" :
                                                                "unmanaged");

        print(strm.str());
    }

    void list_symbolic_names()
    {
        print(std::string("List of all registered symbolic names:"));
        print(std::string("--------------------------------------"));

        std::map<std::string, hpx::id_type> entries =
            agas::find_symbols(hpx::launch::sync);

        for (auto const& e : entries)
        {
            list_symbolic_name(e.first, e.second);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    void list_component_type(
        std::string const& name, components::component_type ctype)
    {
        print(hpx::util::format(
            "{1:-40}, {2}", name, components::get_component_type_name(ctype)));
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
        try
        {
            HPX_ASSERT(qc);
            qc->start();
        }
        catch (...)
        {
            std::cerr << hpx::diagnostic_information(std::current_exception())
                      << std::flush;
            hpx::terminate();
        }
    }
#endif
}}    // namespace hpx::detail

///////////////////////////////////////////////////////////////////////////////
#if defined(HPX_HAVE_DYNAMIC_HPX_MAIN) &&                                      \
    (defined(__linux) || defined(__linux__) || defined(linux) ||               \
        defined(__APPLE__))
namespace hpx_start {
    // Importing weak symbol from libhpx_wrap.a which may be shadowed by one present in
    // hpx_main.hpp.
    HPX_SYMBOL_EXPORT __attribute__((weak)) bool include_libhpx_wrap = false;
}    // namespace hpx_start

#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx {
    // Print stack trace and exit.
#if defined(HPX_WINDOWS)
    extern BOOL WINAPI termination_handler(DWORD ctrl_type);
#else
    extern void termination_handler(int signum);
#endif

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        ///////////////////////////////////////////////////////////////////////
        struct dump_config
        {
            dump_config(hpx::runtime const& rt)
              : rt_(std::cref(rt))
            {
            }

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
        void activate_global_options(
            util::command_line_handling& cms, int argc, char** argv)
        {
#if defined(__linux) || defined(linux) || defined(__linux__) ||                \
    defined(__FreeBSD__)
            threads::coroutines::detail::posix::use_guard_pages =
                cms.rtcfg_.use_stack_guard_pages();
#endif
#ifdef HPX_HAVE_VERIFY_LOCKS
            if (cms.rtcfg_.enable_lock_detection())
            {
                util::enable_lock_detection();
                util::trace_depth_lock_detection(cms.rtcfg_.trace_depth());
            }
            else
            {
                util::disable_lock_detection();
            }
#endif
#if defined(HPX_HAVE_DISTRIBUTED_RUNTIME) &&                                   \
    defined(HPX_HAVE_VERIFY_LOCKS_GLOBALLY)
            if (cms.rtcfg_.enable_global_lock_detection())
            {
                util::enable_global_lock_detection();
            }
            else
            {
                util::disable_global_lock_detection();
            }
#endif
#ifdef HPX_HAVE_THREAD_MINIMAL_DEADLOCK_DETECTION
            threads::policies::set_minimal_deadlock_detection_enabled(
                cms.rtcfg_.enable_minimal_deadlock_detection());
#endif
#ifdef HPX_HAVE_SPINLOCK_DEADLOCK_DETECTION
            util::detail::set_spinlock_break_on_deadlock_enabled(
                cms.rtcfg_.enable_spinlock_deadlock_detection());
            util::detail::set_spinlock_deadlock_detection_limit(
                cms.rtcfg_.get_spinlock_deadlock_detection_limit());
#endif

            // initialize logging
#if defined(HPX_HAVE_DISTRIBUTED_RUNTIME)
            util::detail::init_logging(
                cms.rtcfg_, cms.rtcfg_.mode_ == runtime_mode::console);
#endif

#if defined(HPX_HAVE_NETWORKING)
            if (cms.num_localities_ != 1 || cms.node_ != 0 ||
                cms.rtcfg_.enable_networking())
            {
                parcelset::parcelhandler::init(&argc, &argv, cms);
            }
#endif
        }

        ///////////////////////////////////////////////////////////////////////
#if defined(HPX_HAVE_DISTRIBUTED_RUNTIME)
        void handle_list_and_print_options(hpx::runtime& rt,
            hpx::program_options::variables_map& vm,
            bool print_counters_locally)
        {
            runtime_distributed* rtd =
                dynamic_cast<hpx::runtime_distributed*>(&rt);
            HPX_ASSERT(rtd != nullptr);
            if (vm.count("hpx:list-counters"))
            {
                // Print the names of all registered performance counters.
                std::string option(vm["hpx:list-counters"].as<std::string>());
                if (0 == std::string("minimal").find(option))
                    rt.add_startup_function(&list_counter_names_minimal);
                else if (0 == std::string("full").find(option))
                    rt.add_startup_function(&list_counter_names_full);
                else
                {
                    std::string msg("Invalid command line option value"
                                    "for --hpx:list-counters: ");
                    msg += option;
                    msg += ", allowed values are 'minimal' and 'full'";
                    throw detail::command_line_error(msg.c_str());
                }
            }
            if (vm.count("hpx:list-counter-infos"))
            {
                // Print info about all registered performance counters.
                std::string option(
                    vm["hpx:list-counter-infos"].as<std::string>());
                if (0 == std::string("minimal").find(option))
                    rt.add_startup_function(&list_counter_infos_minimal);
                else if (0 == std::string("full").find(option))
                    rt.add_startup_function(&list_counter_infos_full);
                else
                {
                    std::string msg("Invalid command line option value"
                                    "for --hpx:list-counter-infos: ");
                    msg += option;
                    msg += ", allowed values are 'minimal' and 'full'";
                    throw detail::command_line_error(msg.c_str());
                }
            }
            if (vm.count("hpx:list-symbolic-names"))
            {
                // Print all registered symbolic names.
                rt.add_startup_function(&list_symbolic_names);
            }
            if (vm.count("hpx:list-component-types"))
            {
                // Print all registered component types.
                rt.add_startup_function(&list_component_types);
            }

            if (vm.count("hpx:print-counter") ||
                vm.count("hpx:print-counter-reset"))
            {
                std::size_t interval = 0;
                if (vm.count("hpx:print-counter-interval"))
                {
                    interval =
                        vm["hpx:print-counter-interval"].as<std::size_t>();
                }

                std::vector<std::string> counters;
                if (vm.count("hpx:print-counter"))
                {
                    counters =
                        vm["hpx:print-counter"].as<std::vector<std::string>>();
                }

                std::vector<std::string> reset_counters;
                if (vm.count("hpx:print-counter-reset"))
                {
                    reset_counters = vm["hpx:print-counter-reset"]
                                         .as<std::vector<std::string>>();
                }

                std::vector<std::string> counter_shortnames;
                std::string counter_format("normal");
                if (vm.count("hpx:print-counter-format"))
                {
                    counter_format =
                        vm["hpx:print-counter-format"].as<std::string>();
                    if (counter_format == "csv-short")
                    {
                        for (std::size_t i = 0; i != counters.size(); ++i)
                        {
                            std::vector<std::string> entry;
                            hpx::string_util::split(entry, counters[i],
                                hpx::string_util::is_any_of(","),
                                hpx::string_util::token_compress_mode::on);

                            if (entry.size() != 2)
                            {
                                throw detail::command_line_error(
                                    "Invalid format for command line "
                                    "option "
                                    "--hpx:print-counter-format=csv-short");
                            }

                            counter_shortnames.push_back(entry[0]);
                            counters[i] = entry[1];
                        }
                    }
                }

                bool csv_header = true;
                if (vm.count("hpx:no-csv-header"))
                    csv_header = false;

                std::string destination("cout");
                if (vm.count("hpx:print-counter-destination"))
                    destination =
                        vm["hpx:print-counter-destination"].as<std::string>();

                bool counter_types = false;
                if (vm.count("hpx:print-counter-types"))
                    counter_types = true;

                // schedule the query function at startup, which will schedule
                // itself to run after the given interval
                std::shared_ptr<util::query_counters> qc =
                    std::make_shared<util::query_counters>(std::ref(counters),
                        std::ref(reset_counters), interval, destination,
                        counter_format, counter_shortnames, csv_header,
                        print_counters_locally, counter_types);

                // schedule to print counters at shutdown, if requested
                if (get_config_entry("hpx.print_counter.shutdown", "0") == "1")
                {
                    // schedule to run at shutdown
                    rt.add_pre_shutdown_function(util::bind_front(
                        &util::query_counters::evaluate, qc, true));
                }

                // schedule to start all counters

                rt.add_startup_function(util::bind_front(&start_counters, qc));

                // register the query_counters object with the runtime system
                rtd->register_query_counters(qc);
            }
            else if (vm.count("hpx:print-counter-interval"))
            {
                throw detail::command_line_error(
                    "Invalid command line option "
                    "--hpx:print-counter-interval, valid in conjunction "
                    "with "
                    "--hpx:print-counter only");
            }
            else if (vm.count("hpx:print-counter-destination"))
            {
                throw detail::command_line_error(
                    "Invalid command line option "
                    "--hpx:print-counter-destination, valid in conjunction "
                    "with "
                    "--hpx:print-counter only");
            }
            else if (vm.count("hpx:print-counter-format"))
            {
                throw detail::command_line_error(
                    "Invalid command line option "
                    "--hpx:print-counter-format, valid in conjunction with "
                    "--hpx:print-counter only");
            }
            else if (vm.count("hpx:print-counter-at"))
            {
                throw detail::command_line_error(
                    "Invalid command line option "
                    "--hpx:print-counter-at, valid in conjunction with "
                    "--hpx:print-counter only");
            }
            else if (vm.count("hpx:reset-counters"))
            {
                throw detail::command_line_error(
                    "Invalid command line option "
                    "--hpx:reset-counters, valid in conjunction with "
                    "--hpx:print-counter only");
            }
        }
#endif

        void add_startup_functions(hpx::runtime& rt,
            hpx::program_options::variables_map& vm, runtime_mode mode,
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

#if defined(HPX_HAVE_DISTRIBUTED_RUNTIME)
            // Add startup function related to listing counter names or counter
            // infos (on console only).
            bool print_counters_locally =
                vm.count("hpx:print-counters-locally") != 0;
            if (mode == runtime_mode::console || print_counters_locally)
                handle_list_and_print_options(rt, vm, print_counters_locally);
#endif

            // Dump the configuration before all components have been loaded.
            if (vm.count("hpx:dump-config-initial"))
            {
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
            util::function_nonser<int(
                hpx::program_options::variables_map& vm)> const& f,
            hpx::program_options::variables_map& vm, runtime_mode mode,
            startup_function_type startup, shutdown_function_type shutdown)
        {
            LPROGRESS_;

            add_startup_functions(
                rt, vm, mode, std::move(startup), std::move(shutdown));

            // Run this runtime instance using the given function f.
            if (!f.empty())
                return rt.run(util::bind_front(f, vm));

            // Run this runtime instance without an hpx_main
            return rt.run();
        }

        int start(hpx::runtime& rt,
            util::function_nonser<int(
                hpx::program_options::variables_map& vm)> const& f,
            hpx::program_options::variables_map& vm, runtime_mode mode,
            startup_function_type startup, shutdown_function_type shutdown)
        {
            LPROGRESS_;

            add_startup_functions(
                rt, vm, mode, std::move(startup), std::move(shutdown));

            if (!f.empty())
            {
                // Run this runtime instance using the given function f.
                return rt.start(util::bind_front(f, vm));
            }

            // Run this runtime instance without an hpx_main
            return rt.start();
        }

        int run_or_start(bool blocking, std::unique_ptr<hpx::runtime> rt,
            util::command_line_handling& cfg, startup_function_type startup,
            shutdown_function_type shutdown)
        {
            if (blocking)
            {
                return run(*rt, cfg.hpx_main_f_, cfg.vm_, cfg.rtcfg_.mode_,
                    std::move(startup), std::move(shutdown));
            }

            // non-blocking version
            start(*rt, cfg.hpx_main_f_, cfg.vm_, cfg.rtcfg_.mode_,
                std::move(startup), std::move(shutdown));

            // pointer to runtime is stored in TLS
            hpx::runtime* p = rt.release();
            (void) p;

            return 0;
        }

        ////////////////////////////////////////////////////////////////////////
        void init_environment()
        {
            HPX_UNUSED(hpx::filesystem::initial_path());

            hpx::assertion::set_assertion_handler(&detail::assertion_handler);
            hpx::util::set_test_failure_handler(&detail::test_failure_handler);
#if defined(HPX_HAVE_APEX)
            hpx::util::set_enable_parent_task_handler(
                &detail::enable_parent_task_handler);
#endif
            hpx::set_custom_exception_info_handler(
                &detail::custom_exception_info);
            hpx::serialization::detail::set_save_custom_exception_handler(
                &runtime_local::detail::save_custom_exception);
            hpx::serialization::detail::set_load_custom_exception_handler(
                &runtime_local::detail::load_custom_exception);
            hpx::set_pre_exception_handler(&detail::pre_exception_handler);
            hpx::set_thread_termination_handler(
                [](std::exception_ptr const& e) { report_error(e); });
            hpx::lcos::detail::set_run_on_completed_error_handler(
                [](std::exception_ptr const& e) {
                    report_exception_and_terminate(e);
                });
#if defined(HPX_HAVE_VERIFY_LOCKS)
            hpx::util::set_registered_locks_error_handler(
                &detail::registered_locks_error_handler);
            hpx::util::set_register_locks_predicate(
                &detail::register_locks_predicate);
#endif
#if !defined(HPX_HAVE_DISABLED_SIGNAL_EXCEPTION_HANDLERS)
            set_error_handlers();
#endif
            hpx::threads::detail::set_get_default_pool(
                &detail::get_default_pool);
            hpx::threads::detail::set_get_default_timer_service(
                &hpx::detail::get_default_timer_service);
            hpx::threads::detail::set_get_locality_id(&get_locality_id);
            hpx::parallel::execution::detail::set_get_pu_mask(
                &hpx::detail::get_pu_mask);
            hpx::parallel::execution::detail::set_get_os_thread_count(
                []() { return hpx::get_os_thread_count(); });
            hpx::parallel::v1::detail::set_exception_list_termination_handler(
                &hpx::terminate);
            hpx::parallel::util::detail::
                set_parallel_exception_termination_handler(&hpx::terminate);

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
        }

        // make sure the runtime system is not active yet
        int ensure_no_runtime_is_up()
        {
            // make sure the runtime system is not active yet
            if (get_runtime_ptr() != nullptr)
            {
#if defined(HPX_HAVE_DYNAMIC_HPX_MAIN) &&                                      \
    (defined(__linux) || defined(__linux__) || defined(linux) ||               \
        defined(__APPLE__))
                // make sure the runtime system is not initialized
                // after its activation from int main()
                if (hpx_start::include_libhpx_wrap)
                {
                    std::cerr
                        << "hpx is already initialized from main.\n"
                           "Note: Delete hpx_main.hpp to initialize hpx system "
                           "using hpx::init. Exiting...\n";
                    return -1;
                }
#endif
                std::cerr << "hpx::init: can't initialize runtime system "
                             "more than once! Exiting...\n";
                return -1;
            }
            return 0;
        }

        ///////////////////////////////////////////////////////////////////////
        int run_or_start(util::function_nonser<int(
                             hpx::program_options::variables_map& vm)> const& f,
            int argc, char** argv, init_params const& params, bool blocking)
        {
            init_environment();

            int result = 0;
            try
            {
                // make sure the runtime system is not active yet
                if ((result = ensure_no_runtime_is_up()) != 0)
                {
                    return result;
                }

                // scope exception handling to resource partitioner initialization
                // any exception thrown during run_or_start below are handled
                // separately
                try
                {
                    // Construct resource partitioner if this has not been done
                    // yet and get a handle to it (if the command-line parsing
                    // has not yet been done, do it now)
                    std::vector<
                        std::shared_ptr<components::component_registry_base>>
                        component_registries;
                    hpx::resource::partitioner rp =
                        hpx::resource::detail::make_partitioner(f,
                            params.desc_cmdline, argc, argv,
                            hpx_startup::user_main_config(params.cfg),
                            params.rp_mode, params.mode, false,
                            component_registries, &result);

                    for (auto& registry : component_registries)
                    {
                        hpx::register_startup_function([registry]() {
                            registry->register_component_type();
                        });
                    }

                    activate_global_options(
                        rp.get_command_line_switches(), argc, argv);

                    // check whether HPX should be exited at this point
                    // (parse_result is returning a result > 0, if the program options
                    // contain --hpx:help or --hpx:version, on error result is < 0)
                    if (result != 0)
                    {
                        if (result > 0)
                            result = 0;
                        return result;
                    }

                    // If thread_pools initialization in user main
                    if (params.rp_callback)
                    {
                        params.rp_callback(rp);
                    }

                    // Setup all internal parameters of the resource_partitioner
                    rp.configure_pools();
                }
                catch (hpx::exception const& e)
                {
                    std::cerr << "hpx::init: hpx::exception caught: "
                              << hpx::get_error_what(e) << "\n";
                    return -1;
                }

                // Initialize and start the HPX runtime.
                LPROGRESS_ << "run_local: create runtime";

                util::command_line_handling& cms =
                    resource::get_partitioner().get_command_line_switches();

                // Build and configure this runtime instance.
                std::unique_ptr<hpx::runtime> rt;

                // Command line handling should have updated this by now.
                HPX_ASSERT(cms.rtcfg_.mode_ != runtime_mode::default_);
                switch (cms.rtcfg_.mode_)
                {
                case runtime_mode::local:
                {
                    LPROGRESS_ << "creating local runtime";
                    rt.reset(new hpx::runtime(cms.rtcfg_));
                    break;
                }
                default:
                {
#if defined(HPX_HAVE_DISTRIBUTED_RUNTIME)
                    LPROGRESS_ << "creating distributed runtime";
                    rt.reset(new hpx::runtime_distributed(cms.rtcfg_));
                    break;
#else
                    char const* mode_name =
                        get_runtime_mode_name(cms.rtcfg_.mode_);
                    std::ostringstream s;
                    s << "Attempted to start the runtime in the mode \""
                      << mode_name
                      << "\", but HPX was compiled with "
                         "HPX_WITH_DISTRIBUTED_RUNTIME=OFF, and \""
                      << mode_name
                      << "\" requires HPX_WITH_DISTRIBUTED_RUNTIME=ON. "
                         "Recompile HPX with HPX_WITH_DISTRIBUTED_RUNTIME=ON "
                         "or change the runtime mode.";
                    HPX_THROW_EXCEPTION(
                        invalid_status, "run_or_start", s.str());
                    break;
#endif
                }
                }

                result = run_or_start(blocking, std::move(rt), cms,
                    std::move(params.startup), std::move(params.shutdown));
            }
            catch (detail::command_line_error const& e)
            {
                std::cerr << "hpx::init: std::exception caught: " << e.what()
                          << "\n";
                return -1;
            }
            return result;
        }

        ////////////////////////////////////////////////////////////////////////
        template <typename T>
        inline T get_option(std::string const& config, T default_ = T())
        {
            if (!config.empty())
            {
                try
                {
                    return hpx::util::from_string<T>(
                        get_runtime().get_config().get_entry(config, default_));
                }
                catch (hpx::util::bad_lexical_cast const&)
                {
                    ;    // do nothing
                }
            }
            return default_;
        }
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    int finalize(double shutdown_timeout, double localwait, error_code& ec)
    {
        if (!threads::get_self_ptr())
        {
            HPX_THROWS_IF(ec, invalid_status, "hpx::finalize",
                "this function can be called from an HPX thread only");
            return -1;
        }

        if (!is_running())
        {
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
            do
            {
                current = t.elapsed();
            } while (current - start_time < localwait * 1e-6);
        }

        if (std::abs(shutdown_timeout + 1.0) < 1e-16)
            shutdown_timeout = detail::get_option("hpx.shutdown_timeout", -1.0);

        runtime* rt = get_runtime_ptr();
        if (nullptr == rt)
        {
            HPX_THROWS_IF(ec, invalid_status, "hpx::finalize",
                "the runtime system is not active (did you already "
                "call hpx::stop?)");
            return -1;
        }

        rt->finalize(shutdown_timeout);

        return 0;
    }

    ///////////////////////////////////////////////////////////////////////////
    int disconnect(double shutdown_timeout, double localwait, error_code& ec)
    {
#if defined(HPX_HAVE_DISTRIBUTED_RUNTIME)
        if (!threads::get_self_ptr())
        {
            HPX_THROWS_IF(ec, invalid_status, "hpx::disconnect",
                "this function can be called from an HPX thread only");
            return -1;
        }

        if (!is_running())
        {
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
            do
            {
                current = t.elapsed();
            } while (current - start_time < localwait * 1e-6);
        }

        if (std::abs(shutdown_timeout + 1.0) < 1e-16)
            shutdown_timeout = detail::get_option("hpx.shutdown_timeout", -1.0);

        components::server::runtime_support* p =
            reinterpret_cast<components::server::runtime_support*>(
                get_runtime_distributed().get_runtime_support_lva());

        if (nullptr == p)
        {
            HPX_THROWS_IF(ec, invalid_status, "hpx::disconnect",
                "the runtime system is not active (did you already "
                "call finalize?)");
            return -1;
        }

        p->call_shutdown_functions(true);
        p->call_shutdown_functions(false);

        p->stop(shutdown_timeout, naming::invalid_id, true);
#endif

        return 0;
    }

    ///////////////////////////////////////////////////////////////////////////
    void terminate()
    {
        if (!threads::get_self_ptr())
        {
            // hpx::terminate shouldn't be called from a non-HPX thread
            std::terminate();
        }

#if defined(HPX_HAVE_DISTRIBUTED_RUNTIME)
        components::server::runtime_support* p =
            reinterpret_cast<components::server::runtime_support*>(
                get_runtime_distributed().get_runtime_support_lva());

        if (nullptr == p)
        {
            // the runtime system is not running, just terminate
            std::terminate();
        }

        p->terminate_all();
#else
        std::terminate();
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    int stop(error_code& ec)
    {
        if (threads::get_self_ptr())
        {
            HPX_THROWS_IF(ec, invalid_status, "hpx::stop",
                "this function cannot be called from an HPX thread");
            return -1;
        }

        std::unique_ptr<runtime> rt(get_runtime_ptr());    // take ownership!
        if (nullptr == rt.get())
        {
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

    ///////////////////////////////////////////////////////////////////////////
    int suspend(error_code& ec)
    {
        if (threads::get_self_ptr())
        {
            HPX_THROWS_IF(ec, invalid_status, "hpx::suspend",
                "this function cannot be called from an HPX thread");
            return -1;
        }

        runtime* rt = get_runtime_ptr();
        if (nullptr == rt)
        {
            HPX_THROWS_IF(ec, invalid_status, "hpx::suspend",
                "the runtime system is not active (did you already "
                "call hpx::stop?)");
            return -1;
        }

        return rt->suspend();
    }

    ///////////////////////////////////////////////////////////////////////////
    int resume(error_code& ec)
    {
        if (threads::get_self_ptr())
        {
            HPX_THROWS_IF(ec, invalid_status, "hpx::resume",
                "this function cannot be called from an HPX thread");
            return -1;
        }

        runtime* rt = get_runtime_ptr();
        if (nullptr == rt)
        {
            HPX_THROWS_IF(ec, invalid_status, "hpx::resume",
                "the runtime system is not active (did you already "
                "call hpx::stop?)");
            return -1;
        }

        return rt->resume();
    }

    namespace detail {
        int init_helper(hpx::program_options::variables_map& /*vm*/,
            util::function_nonser<int(int, char**)> const& f)
        {
            std::string cmdline(
                hpx::get_config_entry("hpx.reconstructed_cmd_line", ""));

            using namespace hpx::program_options;
#if defined(HPX_WINDOWS)
            std::vector<std::string> args = split_winmain(cmdline);
#else
            std::vector<std::string> args = split_unix(cmdline);
#endif

            // Copy all arguments which are not hpx related to a temporary array
            std::vector<char*> argv(args.size() + 1);
            std::size_t argcount = 0;
            for (std::size_t i = 0; i != args.size(); ++i)
            {
                if (0 != args[i].find("--hpx:"))
                {
                    argv[argcount++] = const_cast<char*>(args[i].data());
                }
                else if (6 == args[i].find("positional", 6))
                {
                    std::string::size_type p = args[i].find_first_of('=');
                    if (p != std::string::npos)
                    {
                        args[i] = args[i].substr(p + 1);
                        argv[argcount++] = const_cast<char*>(args[i].data());
                    }
                }
            }

            // add a single nullptr in the end as some application rely on that
            argv[argcount] = nullptr;

            // Invoke custom startup functions
            return f(static_cast<int>(argcount), argv.data());
        }
    }    // namespace detail
}    // namespace hpx
