//  Copyright (c)      2021 ETH Zurich
//  Copyright (c)      2018 Mikael Simberg
//  Copyright (c) 2007-2022 Hartmut Kaiser
//  Copyright (c) 2010-2011 Phillip LeBlanc, Dylan Stark
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/functional/bind.hpp>
#include <hpx/functional/bind_back.hpp>
#include <hpx/functional/function.hpp>
#include <hpx/modules/program_options.hpp>
#include <hpx/preprocessor/stringize.hpp>
#include <hpx/resource_partitioner/partitioner.hpp>
#include <hpx/runtime_local/runtime_local.hpp>
#include <hpx/runtime_local/shutdown_function.hpp>
#include <hpx/runtime_local/startup_function.hpp>

#include <csignal>
#include <cstddef>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#if !defined(HPX_PREFIX)
#define HPX_PREFIX ""
#endif

#if defined(HPX_APPLICATION_NAME_DEFAULT) && !defined(HPX_APPLICATION_NAME)
#define HPX_APPLICATION_NAME HPX_APPLICATION_NAME_DEFAULT
#endif

#if !defined(HPX_APPLICATION_STRING)
#if defined(HPX_APPLICATION_NAME)
#define HPX_APPLICATION_STRING HPX_PP_STRINGIZE(HPX_APPLICATION_NAME)
#else
#define HPX_APPLICATION_STRING "unknown HPX application"
#endif
#endif

#if defined(__FreeBSD__)
extern HPX_CORE_EXPORT char** freebsd_environ;
extern char** environ;
#endif

namespace hpx {
    namespace detail {
        HPX_CORE_EXPORT int init_helper(hpx::program_options::variables_map&,
            hpx::function<int(int, char**)> const&);
    }

    namespace local {
        namespace detail {
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

            // Default params to initialize the init_params struct
            [[maybe_unused]] static int dummy_argc = 1;
            [[maybe_unused]] static char app_name[256] = HPX_APPLICATION_STRING;
            static char* default_argv[2] = {app_name, nullptr};
            [[maybe_unused]] static char** dummy_argv = default_argv;

            // HPX_APPLICATION_STRING is specific to an application and therefore
            // cannot be in the source file
            HPX_CORE_EXPORT hpx::program_options::options_description const&
            default_desc(char const*);

            // Utilities to init the thread_pools of the resource partitioner
            using rp_callback_type =
                hpx::function<void(hpx::resource::partitioner&,
                    hpx::program_options::variables_map const&)>;
        }    // namespace detail

        struct init_params
        {
            init_params()
            {
                std::strncpy(detail::app_name, HPX_APPLICATION_STRING,
                    sizeof(detail::app_name) - 1);
            }

            std::reference_wrapper<
                hpx::program_options::options_description const>
                desc_cmdline = detail::default_desc(HPX_APPLICATION_STRING);
            std::vector<std::string> cfg;
            mutable startup_function_type startup;
            mutable shutdown_function_type shutdown;
            hpx::resource::partitioner_mode rp_mode =
                ::hpx::resource::partitioner_mode::default_;
            hpx::local::detail::rp_callback_type rp_callback;
        };

        namespace detail {
            HPX_CORE_EXPORT int run_or_start(
                hpx::function<int(
                    hpx::program_options::variables_map& vm)> const& f,
                int argc, char** argv, init_params const& params,
                bool blocking);

            inline int init_start_impl(
                hpx::function<int(hpx::program_options::variables_map&)> f,
                int argc, char** argv, init_params const& params, bool blocking)
            {
                if (argc == 0 || argv == nullptr)
                {
                    argc = dummy_argc;
                    argv = dummy_argv;
                }

                util::set_hpx_prefix(HPX_PREFIX);
#if defined(__FreeBSD__)
                freebsd_environ = environ;
#endif
                // set a handler for std::abort
                std::signal(SIGABRT, hpx::detail::on_abort);
                std::atexit(hpx::detail::on_exit);
#if defined(HPX_HAVE_CXX11_STD_QUICK_EXIT)
                std::at_quick_exit(hpx::detail::on_exit);
#endif
                return run_or_start(f, argc, argv, params, blocking);
            }
        }    // namespace detail

        inline int init(
            std::function<int(hpx::program_options::variables_map&)> f,
            int argc, char** argv, init_params const& params = init_params())
        {
            return detail::init_start_impl(
                HPX_MOVE(f), argc, argv, params, true);
        }

        inline int init(std::function<int(int, char**)> f, int argc,
            char** argv, init_params const& params = init_params())
        {
            hpx::function<int(hpx::program_options::variables_map&)> main_f =
                hpx::bind_back(hpx::detail::init_helper, f);
            return detail::init_start_impl(
                HPX_MOVE(main_f), argc, argv, params, true);
        }

        inline int init(std::function<int()> f, int argc, char** argv,
            init_params const& params = init_params())
        {
            hpx::function<int(hpx::program_options::variables_map&)> main_f =
                hpx::bind(f);
            return detail::init_start_impl(
                HPX_MOVE(main_f), argc, argv, params, true);
        }

        inline int init(std::nullptr_t, int argc, char** argv,
            init_params const& params = init_params())
        {
            hpx::function<int(hpx::program_options::variables_map&)> main_f;
            return detail::init_start_impl(
                HPX_MOVE(main_f), argc, argv, params, true);
        }

        inline bool start(
            std::function<int(hpx::program_options::variables_map&)> f,
            int argc, char** argv, init_params const& params = init_params())
        {
            return 0 ==
                detail::init_start_impl(HPX_MOVE(f), argc, argv, params, false);
        }

        inline bool start(std::function<int(int, char**)> f, int argc,
            char** argv, init_params const& params = init_params())
        {
            hpx::function<int(hpx::program_options::variables_map&)> main_f =
                hpx::bind_back(hpx::detail::init_helper, f);
            return 0 ==
                detail::init_start_impl(
                    HPX_MOVE(main_f), argc, argv, params, false);
        }

        inline bool start(std::function<int()> f, int argc, char** argv,
            init_params const& params = init_params())
        {
            hpx::function<int(hpx::program_options::variables_map&)> main_f =
                hpx::bind(f);
            return 0 ==
                detail::init_start_impl(
                    HPX_MOVE(main_f), argc, argv, params, false);
        }

        inline bool start(std::nullptr_t, int argc, char** argv,
            init_params const& params = init_params())
        {
            hpx::function<int(hpx::program_options::variables_map&)> main_f;
            return 0 ==
                detail::init_start_impl(
                    HPX_MOVE(main_f), argc, argv, params, false);
        }

        HPX_CORE_EXPORT int finalize(error_code& ec = throws);
        HPX_CORE_EXPORT int stop(error_code& ec = throws);
        HPX_CORE_EXPORT int suspend(error_code& ec = throws);
        HPX_CORE_EXPORT int resume(error_code& ec = throws);
    }    // namespace local
}    // namespace hpx
