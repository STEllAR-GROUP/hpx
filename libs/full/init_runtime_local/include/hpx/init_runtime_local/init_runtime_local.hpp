//  Copyright (c)      2021 ETH Zurich
//  Copyright (c)      2018 Mikael Simberg
//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c) 2010-2011 Phillip LeBlanc, Dylan Stark
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx_init.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/functional/bind.hpp>
#include <hpx/functional/bind_back.hpp>
#include <hpx/functional/function.hpp>
#include <hpx/modules/program_options.hpp>
#include <hpx/resource_partitioner/partitioner.hpp>
#include <hpx/runtime_local/runtime_local.hpp>
#include <hpx/runtime_local/shutdown_function.hpp>
#include <hpx/runtime_local/startup_function.hpp>

#include <csignal>
#include <cstddef>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#if defined(__FreeBSD__)
extern HPX_EXPORT char** freebsd_environ;
extern char** environ;
#endif

namespace hpx {
    namespace detail {
        HPX_EXPORT int init_helper(hpx::program_options::variables_map&,
            util::function_nonser<int(int, char**)> const&);
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
            HPX_MAYBE_UNUSED static int dummy_argc = 1;
            HPX_MAYBE_UNUSED static char app_name[] = HPX_APPLICATION_STRING;
            static char* default_argv[2] = {app_name, nullptr};
            HPX_MAYBE_UNUSED static char** dummy_argv = default_argv;
            // HPX_APPLICATION_STRING is specific to an application and therefore
            // cannot be in the source file
            HPX_MAYBE_UNUSED static const hpx::program_options::
                options_description default_desc =
                    hpx::program_options::options_description(
                        "Usage: " HPX_APPLICATION_STRING " [options]");

            // Utilities to init the thread_pools of the resource partitioner
            using rp_callback_type =
                hpx::util::function_nonser<void(hpx::resource::partitioner&,
                    hpx::program_options::variables_map const&)>;
        }    // namespace detail

        struct init_params
        {
            std::reference_wrapper<
                hpx::program_options::options_description const>
                desc_cmdline = detail::default_desc;
            std::vector<std::string> cfg;
            mutable startup_function_type startup;
            mutable shutdown_function_type shutdown;
            hpx::resource::partitioner_mode rp_mode =
                ::hpx::resource::mode_default;
            hpx::local::detail::rp_callback_type rp_callback;
        };

        namespace detail {
            HPX_EXPORT int run_or_start(
                util::function_nonser<int(
                    hpx::program_options::variables_map& vm)> const& f,
                int argc, char** argv, init_params const& params,
                bool blocking);
        }    // namespace detail

        inline int init(hpx::util::function_nonser<int(
                            hpx::program_options::variables_map&)> const& f,
            int argc, char** argv, init_params const& params = init_params())
        {
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
            return detail::run_or_start(f, argc, argv, params, true);
        }

        inline int init(util::function_nonser<int(int, char**)> const& f,
            int argc, char** argv, init_params const& params = init_params())
        {
            hpx::util::function_nonser<int(
                hpx::program_options::variables_map&)>
                main_f = hpx::util::bind_back(hpx::detail::init_helper, f);

            if (argc == 0 || argv == nullptr)
            {
                return init(
                    main_f, detail::dummy_argc, detail::dummy_argv, params);
            }

            return init(main_f, argc, argv, params);
        }

        inline int init(util::function_nonser<int()> const& f, int argc,
            char** argv, init_params const& params = init_params())
        {
            hpx::util::function_nonser<int(
                hpx::program_options::variables_map&)>
                main_f = hpx::util::bind(f);

            if (argc == 0 || argv == nullptr)
            {
                return init(
                    main_f, detail::dummy_argc, detail::dummy_argv, params);
            }

            return init(main_f, argc, argv, params);
        }

        inline int init(std::nullptr_t, int argc, char** argv,
            init_params const& params = init_params())
        {
            hpx::util::function_nonser<int(
                hpx::program_options::variables_map&)>
                main_f;

            if (argc == 0 || argv == nullptr)
            {
                return init(
                    main_f, detail::dummy_argc, detail::dummy_argv, params);
            }

            return init(main_f, argc, argv, params);
        }

        HPX_EXPORT int finalize(error_code& ec = throws);
        HPX_EXPORT int stop(error_code& ec = throws);
        HPX_EXPORT int suspend(error_code& ec = throws);
        HPX_EXPORT int resume(error_code& ec = throws);
    }    // namespace local
}    // namespace hpx
