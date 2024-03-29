// Copyright (c) 2006, 2007 Julio M. Merino Vidal
// Copyright (c) 2008 Ilya Sokolov, Boris Schaeling
// Copyright (c) 2009 Boris Schaeling
// Copyright (c) 2010 Felipe Tanus, Boris Schaeling
// Copyright (c) 2011, 2012 Jeff Flinn, Boris Schaeling
// Copyright (c) 2016-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if !defined(HPX_WINDOWS)
#include <hpx/components/process/util/child.hpp>

#include <cstdlib>

#include <sys/types.h>
#include <unistd.h>

namespace hpx { namespace components { namespace process { namespace posix {

    struct executor
    {
        executor()
          : exe(nullptr)
          , cmd_line(nullptr)
          , env(nullptr)
        {
        }

        struct call_on_fork_setup
        {
            executor& e_;

            explicit call_on_fork_setup(executor& e)
              : e_(e)
            {
            }

            template <class Arg>
            void operator()(const Arg& arg) const
            {
                arg.on_fork_setup(e_);
            }
        };

        struct call_on_fork_error
        {
            executor& e_;

            explicit call_on_fork_error(executor& e)
              : e_(e)
            {
            }

            template <class Arg>
            void operator()(Arg& arg) const
            {
                arg.on_fork_error(e_);
            }
        };

        struct call_on_fork_success
        {
            executor& e_;

            explicit call_on_fork_success(executor& e)
              : e_(e)
            {
            }

            template <class Arg>
            void operator()(Arg& arg) const
            {
                arg.on_fork_success(e_);
            }
        };

        struct call_on_exec_setup
        {
            executor& e_;

            explicit call_on_exec_setup(executor& e)
              : e_(e)
            {
            }

            template <class Arg>
            void operator()(Arg& arg) const
            {
                arg.on_exec_setup(e_);
            }
        };

        struct call_on_exec_error
        {
            executor& e_;

            explicit call_on_exec_error(executor& e)
              : e_(e)
            {
            }

            template <class Arg>
            void operator()(Arg& arg) const
            {
                arg.on_exec_error(e_);
            }
        };

        template <typename... Ts>
        child operator()(Ts&&... ts)
        {
            (call_on_fork_setup(*this)(ts), ...);

            pid_t pid = ::fork();
            if (pid == -1)
            {
                (call_on_fork_error(*this)(ts), ...);
            }
            else if (pid == 0)
            {
                (call_on_exec_setup(*this)(ts), ...);
                ::execve(exe, cmd_line, env);
                (call_on_exec_error(*this)(ts), ...);

                _exit(EXIT_FAILURE);
            }

            (call_on_fork_success(*this)(ts), ...);

            return child(pid);
        }

        const char* exe;
        char** cmd_line;
        char** env;
    };

}}}}    // namespace hpx::components::process::posix

#endif
