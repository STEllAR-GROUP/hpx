// Copyright (c) 2006, 2007 Julio M. Merino Vidal
// Copyright (c) 2008 Ilya Sokolov, Boris Schaeling
// Copyright (c) 2009 Boris Schaeling
// Copyright (c) 2010 Felipe Tanus, Boris Schaeling
// Copyright (c) 2011, 2012 Jeff Flinn, Boris Schaeling
// Copyright (c) 2016 Hartmut Kaiser
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PROCESS_POSIX_EXECUTOR_HPP
#define HPX_PROCESS_POSIX_EXECUTOR_HPP

#include <hpx/config.hpp>

#if !defined(HPX_WINDOWS)
#include <hpx/components/process/util/child.hpp>

#include <cstdlib>

#include <sys/types.h>
#include <unistd.h>

namespace hpx { namespace components { namespace process { namespace posix {

struct executor
{
    executor() : exe(nullptr), cmd_line(nullptr), env(nullptr) {}

    struct call_on_fork_setup
    {
        executor &e_;

        call_on_fork_setup(executor &e) : e_(e) {}

        template <class Arg>
        void operator()(const Arg &arg) const
        {
            arg.on_fork_setup(e_);
        }
    };

    struct call_on_fork_error
    {
        executor &e_;

        call_on_fork_error(executor &e) : e_(e) {}

        template <class Arg>
        void operator()(Arg &arg) const
        {
            arg.on_fork_error(e_);
        }
    };

    struct call_on_fork_success
    {
        executor &e_;

        call_on_fork_success(executor &e) : e_(e) {}

        template <class Arg>
        void operator()(Arg &arg) const
        {
            arg.on_fork_success(e_);
        }
    };

    struct call_on_exec_setup
    {
        executor &e_;

        call_on_exec_setup(executor &e) : e_(e) {}

        template <class Arg>
        void operator()(Arg &arg) const
        {
            arg.on_exec_setup(e_);
        }
    };

    struct call_on_exec_error
    {
        executor &e_;

        call_on_exec_error(executor &e) : e_(e) {}

        template <class Arg>
        void operator()(Arg &arg) const
        {
            arg.on_exec_error(e_);
        }
    };

    template <typename ... Ts>
    child operator()(Ts &&... ts)
    {
        int const fork_sequencer[] = {
            (call_on_fork_setup(*this)(ts), 0)..., 0
        };
        (void)fork_sequencer;

        pid_t pid = ::fork();
        if (pid == -1)
        {
            int const error_sequencer[] = {
                (call_on_fork_error(*this)(ts), 0)..., 0
            };
            (void)error_sequencer;
        }
        else if (pid == 0)
        {
            int const setup_sequencer[] = {
                (call_on_exec_setup(*this)(ts), 0)..., 0
            };
            (void)setup_sequencer;
            ::execve(exe, cmd_line, env);

            int const error_sequencer[] = {
                (call_on_exec_error(*this)(ts), 0)..., 0
            };
            (void)error_sequencer;
            _exit(EXIT_FAILURE);
        }

        int const success_sequencer[] = {
            (call_on_fork_success(*this)(ts), 0)..., 0
        };
        (void)success_sequencer;

        return child(pid);
    }

    const char *exe;
    char **cmd_line;
    char **env;
};

}}}}

#endif
#endif
