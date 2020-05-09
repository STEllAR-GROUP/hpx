// Copyright (c) 2006, 2007 Julio M. Merino Vidal
// Copyright (c) 2008 Ilya Sokolov, Boris Schaeling
// Copyright (c) 2009 Boris Schaeling
// Copyright (c) 2010 Felipe Tanus, Boris Schaeling
// Copyright (c) 2011, 2012 Jeff Flinn, Boris Schaeling
// Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

// #include <hpx/components/process/util/windows/initializers/bind_stderr.hpp>
// #include <hpx/components/process/util/windows/initializers/bind_stdin.hpp>
// #include <hpx/components/process/util/windows/initializers/bind_stdout.hpp>
#include <hpx/components/process/util/windows/initializers/close_stderr.hpp>
#include <hpx/components/process/util/windows/initializers/close_stdin.hpp>
#include <hpx/components/process/util/windows/initializers/close_stdout.hpp>
#include <hpx/components/process/util/windows/initializers/hide_console.hpp>
#include <hpx/components/process/util/windows/initializers/inherit_env.hpp>
#include <hpx/components/process/util/windows/initializers/on_CreateProcess_error.hpp>
#include <hpx/components/process/util/windows/initializers/on_CreateProcess_setup.hpp>
#include <hpx/components/process/util/windows/initializers/on_CreateProcess_success.hpp>
#include <hpx/components/process/util/windows/initializers/run_exe.hpp>
#include <hpx/components/process/util/windows/initializers/set_args.hpp>
#include <hpx/components/process/util/windows/initializers/set_cmd_line.hpp>
#include <hpx/components/process/util/windows/initializers/set_env.hpp>
#include <hpx/components/process/util/windows/initializers/show_window.hpp>
#include <hpx/components/process/util/windows/initializers/start_in_dir.hpp>
#include <hpx/components/process/util/windows/initializers/throw_on_error.hpp>
#include <hpx/components/process/util/windows/initializers/wait_on_latch.hpp>

