// Boost Logging library
//
// Author: John Torjo, www.torjo.com
//
// Copyright (C) 2007 John Torjo (see www.torjo.com for email)
//
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)
//
// See http://www.boost.org for updates, documentation, and revision history.
// See http://www.torjo.com/log2/ for more details

namespace hpx { namespace util { namespace logging {

/**
@page common_scenarios Usage Scenarios (together with code)

- @ref common_scenarios_1
- @ref common_scenarios_5
- @ref common_scenarios_6

- @ref common_your_scenario

- @ref scenario_multiple_files
    - @ref scenario_multiple_files_program
    - @ref scenario_multiple_files_log_h
    - @ref scenario_multiple_files_log_cpp
    - @ref scenario_multiple_files_main

\n\n\n
@copydoc common_usage_steps_fd



\n\n\n
@section common_scenarios_1 Scenario 1, Common usage: Multiple levels, One logging class, Multiple destinations.

Scenario 1 should be the most common.

@copydoc mul_levels_one_logger

@ref scenarios_code_mom "Click to see the code"
\n\n\n



@section common_scenarios_5 Scenario 2: No levels, One Logger, One Filter

@copydoc one_loger_one_filter

@ref scenarios_code_noo "Click to see the code"
\n\n\n



@section common_scenarios_6 Scenario 3: Fastest: Multiple Loggers, One Filter, Not using Formatters/Destinations, Not using <<

@copydoc fastest_no_ostr_like

@ref scenarios_code_mon "Click to see the code"
\n\n\n




@section common_your_scenario Your Scenario : Find out logger and filter, based on your application's needs

@copydoc your_scenario

@ref common_your_scenario_code "Click to see the code"
\n\n\n





*/

}}}
