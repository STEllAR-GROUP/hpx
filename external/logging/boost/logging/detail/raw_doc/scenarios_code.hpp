namespace boost { namespace logging {

/** 
@page scenarios_code Code for the common scenarios

- @ref scenarios_code_mom 
- @ref scenarios_code_noo 
- @ref scenarios_code_mon 

- @ref common_your_scenario_code

\n\n\n
@section scenarios_code_mom Scenario 1, Common usage: Multiple levels, One logging class, Multiple destinations.

@include mul_levels_one_logger.cpp
\n\n\n



@section scenarios_code_noo Scenario 2: No levels, One Logger, One Filter

@include one_loger_one_filter.cpp
\n\n\n



@section scenarios_code_mon Scenario 3: Fastest: Multiple Loggers, One Filter, Not using Formatters/Destinations, Not using <<

@include fastest_no_ostr_like.cpp
\n\n\n



@section common_your_scenario_code Your Scenario : Find out logger and filter, based on your application's needs

@include your_scenario.cpp
\n\n\n



@section common_your_mul_logger_one_filter Multiple loggers, one filter

@include mul_loggers_one_filter.cpp
\n\n\n


*/

}}
