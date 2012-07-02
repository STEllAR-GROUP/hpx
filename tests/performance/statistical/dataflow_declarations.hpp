//  Copyright (c)      2012 Daniel Kogler
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//this file simply enumerates a multitude of functions used to define the action
//used in the benchmark.  Which function is used is decided based on input args

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/components.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/components/dataflow/dataflow.hpp>
#include <hpx/components/dataflow/dataflow_trigger.hpp>

#include <vector>
#include <string>

using std::vector;
using std::string;
using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::uint64_t;
using hpx::util::high_resolution_timer;
using hpx::cout;
using hpx::flush;

//global dummy variables of several different primitive types
int ivar = 1;
long lvar = 1;
float fvar = 1;
double dvar = 1;

///////////////////////////////////////////////////////////////////////////////
//forward declarations
template <typename Vector, typename Package, typename Result>
void run_empty(uint64_t number);

///////////////////////////////////////////////////////////////////////////////
//empty functions for plain_result_actions
template<typename A1>
A1 empty_thread(){
    A1 dummy = 0;
    return dummy;
}
template<typename A1>
A1 empty_thread(A1 arg1){;
    return arg1;
}
template<typename A1>
A1 empty_thread(A1 arg1, A1 arg2){
    return arg2;
}
template<typename A1>
A1 empty_thread(A1 arg1, A1 arg2, A1 arg3){
    return arg3;
}
template<typename A1>
A1 empty_thread(A1 arg1, A1 arg2, A1 arg3, A1 arg4){
    return arg4;
}

typedef hpx::actions::plain_result_action0<int, empty_thread> empty_actioni0;
HPX_REGISTER_PLAIN_ACTION(empty_actioni0);

typedef hpx::actions::plain_result_action0<long, empty_thread> empty_actionl0;
HPX_REGISTER_PLAIN_ACTION(empty_actionl0);

typedef hpx::actions::plain_result_action0<float, empty_thread> empty_actionf0;
HPX_REGISTER_PLAIN_ACTION(empty_actionf0);

typedef hpx::actions::plain_result_action0<double, empty_thread> empty_actiond0;
HPX_REGISTER_PLAIN_ACTION(empty_actiond0);

typedef hpx::actions::plain_result_action1<int, int, empty_thread> 
    empty_actioni1;
HPX_REGISTER_PLAIN_ACTION(empty_actioni1);

typedef hpx::actions::plain_result_action1<long, long, empty_thread> 
    empty_actionl1;
HPX_REGISTER_PLAIN_ACTION(empty_actionl1);

typedef hpx::actions::plain_result_action1<float, float, empty_thread> 
    empty_actionf1;
HPX_REGISTER_PLAIN_ACTION(empty_actionf1);

typedef hpx::actions::plain_result_action1<double, double, empty_thread> 
    empty_actiond1;
HPX_REGISTER_PLAIN_ACTION(empty_actiond1);

typedef hpx::actions::plain_result_action2<int, int, int, empty_thread> 
    empty_actioni2;
HPX_REGISTER_PLAIN_ACTION(empty_actioni2);

typedef hpx::actions::plain_result_action2<long, long, long, empty_thread> 
    empty_actionl2;
HPX_REGISTER_PLAIN_ACTION(empty_actionl2);

typedef hpx::actions::plain_result_action2<float, float, float, empty_thread> 
    empty_actionf2;
HPX_REGISTER_PLAIN_ACTION(empty_actionf2);

typedef hpx::actions::plain_result_action2<double, double, double, empty_thread> 
    empty_actiond2;
HPX_REGISTER_PLAIN_ACTION(empty_actiond2);

typedef hpx::actions::plain_result_action3<
    int, int, int, int, empty_thread> empty_actioni3;
HPX_REGISTER_PLAIN_ACTION(empty_actioni3);

typedef hpx::actions::plain_result_action3<
    long, long, long, long, empty_thread> empty_actionl3;
HPX_REGISTER_PLAIN_ACTION(empty_actionl3);

typedef hpx::actions::plain_result_action3<
    float, float, float, float, empty_thread> empty_actionf3;
HPX_REGISTER_PLAIN_ACTION(empty_actionf3);

typedef hpx::actions::plain_result_action3<
    double, double, double, double, empty_thread> empty_actiond3;
HPX_REGISTER_PLAIN_ACTION(empty_actiond3);

typedef hpx::actions::plain_result_action4<
    int, int, int, int, int, empty_thread> empty_actioni4;
HPX_REGISTER_PLAIN_ACTION(empty_actioni4);

typedef hpx::actions::plain_result_action4<
    long, long, long, long, long, empty_thread> empty_actionl4;
HPX_REGISTER_PLAIN_ACTION(empty_actionl4);

typedef hpx::actions::plain_result_action4<
    float, float, float, float, float, empty_thread> empty_actionf4;
HPX_REGISTER_PLAIN_ACTION(empty_actionf4);

typedef hpx::actions::plain_result_action4<
    double, double, double, double, double, empty_thread> empty_actiond4;
HPX_REGISTER_PLAIN_ACTION(empty_actiond4);

typedef hpx::lcos::dataflow<empty_actioni0> eflowi0;
typedef hpx::lcos::dataflow<empty_actionl0> eflowl0;
typedef hpx::lcos::dataflow<empty_actionf0> eflowf0;
typedef hpx::lcos::dataflow<empty_actiond0> eflowd0;
typedef hpx::lcos::dataflow<empty_actioni1> eflowi1;
typedef hpx::lcos::dataflow<empty_actionl1> eflowl1;
typedef hpx::lcos::dataflow<empty_actionf1> eflowf1;
typedef hpx::lcos::dataflow<empty_actiond1> eflowd1;
typedef hpx::lcos::dataflow<empty_actioni2> eflowi2;
typedef hpx::lcos::dataflow<empty_actionl2> eflowl2;
typedef hpx::lcos::dataflow<empty_actionf2> eflowf2;
typedef hpx::lcos::dataflow<empty_actiond2> eflowd2;
typedef hpx::lcos::dataflow<empty_actioni3> eflowi3;
typedef hpx::lcos::dataflow<empty_actionl3> eflowl3;
typedef hpx::lcos::dataflow<empty_actionf3> eflowf3;
typedef hpx::lcos::dataflow<empty_actiond3> eflowd3;
typedef hpx::lcos::dataflow<empty_actioni4> eflowi4;
typedef hpx::lcos::dataflow<empty_actionl4> eflowl4;
typedef hpx::lcos::dataflow<empty_actionf4> eflowf4;
typedef hpx::lcos::dataflow<empty_actiond4> eflowd4;

