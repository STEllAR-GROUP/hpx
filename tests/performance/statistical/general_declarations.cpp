//  Copyright (c)      2012 Daniel Kogler
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//this file simply enumerates a multitude of functions used to define the action
//used in the benchmark.  Which function is used is decided based on input args

#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/components.hpp>
#include <hpx/util/high_resolution_timer.hpp>

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

template <typename Vector, typename Package>
void run_void(uint64_t number);

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

//empty functions for regular old plain_actions
void void_thread(){
}
template<typename A1>
void void_thread(A1 arg1){
}
template<typename A1>
void void_thread(A1 arg1, A1 arg2){
}
template<typename A1>
void void_thread(A1 arg1, A1 arg2, A1 arg3){
}
template<typename A1>
void void_thread(A1 arg1, A1 arg2, A1 arg3, A1 arg4){
}

typedef hpx::actions::plain_result_action0<int, empty_thread> empty_actioni0;
typedef hpx::lcos::packaged_action<empty_actioni0> empty_packagei0;
HPX_REGISTER_PLAIN_ACTION(empty_actioni0);

typedef hpx::actions::plain_result_action0<long, empty_thread> empty_actionl0;
typedef hpx::lcos::packaged_action<empty_actionl0> empty_packagel0;
HPX_REGISTER_PLAIN_ACTION(empty_actionl0);

typedef hpx::actions::plain_result_action0<float, empty_thread> empty_actionf0;
typedef hpx::lcos::packaged_action<empty_actionf0> empty_packagef0;
HPX_REGISTER_PLAIN_ACTION(empty_actionf0);

typedef hpx::actions::plain_result_action0<double, empty_thread> empty_actiond0;
typedef hpx::lcos::packaged_action<empty_actiond0> empty_packaged0;
HPX_REGISTER_PLAIN_ACTION(empty_actiond0);

typedef hpx::actions::plain_result_action1<int, int, empty_thread> 
    empty_actioni1;
typedef hpx::lcos::packaged_action<empty_actioni1> empty_packagei1;
HPX_REGISTER_PLAIN_ACTION(empty_actioni1);

typedef hpx::actions::plain_result_action1<long, long, empty_thread> 
    empty_actionl1;
typedef hpx::lcos::packaged_action<empty_actionl1> empty_packagel1;
HPX_REGISTER_PLAIN_ACTION(empty_actionl1);

typedef hpx::actions::plain_result_action1<float, float, empty_thread> 
    empty_actionf1;
typedef hpx::lcos::packaged_action<empty_actionf1> empty_packagef1;
HPX_REGISTER_PLAIN_ACTION(empty_actionf1);

typedef hpx::actions::plain_result_action1<double, double, empty_thread> 
    empty_actiond1;
typedef hpx::lcos::packaged_action<empty_actiond1> empty_packaged1;
HPX_REGISTER_PLAIN_ACTION(empty_actiond1);

typedef hpx::actions::plain_result_action2<int, int, int, empty_thread> 
    empty_actioni2;
typedef hpx::lcos::packaged_action<empty_actioni2> empty_packagei2;
HPX_REGISTER_PLAIN_ACTION(empty_actioni2);

typedef hpx::actions::plain_result_action2<long, long, long, empty_thread> 
    empty_actionl2;
typedef hpx::lcos::packaged_action<empty_actionl2> empty_packagel2;
HPX_REGISTER_PLAIN_ACTION(empty_actionl2);

typedef hpx::actions::plain_result_action2<float, float, float, empty_thread> 
    empty_actionf2;
typedef hpx::lcos::packaged_action<empty_actionf2> empty_packagef2;
HPX_REGISTER_PLAIN_ACTION(empty_actionf2);

typedef hpx::actions::plain_result_action2<double, double, double, empty_thread> 
    empty_actiond2;
typedef hpx::lcos::packaged_action<empty_actiond2> empty_packaged2;
HPX_REGISTER_PLAIN_ACTION(empty_actiond2);

typedef hpx::actions::plain_result_action3<
    int, int, int, int, empty_thread> empty_actioni3;
typedef hpx::lcos::packaged_action<empty_actioni3> empty_packagei3;
HPX_REGISTER_PLAIN_ACTION(empty_actioni3);

typedef hpx::actions::plain_result_action3<
    long, long, long, long, empty_thread> empty_actionl3;
typedef hpx::lcos::packaged_action<empty_actionl3> empty_packagel3;
HPX_REGISTER_PLAIN_ACTION(empty_actionl3);

typedef hpx::actions::plain_result_action3<
    float, float, float, float, empty_thread> empty_actionf3;
typedef hpx::lcos::packaged_action<empty_actionf3> empty_packagef3;
HPX_REGISTER_PLAIN_ACTION(empty_actionf3);

typedef hpx::actions::plain_result_action3<
    double, double, double, double, empty_thread> empty_actiond3;
typedef hpx::lcos::packaged_action<empty_actiond3> empty_packaged3;
HPX_REGISTER_PLAIN_ACTION(empty_actiond3);

typedef hpx::actions::plain_result_action4<
    int, int, int, int, int, empty_thread> empty_actioni4;
typedef hpx::lcos::packaged_action<empty_actioni4> empty_packagei4;
HPX_REGISTER_PLAIN_ACTION(empty_actioni4);

typedef hpx::actions::plain_result_action4<
    long, long, long, long, long, empty_thread> empty_actionl4;
typedef hpx::lcos::packaged_action<empty_actionl4> empty_packagel4;
HPX_REGISTER_PLAIN_ACTION(empty_actionl4);

typedef hpx::actions::plain_result_action4<
    float, float, float, float, float, empty_thread> empty_actionf4;
typedef hpx::lcos::packaged_action<empty_actionf4> empty_packagef4;
HPX_REGISTER_PLAIN_ACTION(empty_actionf4);

typedef hpx::actions::plain_result_action4<
    double, double, double, double, double, empty_thread> empty_actiond4;
typedef hpx::lcos::packaged_action<empty_actiond4> empty_packaged4;
HPX_REGISTER_PLAIN_ACTION(empty_actiond4);


typedef hpx::actions::plain_action0<void_thread> void_action0;
typedef hpx::lcos::packaged_action<void_action0> void_package0;
HPX_REGISTER_PLAIN_ACTION(void_action0);

typedef hpx::actions::plain_action1<int, void_thread> 
    void_actioni1;
typedef hpx::lcos::packaged_action<void_actioni1> void_packagei1;
HPX_REGISTER_PLAIN_ACTION(void_actioni1);

typedef hpx::actions::plain_action1<long, void_thread> 
    void_actionl1;
typedef hpx::lcos::packaged_action<void_actionl1> void_packagel1;
HPX_REGISTER_PLAIN_ACTION(void_actionl1);

typedef hpx::actions::plain_action1<float, void_thread> 
    void_actionf1;
typedef hpx::lcos::packaged_action<void_actionf1> void_packagef1;
HPX_REGISTER_PLAIN_ACTION(void_actionf1);

typedef hpx::actions::plain_action1<double, void_thread> 
    void_actiond1;
typedef hpx::lcos::packaged_action<void_actiond1> void_packaged1;
HPX_REGISTER_PLAIN_ACTION(void_actiond1);

typedef hpx::actions::plain_action2<int, int, void_thread> 
    void_actioni2;
typedef hpx::lcos::packaged_action<void_actioni2> void_packagei2;
HPX_REGISTER_PLAIN_ACTION(void_actioni2);

typedef hpx::actions::plain_action2<long, long, void_thread> 
    void_actionl2;
typedef hpx::lcos::packaged_action<void_actionl2> void_packagel2;
HPX_REGISTER_PLAIN_ACTION(void_actionl2);

typedef hpx::actions::plain_action2<float, float, void_thread> 
    void_actionf2;
typedef hpx::lcos::packaged_action<void_actionf2> void_packagef2;
HPX_REGISTER_PLAIN_ACTION(void_actionf2);

typedef hpx::actions::plain_action2<double, double, void_thread> 
    void_actiond2;
typedef hpx::lcos::packaged_action<void_actiond2> void_packaged2;
HPX_REGISTER_PLAIN_ACTION(void_actiond2);

typedef hpx::actions::plain_action3<
    int, int, int, void_thread> void_actioni3;
typedef hpx::lcos::packaged_action<void_actioni3> void_packagei3;
HPX_REGISTER_PLAIN_ACTION(void_actioni3);

typedef hpx::actions::plain_action3<
    long, long, long, void_thread> void_actionl3;
typedef hpx::lcos::packaged_action<void_actionl3> void_packagel3;
HPX_REGISTER_PLAIN_ACTION(void_actionl3);

typedef hpx::actions::plain_action3<
    float, float, float, void_thread> void_actionf3;
typedef hpx::lcos::packaged_action<void_actionf3> void_packagef3;
HPX_REGISTER_PLAIN_ACTION(void_actionf3);

typedef hpx::actions::plain_action3<
    double, double, double, void_thread> void_actiond3;
typedef hpx::lcos::packaged_action<void_actiond3> void_packaged3;
HPX_REGISTER_PLAIN_ACTION(void_actiond3);

typedef hpx::actions::plain_action4<
    int, int, int, int, void_thread> void_actioni4;
typedef hpx::lcos::packaged_action<void_actioni4> void_packagei4;
HPX_REGISTER_PLAIN_ACTION(void_actioni4);

typedef hpx::actions::plain_action4<
    long, long, long, long, void_thread> void_actionl4;
typedef hpx::lcos::packaged_action<void_actionl4> void_packagel4;
HPX_REGISTER_PLAIN_ACTION(void_actionl4);

typedef hpx::actions::plain_action4<
    float, float, float, float, void_thread> void_actionf4;
typedef hpx::lcos::packaged_action<void_actionf4> void_packagef4;
HPX_REGISTER_PLAIN_ACTION(void_actionf4);

typedef hpx::actions::plain_action4<
    double, double, double, double, void_thread> void_actiond4;
typedef hpx::lcos::packaged_action<void_actiond4> void_packaged4;
HPX_REGISTER_PLAIN_ACTION(void_actiond4);

