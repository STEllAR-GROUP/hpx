//  Copyright (c)      2012 Daniel Kogler
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "dataflow_declarations.hpp"
#include "statstd.hpp"

///////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////

//this runs a series of tests for a plain_result_action
template <typename Vector, typename Dataflow, typename Action, typename T>
void run_empty(uint64_t);
template <typename Vector, typename Dataflow, typename Action, typename T>
void run_empty(uint64_t, T);
template <typename Vector, typename Dataflow, typename Action, typename T>
void run_empty(uint64_t, T, T);
template <typename Vector, typename Dataflow, typename Action, typename T>
void run_empty(uint64_t, T, T, T);
template <typename Vector, typename Dataflow, typename Action, typename T>
void run_empty(uint64_t, T, T, T, T);

///////////////////////////////////////////////////////////////////////////////

//decide which action type to use
void decide_action_type(uint64_t num, int type, int argc);

//parse the argument types
void parse_arg(string atype, uint64_t num, int argc){
    int type;
    if(atype.compare("int") == 0) type = 0;
    else if(atype.compare("long") == 0) type = 1;
    else if(atype.compare("float") == 0) type = 2;
    else if(atype.compare("double") == 0) type = 3;
    else{
        std::cerr<<"Error parsing input, see help entry for a list of ";
        std::cerr<<"available types\n";
        return;
    }
    decide_action_type(num, type, argc);
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm){
    uint64_t num = vm["number-created"].as<uint64_t>();
    string atype = vm["arg-type"].as<string>();
    int c = vm["argc"].as<int>();
    csv = (vm.count("csv") ? true : false);
    parse_arg(atype, num, c);
    return hpx::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[]){
    // Configure application-specific options.
    options_description
        desc_commandline("usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ("number-created,N",
            boost::program_options::value<uint64_t>()
                ->default_value(50000),
            "number of dataflow objects created and tested")
        ("arg-type,A",
            boost::program_options::value<string>()
                ->default_value("int"),
            "specifies the argument type of the action, as well as the result "
            "type of the value returned, to be either int, long, "
            "float, or double.")
        ("argc,C",
            boost::program_options::value<int>()
                ->default_value(1),
            "the number of arguments each action takes")
        ("csv",
            "output results as csv "
            "(format:count,mean,accurate mean,variance,min,max)");

    // Initialize and run HPX
    return hpx::init(desc_commandline, argc, argv);
}

///////////////////////////////////////////////////////////////////////////////
//Below is purely tedium

//decide which action type to use
void decide_action_type(uint64_t num, int type, int argc){
    if(type == 0){
        switch(argc){
            case 0: run_empty<vector<eflowi0*>, eflowi0, 
                empty_actioni0, int> (num); break;
            case 1: run_empty<vector<eflowi1*>, eflowi1, 
                empty_actioni1, int> (num, ivar); break;
/*            case 2: run_empty<vector<eflowi2*>, eflowi2, 
                empty_actioni2, int> (num, ivar, ivar); break;
            case 3: run_empty<vector<eflowi3*>, eflowi3, 
                empty_actioni3, int> (num, ivar, ivar, ivar); break;
            default: run_empty<vector<eflowi4*>, eflowi4, 
                empty_actioni4, int> (num, ivar, ivar, ivar, ivar); break;*/
        }
    }
    else if(type == 1){
        switch(argc){
            case 0: run_empty<vector<eflowl0*>, eflowl0, 
                empty_actionl0, long> (num); break;
            case 1: run_empty<vector<eflowl1*>, eflowl1, 
                empty_actionl1, long> (num, lvar); break;
/*            case 2: run_empty<vector<eflowl2*>, eflowl2, 
                empty_actionl2, long> (num, lvar, lvar); break;
            case 3: run_empty<vector<eflowl3*>, eflowl3, 
                empty_actionl3, long> (num, lvar, lvar, lvar); break;
            default: run_empty<vector<eflowl4*>, eflowl4, 
                empty_actionl4, long> (num, lvar, lvar, lvar, lvar); break;*/
        }
    }
    else if(type == 2){
        switch(argc){
            case 0: run_empty<vector<eflowf0*>, eflowf0, 
                empty_actionf0, float> (num); break;
            case 1: run_empty<vector<eflowf1*>, eflowf1, 
                empty_actionf1, float> (num, fvar); break;
/*            case 2: run_empty<vector<eflowf2*>, eflowf2, 
                empty_actionf2, float> (num, fvar, fvar); break;
            case 3: run_empty<vector<eflowf3*>, eflowf3, 
                empty_actionf3, float> (num, fvar, fvar, fvar); break;
            default: run_empty<vector<eflowf4*>, eflowf4, 
                empty_actionf4, float> (num, fvar, fvar, fvar, fvar); break;*/
        }
    }
    else{
        switch(argc){
            case 0: run_empty<vector<eflowd0*>, eflowd0, 
                empty_actiond0, double> (num); break;
            case 1: run_empty<vector<eflowd1*>, eflowd1, 
                empty_actiond1, double> (num, dvar); break;
 /*           case 2: run_empty<vector<eflowd2*>, eflowd2, 
                empty_actiond2, double> (num, dvar, dvar); break;
            case 3: run_empty<vector<eflowd3*>, eflowd3, 
                empty_actiond3, double> (num, dvar, dvar, dvar); break;
            default: run_empty<vector<eflowd4*>, eflowd4, 
                empty_actiond4, double> (num, dvar, dvar, dvar, dvar); break;*/
        }
    }
}

//this runs a series of tests for dataflow objects
template <typename Vector, typename Dataflow, typename Action, typename T>
void run_empty(uint64_t num){
    uint64_t i;
    double ot = timer_overhead(num);
    double mean1, mean2, mean3;
    vector<double> time;
    Vector lcos;

    string message = "Measuring time required to create dataflow objects:";
    lcos.reserve(2*num);

    hpx::naming::id_type lid = hpx::find_here();
    //the first measurement is of the amount of time it takes to generate
    //the specified number of dataflow_objects
    //the second test attempts to measure trigger times
    //the method will probably need to change for accurate results however

    high_resolution_timer t;
    for(i = 0; i < num; ++i)
        lcos.push_back(new Dataflow(lid));
    mean1 = t.elapsed()/num;

    time.reserve(num);
    for(i = 0; i < num; ++i){
        high_resolution_timer t1;
        lcos.push_back(new Dataflow(lid));
        time.push_back(t1.elapsed());
    }
    printout(time, ot, mean1, message);
    time.clear();

    vector<hpx::lcos::future<T> > futures;
    futures.reserve(2*num);
    t.restart();
    for(i = 0; i < num; i++)
        futures.push_back(lcos[i]->get_future());
    mean2 = t.elapsed()/num;
    message = "Measuring time required to return futures from dataflow lcos:";
    time.reserve(num);
    futures.reserve(num);
    for(i = num; i < num+num; i++){
        high_resolution_timer t1;
        futures.push_back(lcos[i]->get_future());
        time.push_back(t1.elapsed());
    }
    printout(time, ot, mean2, message);
    lcos.clear();

    t.restart();
    for(i = 0; i < num; i++){
        futures[i].get();
    }
    mean3 = t.elapsed()/num;

    time.clear();
    time.reserve(num);
    message = "Measuring time to get results from futures returned from dataflow lcos:";
    for(i = num; i < num+num; i++){
        high_resolution_timer t1;
        futures[i].get();
        time.push_back(t1.elapsed());
    }
    printout(time, ot, mean3, message);
}
template <typename Vector, typename Dataflow, typename Action, typename T>
void run_empty(uint64_t num, T a1){
    uint64_t i;
    double ot = timer_overhead(num);
    double mean1, mean2, mean3;
    vector<double> time;
    Vector lcos;

    string message = "Measuring time required to create dataflow objects:";
    lcos.reserve(num);

    hpx::naming::id_type lid = hpx::find_here();
    //the first measurement is of the amount of time it takes to generate
    //the specified number of dataflow_objects
    //the second test attempts to measure trigger times
    //the method will probably need to change for accurate results however

    high_resolution_timer t;
    lcos.push_back(new Dataflow(lid, a1));
    for(i = 1; i < num; ++i)
        lcos.push_back(new Dataflow(lid, *lcos[i-1]));
    mean1 = t.elapsed()/num;

    t.restart();
    lcos[num-1]->get_future().get();
    mean2 = t.elapsed()/num;

    printf("Mean time for completing each action \n(when requesting the result of "
        "the final action to be executed): %f ns\n\n", mean2*1e9);

    lcos.clear();
    lcos.reserve(2*num);
    t.restart();
    lcos.push_back(new Dataflow(lid, a1));
    for(i = 1; i < num; ++i)
        lcos.push_back(new Dataflow(lid, *lcos[i-1]));
    mean1 = t.elapsed()/num;

    time.reserve(num);
    for(i = num; i < num+num; ++i){
        high_resolution_timer t1;
        lcos.push_back(new Dataflow(lid, *lcos[i-1]));
        time.push_back(t1.elapsed());
    }
    printout(time, ot, mean1, message);
    time.clear();

    vector<hpx::lcos::future<T> > futures;
    futures.reserve(2*num);
    t.restart();
    for(i = 0; i < num; i++)
        futures.push_back(lcos[i]->get_future());
    mean2 = t.elapsed()/num;
    message = "Measuring time required to return futures from dataflow lcos:";
    time.reserve(num);
    futures.reserve(num);
    for(i = num; i < num+num; i++){
        high_resolution_timer t1;
        futures.push_back(lcos[i]->get_future());
        time.push_back(t1.elapsed());
    }
    printout(time, ot, mean2, message);
    lcos.clear();

    t.restart();
    for(i = 0; i < num; i++){
        futures[i].get();
    }
    mean3 = t.elapsed()/num;

    time.clear();
    time.reserve(num);
    message = "Measuring time to get results from futures returned from dataflow lcos:";
    for(i = num; i < num+num; i++){
        high_resolution_timer t1;
        futures[i].get();
        time.push_back(t1.elapsed());
    }
    printout(time, ot, mean3, message);
}
/*
template <typename Vector, typename Dataflow, typename Action, typename T>
void run_empty(uint64_t num, T a1, T a2){
    uint64_t i;
    double ot = timer_overhead(num);
    double mean1, mean2, mean3;
    vector<double> time;
    Vector lcos;

    string message = "Measuring time required to create dataflow objects:";
    lcos.reserve(num);

    hpx::naming::id_type lid = hpx::find_here();
    //the first measurement is of the amount of time it takes to generate
    //the specified number of dataflow_objects
    //the second test attempts to measure trigger times
    //the method will probably need to change for accurate results however

    high_resolution_timer t;
    lcos.push_back(new Dataflow(lid, a1, a2));
    for(i = 1; i < num; ++i)
        lcos.push_back(new Dataflow(lid, *lcos[i-1], a2));
    mean1 = t.elapsed()/num;

    t.restart();
    lcos[0]->get_future().get();
    mean2 = t.elapsed()/num;

    printf("Mean time for completing each action \n(when requesting the result of "
        "the final action to be executed): %f ns\n\n", mean2*1e9);

    lcos.push_back(new Dataflow(lid, a1, a2));
    for(i = 1; i < num; ++i)
        lcos.push_back(new Dataflow(lid, *lcos[i-1], a2));

    vector<hpx::lcos::future<T> > futures;
    futures.reserve(num);
    t.restart();
    for(i = num; i > 0; i--)
        futures.push_back(lcos[i-1]->get_future());
    mean2 = t.elapsed()/num;

    t.restart();
    for(i = num; i > 0; i--){
        futures[i-1].get();
    }
    mean3 = t.elapsed()/num;
    futures.clear();

    lcos.clear();
    time.reserve(num);
    lcos.reserve(num);
    t.restart();
    lcos.push_back(new Dataflow(lid, a1, a2));
    time.push_back(t.elapsed());
    for(i = 1; i < num; ++i){
        high_resolution_timer t1;
        lcos.push_back(new Dataflow(lid, *lcos[i-1], a2));
        time.push_back(t1.elapsed());
    }
    printout(time, ot, mean1, message);

    message = "Measuring time required to return futures from dataflow lcos:";
    time.clear();
    time.reserve(num);
    futures.reserve(num);
    for(i = num; i > 0; i--){
        high_resolution_timer t1;
        futures.push_back(lcos[i-1]->get_future());
        time.push_back(t1.elapsed());
    }
    printout(time, ot, mean2, message);

    time.clear();
    time.reserve(num);
    message = "Measuring time to get results from futures returned from dataflow lcos:";
    for(i = num; i > 0; i--){
        high_resolution_timer t1;
        futures[i-1].get();
        time.push_back(t1.elapsed());
    }
    printout(time, ot, mean3, message);
}
template <typename Vector, typename Dataflow, typename Action, typename T>
void run_empty(uint64_t num, T a1, T a2, T a3){
    uint64_t i;
    double ot = timer_overhead(num);
    double mean1, mean2, mean3;
    vector<double> time;
    Vector lcos;

    string message = "Measuring time required to create dataflow objects:";
    lcos.reserve(num);

    hpx::naming::id_type lid = hpx::find_here();
    //the first measurement is of the amount of time it takes to generate
    //the specified number of dataflow_objects
    //the second test attempts to measure trigger times
    //the method will probably need to change for accurate results however

    high_resolution_timer t;
    lcos.push_back(new Dataflow(lid, a1, a2, a3));
    for(i = 1; i < num; ++i)
        lcos.push_back(new Dataflow(lid, *lcos[i-1], a2, a3));
    mean1 = t.elapsed()/num;

    t.restart();
    lcos[0]->get_future().get();
    mean2 = t.elapsed()/num;

    printf("Mean time for completing each action \n(when requesting the result of "
        "the final action to be executed): %f ns\n\n", mean2*1e9);

    lcos.push_back(new Dataflow(lid, a1, a2, a3));
    for(i = 1; i < num; ++i)
        lcos.push_back(new Dataflow(lid, *lcos[i-1], a2, a3));

    vector<hpx::lcos::future<T> > futures;
    futures.reserve(num);
    t.restart();
    for(i = num; i > 0; i--)
        futures.push_back(lcos[i-1]->get_future());
    mean2 = t.elapsed()/num;

    t.restart();
    for(i = num; i > 0; i--){
        futures[i-1].get();
    }
    mean3 = t.elapsed()/num;
    futures.clear();

    lcos.clear();
    time.reserve(num);
    lcos.reserve(num);
    t.restart();
    lcos.push_back(new Dataflow(lid, a1, a2, a3));
    time.push_back(t.elapsed());
    for(i = 1; i < num; ++i){
        high_resolution_timer t1;
        lcos.push_back(new Dataflow(lid, *lcos[i-1], a2, a3));
        time.push_back(t1.elapsed());
    }
    printout(time, ot, mean1, message);

    message = "Measuring time required to return futures from dataflow lcos:";
    time.clear();
    time.reserve(num);
    futures.reserve(num);
    for(i = num; i > 0; i--){
        high_resolution_timer t1;
        futures.push_back(lcos[i-1]->get_future());
        time.push_back(t1.elapsed());
    }
    printout(time, ot, mean2, message);

    time.clear();
    time.reserve(num);
    message = "Measuring time to get results from futures returned from dataflow lcos:";
    for(i = num; i > 0; i--){
        high_resolution_timer t1;
        futures[i-1].get();
        time.push_back(t1.elapsed());
    }
    printout(time, ot, mean3, message);
}
template <typename Vector, typename Dataflow, typename Action, typename T>
void run_empty(uint64_t num, T a1, T a2, T a3, T a4){
    uint64_t i;
    double ot = timer_overhead(num);
    double mean1, mean2, mean3;
    vector<double> time;
    Vector lcos;

    string message = "Measuring time required to create dataflow objects:";
    lcos.reserve(num);

    hpx::naming::id_type lid = hpx::find_here();
    //the first measurement is of the amount of time it takes to generate
    //the specified number of dataflow_objects
    //the second test attempts to measure trigger times
    //the method will probably need to change for accurate results however

    high_resolution_timer t;
    lcos.push_back(new Dataflow(lid, a1, a2, a3, a4));
    for(i = 1; i < num; ++i)
        lcos.push_back(new Dataflow(lid, *lcos[i-1], a2, a3, a4));
    mean1 = t.elapsed()/num;

    t.restart();
    lcos[0]->get_future().get();
    mean2 = t.elapsed()/num;

    printf("Mean time for completing each action \n(when requesting the result of "
        "the final action to be executed): %f ns\n\n", mean2*1e9);

    lcos.push_back(new Dataflow(lid, a1, a2, a3, a4));
    for(i = 1; i < num; ++i)
        lcos.push_back(new Dataflow(lid, *lcos[i-1], a2, a3, a4));

    vector<hpx::lcos::future<T> > futures;
    futures.reserve(num);
    t.restart();
    for(i = num; i > 0; i--)
        futures.push_back(lcos[i-1]->get_future());
    mean2 = t.elapsed()/num;

    t.restart();
    for(i = num; i > 0; i--){
        futures[i-1].get();
    }
    mean3 = t.elapsed()/num;
    futures.clear();

    lcos.clear();
    time.reserve(num);
    lcos.reserve(num);
    t.restart();
    lcos.push_back(new Dataflow(lid, a1, a2, a3, a4));
    time.push_back(t.elapsed());
    for(i = 1; i < num; ++i){
        high_resolution_timer t1;
        lcos.push_back(new Dataflow(lid, *lcos[i-1], a2, a3, a4));
        time.push_back(t1.elapsed());
    }
    printout(time, ot, mean1, message);

    message = "Measuring time required to return futures from dataflow lcos:";
    time.clear();
    time.reserve(num);
    futures.reserve(num);
    for(i = num; i > 0; i--){
        high_resolution_timer t1;
        futures.push_back(lcos[i-1]->get_future());
        time.push_back(t1.elapsed());
    }
    printout(time, ot, mean2, message);

    time.clear();
    time.reserve(num);
    message = "Measuring time to get results from futures returned from dataflow lcos:";
    for(i = num; i > 0; i--){
        high_resolution_timer t1;
        futures[i-1].get();
        time.push_back(t1.elapsed());
    }
    printout(time, ot, mean3, message);
}
*/
