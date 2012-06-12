//  Copyright (c)      2012 Daniel Kogler
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "general_declarations.cpp"
#include "statstd.cpp"

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
//The first test creates the packaged_actions
template <typename Vector, typename Package>
void create_packages(Vector& packages, uint64_t number, double ot){
    uint64_t i = 0;
    string message = "Measuring time required to create packaged_actions:";
    vector<double> time;
    time.reserve(number);
    packages.reserve(number);

    //the first measurement is of the amount of time it takes to generate
    //the specified number of packaged_actions
    for(; i < number; ++i){
        high_resolution_timer t1;
        packages.push_back(new Package());
        time.push_back(t1.elapsed());
    }
    printout(time, ot, message);
}

//the third test collects the generated futures
template <typename Vector1, typename Vector2>
void get_futures(Vector1& packages, Vector2& futures, uint64_t number, double ot){
    uint64_t i = 0;
    string message = "Measuring time required to get futures associated with the actions:";
    vector<double> time;
    time.reserve(number);
    futures.reserve(number);

    //the third measurement is for the amount of time required to get the 
    //associated futures for each packaged action
    for(; i < number; ++i){
        high_resolution_timer t1;
        futures.push_back(packages[i]->get_future());
        time.push_back(t1.elapsed());
    }
    printout(time, ot, message);
}

//the fourth test gets results from futures
template <typename Vector>
void get_results(Vector futures, uint64_t number, double ot){
    uint64_t i = 0;
    string message = "Measuring time required to get results from futures:";
    vector<double> time;
    time.reserve(number);

    //the fourth measurement is for the amount of time required to get the 
    //results of each future
    for(; i < number; ++i){
        high_resolution_timer t1;
        futures[i].get();
        time.push_back(t1.elapsed());
    }
    printout(time, ot, message);
}

///////////////////////////////////////////////////////////////////////////////

//this runs a series of tests for a plain_result_action
template <typename Vector, typename Package, typename Action, typename T>
void run_empty(uint64_t);
template <typename Vector, typename Package, typename Action, typename T>
void run_empty(uint64_t, T);
template <typename Vector, typename Package, typename Action, typename T>
void run_empty(uint64_t, T, T);
template <typename Vector, typename Package, typename Action, typename T>
void run_empty(uint64_t, T, T, T);
template <typename Vector, typename Package, typename Action, typename T>
void run_empty(uint64_t, T, T, T, T);

//this runs a series of tests for a plain_action
template <typename Vector, typename Package, typename Action>
void run_void(uint64_t);
template <typename Vector, typename Package, typename Action, typename T>
void run_void(uint64_t, T);
template <typename Vector, typename Package, typename Action, typename T>
void run_void(uint64_t, T, T);
template <typename Vector, typename Package, typename Action, typename T>
void run_void(uint64_t, T, T, T);
template <typename Vector, typename Package, typename Action, typename T>
void run_void(uint64_t, T, T, T, T);

///////////////////////////////////////////////////////////////////////////////

//decide which action type to use
void decide_action_type(bool rtype, uint64_t number, int type, int argc);

//parse the argument types
void parse_arg(string atype, bool rtype, uint64_t number, int argc){
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
    decide_action_type(rtype, number, type, argc);
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm){
    uint64_t number = vm["number-spawned"].as<uint64_t>();
    bool rtype = (vm.count("result-action") ? true : false);
    string atype = vm["arg-type"].as<string>();
    int c = vm["argc"].as<int>();
    parse_arg(atype, rtype, number, c);
    return hpx::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[]){
    // Configure application-specific options.
    options_description
        desc_commandline("usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ("number-spawned,N",
            boost::program_options::value<uint64_t>()
                ->default_value(500000),
            "number of packaged_actions created and run")
        ("result-action,R",
            "specifies that we are also to use plain_result_actions instead of "
            "the default plain_actions.  Additional tests will be performed "
            "regarding futures and return values if this option is specified")
        ("arg-type,A",
            boost::program_options::value<string>()
                ->default_value("int"),
            "specifies the argument type of the action, as well as the result "
            "type if a plain_result_action is used, to be either int, long, "
            "float, or double. This argument has no effect if the number of "
            "arguments specified is 0")
        ("argc,C",
            boost::program_options::value<int>()
                ->default_value(1),
            "the number of arguments each action takes");

    // Initialize and run HPX
    return hpx::init(desc_commandline, argc, argv);
}

///////////////////////////////////////////////////////////////////////////////
//Below is purely tedium

//decide which action type to use
void decide_action_type(bool rtype, uint64_t number, int type, int argc){
    if(rtype){
        if(type == 0){
            switch(argc){
                case 0: run_empty<vector<empty_packagei0*>, empty_packagei0, 
                    empty_actioni0, int> (number); break;
                case 1: run_empty<vector<empty_packagei1*>, empty_packagei1, 
                    empty_actioni1, int> (number, ivar); break;
                case 2: run_empty<vector<empty_packagei2*>, empty_packagei2, 
                    empty_actioni2, int> (number, ivar, ivar); break;
                case 3: run_empty<vector<empty_packagei3*>, empty_packagei3, 
                    empty_actioni3, int> (number, ivar, ivar, ivar); break;
                default: run_empty<vector<empty_packagei4*>, empty_packagei4, 
                    empty_actioni4, int> (number, ivar, ivar, ivar, ivar); break;
            }
        }
        else if(type == 1){
            switch(argc){
                case 0: run_empty<vector<empty_packagel0*>, empty_packagel0, 
                    empty_actionl0, long> (number); break;
                case 1: run_empty<vector<empty_packagel1*>, empty_packagel1, 
                    empty_actionl1, long> (number, lvar); break;
                case 2: run_empty<vector<empty_packagel2*>, empty_packagel2, 
                    empty_actionl2, long> (number, lvar, lvar); break;
                case 3: run_empty<vector<empty_packagel3*>, empty_packagel3, 
                    empty_actionl3, long> (number, lvar, lvar, lvar); break;
                default: run_empty<vector<empty_packagel4*>, empty_packagel4, 
                    empty_actionl4, long> (number, lvar, lvar, lvar, lvar); break;
            }
        }
        else if(type == 2){
            switch(argc){
                case 0: run_empty<vector<empty_packagef0*>, empty_packagef0, 
                    empty_actionf0, float> (number); break;
                case 1: run_empty<vector<empty_packagef1*>, empty_packagef1, 
                    empty_actionf1, float> (number, fvar); break;
                case 2: run_empty<vector<empty_packagef2*>, empty_packagef2, 
                    empty_actionf2, float> (number, fvar, fvar); break;
                case 3: run_empty<vector<empty_packagef3*>, empty_packagef3, 
                    empty_actionf3, float> (number, fvar, fvar, fvar); break;
                default: run_empty<vector<empty_packagef4*>, empty_packagef4, 
                    empty_actionf4, float> (number, fvar, fvar, fvar, fvar); break;
            }
        }
        else{
            switch(argc){
                case 0: run_empty<vector<empty_packaged0*>, empty_packaged0, 
                    empty_actiond0, double> (number); break;
                case 1: run_empty<vector<empty_packaged1*>, empty_packaged1, 
                    empty_actiond1, double> (number, dvar); break;
                case 2: run_empty<vector<empty_packaged2*>, empty_packaged2, 
                    empty_actiond2, double> (number, dvar, dvar); break;
                case 3: run_empty<vector<empty_packaged3*>, empty_packaged3, 
                    empty_actiond3, double> (number, dvar, dvar, dvar); break;
                default: run_empty<vector<empty_packaged4*>, empty_packaged4, 
                    empty_actiond4, double> (number, dvar, dvar, dvar, dvar); break;
            }
        }
    }
    else{
        if(type == 0){
            switch(argc){
                case 0: run_void<vector<void_package0*>, void_package0,
                    void_action0> (number); break;
                case 1: run_void<vector<void_packagei1*>, void_packagei1, 
                    void_actioni1, int> (number, ivar); break;
                case 2: run_void<vector<void_packagei2*>, void_packagei2, 
                    void_actioni2, int> (number, ivar, ivar); break;
                case 3: run_void<vector<void_packagei3*>, void_packagei3, 
                    void_actioni3, int> (number, ivar, ivar, ivar); break;
                default: run_void<vector<void_packagei4*>, void_packagei4, 
                    void_actioni4, int> (number, ivar, ivar, ivar, ivar); break;
            }
        }
        else if(type == 1){
            switch(argc){
                case 0: run_void<vector<void_package0*>, void_package0,
                    void_action0> (number); break;
                case 1: run_void<vector<void_packagel1*>, void_packagel1, 
                    void_actionl1, long> (number, lvar); break;
                case 2: run_void<vector<void_packagel2*>, void_packagel2, 
                    void_actionl2, long> (number, lvar, lvar); break;
                case 3: run_void<vector<void_packagel3*>, void_packagel3, 
                    void_actionl3, long> (number, lvar, lvar, lvar); break;
                default: run_void<vector<void_packagel4*>, void_packagel4, 
                    void_actionl4, long> (number, lvar, lvar, lvar, lvar); break;
            }
        }
        else if(type == 2){
            switch(argc){
                case 0: run_void<vector<void_package0*>, void_package0,
                    void_action0> (number); break;
                case 1: run_void<vector<void_packagef1*>, void_packagef1, 
                    void_actionf1, float> (number, fvar); break;
                case 2: run_void<vector<void_packagef2*>, void_packagef2, 
                    void_actionf2, float> (number, fvar, fvar); break;
                case 3: run_void<vector<void_packagef3*>, void_packagef3, 
                    void_actionf3, float> (number, fvar, fvar, fvar); break;
                default: run_void<vector<void_packagef4*>, void_packagef4, 
                    void_actionf4, float> (number, fvar, fvar, fvar, fvar); break;
            }
        }
        else{
            switch(argc){
                case 0: run_void<vector<void_package0*>, void_package0,
                    void_action0> (number); break;
                case 1: run_void<vector<void_packaged1*>, void_packaged1, 
                    void_actiond1, double> (number, dvar); break;
                case 2: run_void<vector<void_packaged2*>, void_packaged2, 
                    void_actiond2, double> (number, dvar, dvar); break;
                case 3: run_void<vector<void_packaged3*>, void_packaged3, 
                    void_actiond3, double> (number, dvar, dvar, dvar); break;
                default: run_void<vector<void_packaged4*>, void_packaged4, 
                    void_actiond4, double> (number, dvar, dvar, dvar, dvar); break;
            }
        }
    }
}

//this runs a series of tests for a plain_result_action
template <typename Vector, typename Package, typename Action, typename T>
void run_empty(uint64_t number){
    uint64_t i = 0;
    double ot = timer_overhead(number);
    string message;
    vector<double> time;
    Vector packages;

    //first test
    create_packages<Vector, Package>(packages, number, ot);

    //second test
    hpx::naming::id_type lid = hpx::find_here();
    //part1

    //part2
    time.reserve(number);
    for(; i < number; ++i){
        high_resolution_timer t1;
        packages[i]->apply(lid);
        time.push_back(t1.elapsed());
    }
    message = "Measuring time required to apply packaged_actions:";
    printout(time, ot, message);
    time.clear();

    vector<hpx::lcos::future<T> > futures;

    //third test
    get_futures<Vector,vector<hpx::lcos::future<T> > >(packages,futures,number,ot); 
    packages.clear();

    //fourth test
    get_results<vector<hpx::lcos::future<T> > >(futures, number, ot);
}
template <typename Vector, typename Package, typename Action, typename T>
void run_empty(uint64_t number, T a1){
    uint64_t i = 0;
    double ot = timer_overhead(number);
    string message;
    vector<double> time;
    Vector packages;
    create_packages<Vector, Package>(packages, number, ot);
    hpx::naming::id_type lid = hpx::find_here();
    time.reserve(number);
    for(; i < number; ++i){
        high_resolution_timer t1;
        packages[i]->apply(lid, a1);
        time.push_back(t1.elapsed());
    }
    message = "Measuring time required to apply packaged_actions:";
    printout(time, ot, message);
    time.clear();
    vector<hpx::lcos::future<T> > futures;
    get_futures<Vector,vector<hpx::lcos::future<T> > >(packages,futures,number,ot); 
    packages.clear();
    get_results<vector<hpx::lcos::future<T> > >(futures, number, ot);
}
template <typename Vector, typename Package, typename Action, typename T>
void run_empty(uint64_t number, T a1, T a2){
    uint64_t i = 0;
    double ot = timer_overhead(number);
    string message;
    vector<double> time;
    Vector packages;
    create_packages<Vector, Package>(packages, number, ot);
    hpx::naming::id_type lid = hpx::find_here();
    time.reserve(number);
    for(; i < number; ++i){
        high_resolution_timer t1;
        packages[i]->apply(lid, a1, a2);
        time.push_back(t1.elapsed());
    }
    message = "Measuring time required to apply packaged_actions:";
    printout(time, ot, message);
    time.clear();
    vector<hpx::lcos::future<T> > futures;
    get_futures<Vector,vector<hpx::lcos::future<T> > >(packages,futures,number,ot); 
    packages.clear();
    get_results<vector<hpx::lcos::future<T> > >(futures, number, ot);
}
template <typename Vector, typename Package, typename Action, typename T>
void run_empty(uint64_t number, T a1, T a2, T a3){
    uint64_t i = 0;
    double ot = timer_overhead(number);
    string message;
    vector<double> time;
    Vector packages;
    create_packages<Vector, Package>(packages, number, ot);
    hpx::naming::id_type lid = hpx::find_here();
    time.reserve(number);
    for(; i < number; ++i){
        high_resolution_timer t1;
        packages[i]->apply(lid, a1, a2, a3);
        time.push_back(t1.elapsed());
    }
    message = "Measuring time required to apply packaged_actions:";
    printout(time, ot, message);
    time.clear();
    vector<hpx::lcos::future<T> > futures;
    get_futures<Vector,vector<hpx::lcos::future<T> > >(packages,futures,number,ot); 
    packages.clear();
    get_results<vector<hpx::lcos::future<T> > >(futures, number, ot);
}
template <typename Vector, typename Package, typename Action, typename T>
void run_empty(uint64_t number, T a1, T a2, T a3, T a4){
    uint64_t i = 0;
    double ot = timer_overhead(number);
    string message;
    vector<double> time;
    Vector packages;
    create_packages<Vector, Package>(packages, number, ot);
    hpx::naming::id_type lid = hpx::find_here();
    time.reserve(number);
    for(; i < number; ++i){
        high_resolution_timer t1;
        packages[i]->apply(lid, a1, a2, a3, a4);
        time.push_back(t1.elapsed());
    }
    message = "Measuring time required to apply packaged_actions:";
    printout(time, ot, message);
    time.clear();
    vector<hpx::lcos::future<T> > futures;
    get_futures<Vector,vector<hpx::lcos::future<T> > >(packages,futures,number,ot);
    packages.clear();
    get_results<vector<hpx::lcos::future<T> > >(futures, number, ot);
}

//this runs a series of tests for a plain_action
template <typename Vector, typename Package, typename Action>
void run_void(uint64_t number){
    uint64_t i = 0;
    double ot = timer_overhead(number);
    string message;
    vector<double> time;
    Vector packages;

    //first test
    create_packages<Vector, Package>(packages, number, ot);

    //second test
    hpx::naming::id_type lid = hpx::find_here();
    //part1
    //part2
    time.reserve(number);
    //the second measurement records the time needed to apply actions. 
    //the time needed to complete the actions is not yet recorded
    for(; i < number; ++i){
        high_resolution_timer t1;
        packages[i]->apply(lid);
        time.push_back(t1.elapsed());
    }
    message = "Measuring time required to apply packaged_actions:";
    printout(time, ot, message);
}
template <typename Vector, typename Package, typename Action, typename T>
void run_void(uint64_t number, T a1){
    uint64_t i = 0;
    double ot = timer_overhead(number);
    string message;
    vector<double> time;
    Vector packages;
    create_packages<Vector, Package>(packages, number, ot);
    hpx::naming::id_type lid = hpx::find_here();
    time.reserve(number);
    for(; i < number; ++i){
        high_resolution_timer t1;
        packages[i]->apply(lid, a1);
        time.push_back(t1.elapsed());
    }
    message = "Measuring time required to apply packaged_actions:";
    printout(time, ot, message);
}
template <typename Vector, typename Package, typename Action, typename T>
void run_void(uint64_t number, T a1, T a2){
    uint64_t i = 0;
    double ot = timer_overhead(number);
    string message;
    vector<double> time;
    Vector packages;
    create_packages<Vector, Package>(packages, number, ot);
    hpx::naming::id_type lid = hpx::find_here();
    time.reserve(number);
    for(; i < number; ++i){
        high_resolution_timer t1;
        packages[i]->apply(lid, a1, a2);
        time.push_back(t1.elapsed());
    }
    message = "Measuring time required to apply packaged_actions:";
    printout(time, ot, message);
}
template <typename Vector, typename Package, typename Action, typename T>
void run_void(uint64_t number, T a1, T a2, T a3){
    uint64_t i = 0;
    double ot = timer_overhead(number);
    string message;
    vector<double> time;
    Vector packages;
    create_packages<Vector, Package>(packages, number, ot);
    hpx::naming::id_type lid = hpx::find_here();
    time.reserve(number);
    for(; i < number; ++i){
        high_resolution_timer t1;
        packages[i]->apply(lid, a1, a2, a3);
        time.push_back(t1.elapsed());
    }
    message = "Measuring time required to apply packaged_actions:";
    printout(time, ot, message);
}
template <typename Vector, typename Package, typename Action, typename T>
void run_void(uint64_t number, T a1, T a2, T a3, T a4){
    uint64_t i = 0;
    double ot = timer_overhead(number);
    string message;
    vector<double> time;
    Vector packages;
    create_packages<Vector, Package>(packages, number, ot);
    hpx::naming::id_type lid = hpx::find_here();
    time.reserve(number);
    for(; i < number; ++i){
        high_resolution_timer t1;
        packages[i]->apply(lid, a1, a2, a3, a4);
        time.push_back(t1.elapsed());
    }
    message = "Measuring time required to apply packaged_actions:";
    printout(time, ot, message);
}
