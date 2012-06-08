//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2012 Daniel Kogler
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "general_pa_declarations.cpp"

///////////////////////////////////////////////////////////////////////////////
//find approximate overhead of instantiating and obtaining results from
//high_resolution_timer
double timer_overhead(uint64_t iterations){
    vector<double> record;
    double total = 0;
    record.reserve(iterations);
    for(uint64_t i = 0; i < iterations; i++){
        high_resolution_timer in_t;
        record.push_back(in_t.elapsed());
    }
    for(uint64_t i = 0; i < iterations; i++) total += record[i];
    cout<<"\nAverage overhead of taking timed measurements: "<<
        total/iterations*1e9<<" ns\n";
    cout<<"NOTE - this value will be subtracted from all subsequent timings.\n";
    cout<<"       This value is an average taken at the start of the program\n";
    cout<<"       and as such the timings output by the program may be off\n";
    cout<<"       by several ( < 10) nanoseconds. \n\n";
    cout<<flush;
    return total/iterations;
}

//return the average time a particular action takes
inline double sample_mean(vector<double> time, double ot){
    long double sum = 0;
    for(uint64_t i = 0; i < time.size(); i++)
        sum += time[i] - ot;
    return sum/time.size();
}

//return the variance of the collected measurements
inline double sample_variance(vector<double> time, double ot, double mean){
    long double sum = 0;
    for(uint64_t i = 0; i < time.size(); i++)
        sum += pow(time[i] - ot - mean, 2);
    return sum/time.size();
}

//below returns the maximum and minimum time required per action
inline double sample_min(vector<double> time, double ot){
    double min = time[0]-ot;
    for(uint64_t i = 1; i < time.size(); i++)
        if(time[i]-ot < min) min = time[i]-ot;
    return min;
}
inline double sample_max(vector<double> time, double ot){
    double max = time[0]-ot;
    for(uint64_t i = 1; i < time.size(); i++)
        if(time[i]-ot > max) max = time[i]-ot;
    return max;
}

//print out the statistical results of the benchmark
void printout(vector<double> time, double ot, string message){
    double avg = sample_mean(time, ot);
    double var = sample_variance(time, ot, avg);
    double min = sample_min(time, ot);
    double max = sample_max(time, ot);
    cout<<message<<"\n";
    cout<<"Mean time:       "<<avg * 1e9<<" ns\n";
    cout<<"Variance:        "<<var * 1e9<<" ns\n";
    cout<<"Minimum time:    "<<min * 1e9<<" ns\n";
    cout<<"Maximum time:    "<<max * 1e9<<" ns\n\n";
    cout<<flush;
}

///////////////////////////////////////////////////////////////////////////////
template <typename Vector, typename Package>
void create_packages(Vector& packages, uint64_t num, double ot){
    uint64_t i = 0;
    packages.reserve(num);

    for(; i < num; ++i)
        packages.push_back(new Package());
}

///////////////////////////////////////////////////////////////////////////////

//this runs a series of tests for a packaged_action.apply()
template <typename Vector, typename Package, typename Action, typename T>
void run_tests(uint64_t);
template <typename Vector, typename Package, typename Action, typename T>
void run_tests(uint64_t, T);
template <typename Vector, typename Package, typename Action, typename T>
void run_tests(uint64_t, T, T);
template <typename Vector, typename Package, typename Action, typename T>
void run_tests(uint64_t, T, T, T);
template <typename Vector, typename Package, typename Action, typename T>
void run_tests(uint64_t, T, T, T, T);

///////////////////////////////////////////////////////////////////////////////
//all of the measured tests are declared in this section

//base case, just call apply.  used as control to validate other results

//first measure the time it takes to apply the logger. It's mostly here for 
//completion's sake, as the impact is very small
void apply_profiler(uint64_t num, double ot, hpx::naming::id_type const& gid){
    struct tag{};
    uint64_t i = 0;
    string message = "Measuring time to apply profiler logger for each action:";
    vector<double> time;
    time.reserve(num);
    hpx::util::block_profiler<tag> apply_logger_ = "packaged_action";
    
    for(; i < num; i++){
        high_resolution_timer t1;
        hpx::util::block_profiler_wrapper<tag> bp(apply_logger_);
        time.push_back(t1.elapsed());
    }
    printout(time, ot, message);
}

//next measure how long it takes to extract type information from the action
template <typename Action>
void apply_extraction(uint64_t num, double ot){
    uint64_t i = 0;
    string message = "Measuring time required to extract type information:";
    vector<double> time;
    time.reserve(num);
    typedef typename hpx::actions::extract_action<Action>::type action_type;

    for(; i < num; i++){
        high_resolution_timer t1;
        typedef 
            typename hpx::actions::extract_action<action_type>::result_type
            result_type;
        time.push_back(t1.elapsed());
    }
    printout(time, ot, message);
}

//next measure how long it takes to extract type information from the action
template <typename Vector>
void apply_get_gid(Vector packages, uint64_t num, double ot){
    uint64_t i = 0;
    string message = "Measuring time required to get package gids:";
    vector<double> time;
    time.reserve(num);

    for(; i < num; i++){
        high_resolution_timer t1;
        packages[i]->get_gid();
        time.push_back(t1.elapsed());
    }
    printout(time, ot, message);
}

//here measures the time it takes to create continuations
template<typename VectorP, typename VectorC, typename Result>
void apply_create_continuations(VectorP packages, VectorC& continuations, 
                                uint64_t num, double ot){
    typedef hpx::actions::base_lco_continuation<Result> rc_type;
    uint64_t i = 0;
    string message = "Measuring time required to create continuations:";
    vector<double> time;
    time.reserve(num);
    continuations.reserve(num);

    for(; i < num; ++i){
        const hpx::naming::id_type cgid = packages[i]->get_gid();
        high_resolution_timer t1;
        continuations.push_back(new rc_type(cgid));
        time.push_back(t1.elapsed());
    }
    printout(time, ot, message);
}


///////////////////////////////////////////////////////////////////////////////
//decide which action type to use
void decide_action_type(bool rtype, uint64_t num, int type, int argc);

//parse the argument types
void parse_arg(string atype, bool rtype, uint64_t num, int argc){
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
    decide_action_type(rtype, num, type, argc);
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm){
    uint64_t num = vm["number-spawned"].as<uint64_t>();
    bool rtype = (vm.count("result-action") ? true : false);
    string atype = vm["arg-type"].as<string>();
    int c = vm["argc"].as<int>();
    parse_arg(atype, rtype, num, c);
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
            "the default plain_actions.")
        ("arg-type,A",
            boost::program_options::value<string>()
                ->default_value("int"),
            "specifies the argument type of the action, as well as the result "
            "type if a plain_result_action is used, to be either int, long, "
            "float, or double")
        ("argc,C",
            boost::program_options::value<int>()
                ->default_value(2),
            "the number of arguments each action takes");

    // Initialize and run HPX
    return hpx::init(desc_commandline, argc, argv);
}

///////////////////////////////////////////////////////////////////////////////

//decide which action type to use
void decide_action_type(bool rtype, uint64_t num, int type, int argc){
    if(rtype){
        if(type == 0){
            switch(argc){
                case 0: run_tests<vector<empty_packagei0*>, empty_packagei0, 
                    empty_actioni0, int> (num); break;
                case 1: run_tests<vector<empty_packagei1*>, empty_packagei1, 
                    empty_actioni1, int> (num, ivar); break;
                case 2: run_tests<vector<empty_packagei2*>, empty_packagei2, 
                    empty_actioni2, int> (num, ivar, ivar); break;
                case 3: run_tests<vector<empty_packagei3*>, empty_packagei3, 
                    empty_actioni3, int> (num, ivar, ivar, ivar); break;
                default: run_tests<vector<empty_packagei4*>, empty_packagei4, 
                    empty_actioni4, int> (num, ivar, ivar, ivar, ivar); break;
            }
        }
        else if(type == 1){
            switch(argc){
                case 0: run_tests<vector<empty_packagel0*>, empty_packagel0, 
                    empty_actionl0, long> (num); break;
                case 1: run_tests<vector<empty_packagel1*>, empty_packagel1, 
                    empty_actionl1, long> (num, lvar); break;
                case 2: run_tests<vector<empty_packagel2*>, empty_packagel2, 
                    empty_actionl2, long> (num, lvar, lvar); break;
                case 3: run_tests<vector<empty_packagel3*>, empty_packagel3, 
                    empty_actionl3, long> (num, lvar, lvar, lvar); break;
                default: run_tests<vector<empty_packagel4*>, empty_packagel4, 
                    empty_actionl4, long> (num, lvar, lvar, lvar, lvar); break;
            }
        }
        else if(type == 2){
            switch(argc){
                case 0: run_tests<vector<empty_packagef0*>, empty_packagef0, 
                    empty_actionf0, float> (num); break;
                case 1: run_tests<vector<empty_packagef1*>, empty_packagef1, 
                    empty_actionf1, float> (num, fvar); break;
                case 2: run_tests<vector<empty_packagef2*>, empty_packagef2, 
                    empty_actionf2, float> (num, fvar, fvar); break;
                case 3: run_tests<vector<empty_packagef3*>, empty_packagef3, 
                    empty_actionf3, float> (num, fvar, fvar, fvar); break;
                default: run_tests<vector<empty_packagef4*>, empty_packagef4, 
                    empty_actionf4, float> (num, fvar, fvar, fvar, fvar); break;
            }
        }
        else{
            switch(argc){
                case 0: run_tests<vector<empty_packaged0*>, empty_packaged0, 
                    empty_actiond0, double> (num); break;
                case 1: run_tests<vector<empty_packaged1*>, empty_packaged1, 
                    empty_actiond1, double> (num, dvar); break;
                case 2: run_tests<vector<empty_packaged2*>, empty_packaged2, 
                    empty_actiond2, double> (num, dvar, dvar); break;
                case 3: run_tests<vector<empty_packaged3*>, empty_packaged3, 
                    empty_actiond3, double> (num, dvar, dvar, dvar); break;
                default: run_tests<vector<empty_packaged4*>, empty_packaged4, 
                    empty_actiond4, double> (num, dvar, dvar, dvar, dvar); break;
            }
        }
    }
    else{
        if(type == 0){
            switch(argc){
                case 0: run_tests<vector<void_package0*>, void_package0,
                    void_action0, void> (num); break;
                case 1: run_tests<vector<void_packagei1*>, void_packagei1, 
                    void_actioni1, int> (num, ivar); break;
                case 2: run_tests<vector<void_packagei2*>, void_packagei2, 
                    void_actioni2, int> (num, ivar, ivar); break;
                case 3: run_tests<vector<void_packagei3*>, void_packagei3, 
                    void_actioni3, int> (num, ivar, ivar, ivar); break;
                default: run_tests<vector<void_packagei4*>, void_packagei4, 
                    void_actioni4, int> (num, ivar, ivar, ivar, ivar); break;
            }
        }
        else if(type == 1){
            switch(argc){
                case 0: run_tests<vector<void_package0*>, void_package0,
                    void_action0, void> (num); break;
                case 1: run_tests<vector<void_packagel1*>, void_packagel1, 
                    void_actionl1, long> (num, lvar); break;
                case 2: run_tests<vector<void_packagel2*>, void_packagel2, 
                    void_actionl2, long> (num, lvar, lvar); break;
                case 3: run_tests<vector<void_packagel3*>, void_packagel3, 
                    void_actionl3, long> (num, lvar, lvar, lvar); break;
                default: run_tests<vector<void_packagel4*>, void_packagel4, 
                    void_actionl4, long> (num, lvar, lvar, lvar, lvar); break;
            }
        }
        else if(type == 2){
            switch(argc){
                case 0: run_tests<vector<void_package0*>, void_package0,
                    void_action0, void> (num); break;
                case 1: run_tests<vector<void_packagef1*>, void_packagef1, 
                    void_actionf1, float> (num, fvar); break;
                case 2: run_tests<vector<void_packagef2*>, void_packagef2, 
                    void_actionf2, float> (num, fvar, fvar); break;
                case 3: run_tests<vector<void_packagef3*>, void_packagef3, 
                    void_actionf3, float> (num, fvar, fvar, fvar); break;
                default: run_tests<vector<void_packagef4*>, void_packagef4, 
                    void_actionf4, float> (num, fvar, fvar, fvar, fvar); break;
            }
        }
        else{
            switch(argc){
                case 0: run_tests<vector<void_package0*>, void_package0,
                    void_action0, void> (num); break;
                case 1: run_tests<vector<void_packaged1*>, void_packaged1, 
                    void_actiond1, double> (num, dvar); break;
                case 2: run_tests<vector<void_packaged2*>, void_packaged2, 
                    void_actiond2, double> (num, dvar, dvar); break;
                case 3: run_tests<vector<void_packaged3*>, void_packaged3, 
                    void_actiond3, double> (num, dvar, dvar, dvar); break;
                default: run_tests<vector<void_packaged4*>, void_packaged4, 
                    void_actiond4, double> (num, dvar, dvar, dvar, dvar); break;
            }
        }
    }
}

//this runs a series of tests for packaged_action.apply()
template <typename Vector, typename Package, typename Action, typename T>
void run_tests(uint64_t num){
    uint64_t i = 0;
    double ot = timer_overhead(num);
    string message;
    vector<double> time;
    Vector packages;
    create_packages<Vector, Package>(packages, num, ot);
    hpx::naming::id_type lid = hpx::find_here();
    time.reserve(num);
    //first measure base case
    for(; i < num; ++i){
        high_resolution_timer t1;
        packages[i]->apply(lid);
        time.push_back(t1.elapsed());
    }
    message = "Measuring time required to apply packaged_actions by calling "
              "packaged_action.apply():";
    printout(time, ot, message);
    time.clear();

    //now we begin measuring the individual components
    packages.clear();
    packages.reserve(num);
    create_packages<Vector, Package>(packages, num, ot);

    //first measures applying profiler logger
    apply_profiler(num, ot, lid);

    //second measures extracting type information
    apply_extraction<Action>(num, ot);

    //third measures getting gid's from packages
    apply_get_gid<Vector>(packages, num, ot);

    typedef typename hpx::actions::extract_action<Action>::type action_type;
    typedef 
        typename hpx::actions::extract_action<action_type>::result_type
        result_type;
    typedef hpx::actions::base_lco_continuation<result_type> rc_type;
    vector<rc_type*> continuations;

    //measures creation time of continuations
    apply_create_continuations<Vector, vector<rc_type*>, result_type>
        (packages, continuations, num, ot);

    time.reserve(num);
    i = 0;
    message = "Measuring time required to apply continuations directly:";
    //finally apply the continuations directly
    for(; i < num; i++){
        high_resolution_timer t1;
        hpx::apply<action_type>(continuations[i], lid);
        time.push_back(t1.elapsed());
    }
    printout(time, ot, message);

    //next test creates continuations and applies them immediately
    time.clear();
    time.reserve(num);
    packages.clear();
    packages.reserve(num);
    create_packages<Vector, Package>(packages, num, ot);
    i = 0;
    message = "Measuring time required to create and apply continuations "
              "consecutively:";

    for(; i < num; i++){
        rc_type* cont;
        high_resolution_timer t1;
        cont = new rc_type(packages[i]->get_gid());
        hpx::apply<action_type>(cont, lid);
        time.push_back(t1.elapsed());
    }
    printout(time, ot, message);
}
template <typename Vector, typename Package, typename Action, typename T>
void run_tests(uint64_t num, T a1){
    uint64_t i = 0;
    double ot = timer_overhead(num);
    string message;
    vector<double> time;
    Vector packages;
    create_packages<Vector, Package>(packages, num, ot);
    hpx::naming::id_type lid = hpx::find_here();
    time.reserve(num);
    //first measure base case
    for(; i < num; ++i){
        high_resolution_timer t1;
        packages[i]->apply(lid, a1);
        time.push_back(t1.elapsed());
    }
    message = "Measuring time required to apply packaged_actions by calling "
              "packaged_action.apply():";
    printout(time, ot, message);
    time.clear();

    //now we begin measuring the individual components
    packages.clear();
    packages.reserve(num);
    create_packages<Vector, Package>(packages, num, ot);

    //first measures applying profiler logger
    apply_profiler(num, ot, lid);

    //second measures extracting type information
    apply_extraction<Action>(num, ot);

    //third measures getting gid's from packages
    apply_get_gid<Vector>(packages, num, ot);

    typedef typename hpx::actions::extract_action<Action>::type action_type;
    typedef 
        typename hpx::actions::extract_action<action_type>::result_type
        result_type;
    typedef hpx::actions::base_lco_continuation<result_type> rc_type;
    vector<rc_type*> continuations;

    //measures creation time of continuations
    apply_create_continuations<Vector, vector<rc_type*>, result_type>
        (packages, continuations, num, ot);

    time.reserve(num);
    i = 0;
    message = "Measuring time required to apply continuations directly:";
    //finally apply the continuations directly
    for(; i < num; i++){
        high_resolution_timer t1;
        hpx::apply<action_type>(continuations[i], lid, a1);
        time.push_back(t1.elapsed());
    }
    printout(time, ot, message);

    //next test creates continuations and applies them immediately
    time.clear();
    time.reserve(num);
    packages.clear();
    packages.reserve(num);
    create_packages<Vector, Package>(packages, num, ot);
    i = 0;
    message = "Measuring time required to create and apply continuations "
              "consecutively:";

    for(; i < num; i++){
        rc_type* cont;
        high_resolution_timer t1;
        cont = new rc_type(packages[i]->get_gid());
        hpx::apply<action_type>(cont, lid, a1);
        time.push_back(t1.elapsed());
    }
    printout(time, ot, message);
}
template <typename Vector, typename Package, typename Action, typename T>
void run_tests(uint64_t num, T a1, T a2){
    uint64_t i = 0;
    double ot = timer_overhead(num);
    string message;
    vector<double> time;
    Vector packages;
    create_packages<Vector, Package>(packages, num, ot);
    hpx::naming::id_type lid = hpx::find_here();
    time.reserve(num);
    //first measure base case
    for(; i < num; ++i){
        high_resolution_timer t1;
        packages[i]->apply(lid, a1, a2);
        time.push_back(t1.elapsed());
    }
    message = "Measuring time required to apply packaged_actions by calling "
              "packaged_action.apply():";
    printout(time, ot, message);
    time.clear();

    //now we begin measuring the individual components
    packages.clear();
    packages.reserve(num);
    create_packages<Vector, Package>(packages, num, ot);

    //first measures applying profiler logger
    apply_profiler(num, ot, lid);

    //second measures extracting type information
    apply_extraction<Action>(num, ot);

    //third measures getting gid's from packages
    apply_get_gid<Vector>(packages, num, ot);

    typedef typename hpx::actions::extract_action<Action>::type action_type;
    typedef 
        typename hpx::actions::extract_action<action_type>::result_type
        result_type;
    typedef hpx::actions::base_lco_continuation<result_type> rc_type;
    vector<rc_type*> continuations;

    //measures creation time of continuations
    apply_create_continuations<Vector, vector<rc_type*>, result_type>
        (packages, continuations, num, ot);

    time.reserve(num);
    i = 0;
    message = "Measuring time required to apply continuations directly:";
    //finally apply the continuations directly
    for(; i < num; i++){
        high_resolution_timer t1;
        hpx::apply<action_type>(continuations[i], lid, a1, a2);
        time.push_back(t1.elapsed());
    }
    printout(time, ot, message);

    //next test creates continuations and applies them immediately
    time.clear();
    time.reserve(num);
    packages.clear();
    packages.reserve(num);
    create_packages<Vector, Package>(packages, num, ot);
    i = 0;
    message = "Measuring time required to create and apply continuations "
              "consecutively:";

    for(; i < num; i++){
        rc_type* cont;
        high_resolution_timer t1;
        cont = new rc_type(packages[i]->get_gid());
        hpx::apply<action_type>(cont, lid, a1, a2);
        time.push_back(t1.elapsed());
    }
    printout(time, ot, message);
}
template <typename Vector, typename Package, typename Action, typename T>
void run_tests(uint64_t num, T a1, T a2, T a3){
    uint64_t i = 0;
    double ot = timer_overhead(num);
    string message;
    vector<double> time;
    Vector packages;
    create_packages<Vector, Package>(packages, num, ot);
    hpx::naming::id_type lid = hpx::find_here();
    time.reserve(num);
    //first measure base case
    for(; i < num; ++i){
        high_resolution_timer t1;
        packages[i]->apply(lid, a1, a2, a3);
        time.push_back(t1.elapsed());
    }
    message = "Measuring time required to apply packaged_actions by calling "
              "packaged_action.apply():";
    printout(time, ot, message);
    time.clear();

    //now we begin measuring the individual components
    packages.clear();
    packages.reserve(num);
    create_packages<Vector, Package>(packages, num, ot);

    //first measures applying profiler logger
    apply_profiler(num, ot, lid);

    //second measures extracting type information
    apply_extraction<Action>(num, ot);

    //third measures getting gid's from packages
    apply_get_gid<Vector>(packages, num, ot);

    typedef typename hpx::actions::extract_action<Action>::type action_type;
    typedef 
        typename hpx::actions::extract_action<action_type>::result_type
        result_type;
    typedef hpx::actions::base_lco_continuation<result_type> rc_type;
    vector<rc_type*> continuations;

    //measures creation time of continuations
    apply_create_continuations<Vector, vector<rc_type*>, result_type>
        (packages, continuations, num, ot);

    time.reserve(num);
    i = 0;
    message = "Measuring time required to apply continuations directly:";
    //finally apply the continuations directly
    for(; i < num; i++){
        high_resolution_timer t1;
        hpx::apply<action_type>(continuations[i], lid, a1, a2, a3);
        time.push_back(t1.elapsed());
    }
    printout(time, ot, message);

    //next test creates continuations and applies them immediately
    time.clear();
    time.reserve(num);
    packages.clear();
    packages.reserve(num);
    create_packages<Vector, Package>(packages, num, ot);
    i = 0;
    message = "Measuring time required to create and apply continuations "
              "consecutively:";

    for(; i < num; i++){
        rc_type* cont;
        high_resolution_timer t1;
        cont = new rc_type(packages[i]->get_gid());
        hpx::apply<action_type>(cont, lid, a1, a2, a3);
        time.push_back(t1.elapsed());
    }
    printout(time, ot, message);
}
template <typename Vector, typename Package, typename Action, typename T>
void run_tests(uint64_t num, T a1, T a2, T a3, T a4){
    uint64_t i = 0;
    double ot = timer_overhead(num);
    string message;
    vector<double> time;
    Vector packages;
    create_packages<Vector, Package>(packages, num, ot);
    hpx::naming::id_type lid = hpx::find_here();
    time.reserve(num);
    //first measure base case
    for(; i < num; ++i){
        high_resolution_timer t1;
        packages[i]->apply(lid, a1, a2, a3, a4);
        time.push_back(t1.elapsed());
    }
    message = "Measuring time required to apply packaged_actions by calling "
              "packaged_action.apply():";
    printout(time, ot, message);
    time.clear();

    //now we begin measuring the individual components
    packages.clear();
    packages.reserve(num);
    create_packages<Vector, Package>(packages, num, ot);

    //first measures applying profiler logger
    apply_profiler(num, ot, lid);

    //second measures extracting type information
    apply_extraction<Action>(num, ot);

    //third measures getting gid's from packages
    apply_get_gid<Vector>(packages, num, ot);

    typedef typename hpx::actions::extract_action<Action>::type action_type;
    typedef 
        typename hpx::actions::extract_action<action_type>::result_type
        result_type;
    typedef hpx::actions::base_lco_continuation<result_type> rc_type;
    vector<rc_type*> continuations;

    //measures creation time of continuations
    apply_create_continuations<Vector, vector<rc_type*>, result_type>
        (packages, continuations, num, ot);

    time.reserve(num);
    i = 0;
    message = "Measuring time required to apply continuations directly:";
    //finally apply the continuations directly
    for(; i < num; i++){
        high_resolution_timer t1;
        hpx::apply<action_type>(continuations[i], lid, a1, a2, a3, a4);
        time.push_back(t1.elapsed());
    }
    printout(time, ot, message);

    //next test creates continuations and applies them immediately
    time.clear();
    time.reserve(num);
    packages.clear();
    packages.reserve(num);
    create_packages<Vector, Package>(packages, num, ot);
    i = 0;
    message = "Measuring time required to create and apply continuations "
              "consecutively:";

    for(; i < num; i++){
        rc_type* cont;
        high_resolution_timer t1;
        cont = new rc_type(packages[i]->get_gid());
        hpx::apply<action_type>(cont, lid, a1, a2, a3, a4);
        time.push_back(t1.elapsed());
    }
    printout(time, ot, message);
}
