//  Copyright (c)      2012 Daniel Kogler
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "general_declarations.cpp"
#include "statstd.cpp"

///////////////////////////////////////////////////////////////////////////////
template <typename Vector, typename Package>
void create_packages(Vector& packages, uint64_t num){
    uint64_t i = 0;
    packages.reserve(num);

    for(; i < num; ++i)
        packages.push_back(new Package());
}

///////////////////////////////////////////////////////////////////////////////

//this runs a series of tests for a packaged_action.apply()
template <typename Vector, typename Package, typename Action, typename T>
void run_tests(bool, uint64_t);
template <typename Vector, typename Package, typename Action, typename T>
void run_tests(bool, uint64_t, T);
template <typename Vector, typename Package, typename Action, typename T>
void run_tests(bool, uint64_t, T, T);
template <typename Vector, typename Package, typename Action, typename T>
void run_tests(bool, uint64_t, T, T, T);
template <typename Vector, typename Package, typename Action, typename T>
void run_tests(bool, uint64_t, T, T, T, T);

///////////////////////////////////////////////////////////////////////////////
//all of the measured tests are declared in this section

//base case, just call apply.  used as control to validate other results

//next measure how long it takes to obtain gids of packages
template <typename Vector>
void apply_get_gid(Vector packages, uint64_t num, double ot){
    uint64_t i = 0;
    double mean;
    string message = "Measuring time required to get package gids:";
    vector<double> time;
    time.reserve(num);

    high_resolution_timer t;
    for(; i < num; i++)
        packages[i]->get_gid();
    mean = t.elapsed()/num;
    for(i = 0; i < num; i++){
        high_resolution_timer t1;
        packages[i]->get_gid();
        time.push_back(t1.elapsed());
    }
    printout(time, ot, mean, message);
}

//here measures the time it takes to create continuations
template<typename VectorP, typename VectorC, typename Result>
void apply_create_continuations(VectorP packages, VectorC& continuations, 
                                uint64_t num, double ot){
    typedef hpx::actions::base_lco_continuation<Result> rc_type;
    uint64_t i = 0;
    double mean;
    string message = "Measuring time required to create continuations:";
    vector<double> time;
    vector<hpx::naming::id_type> cgids;
    rc_type* temp;
    for(; i < num; i++)
        cgids.push_back(packages[i]->get_gid());
    high_resolution_timer t;
    for(i = 0; i < num; i++)
        temp = new rc_type(cgids[i]);
    mean = t.elapsed()/num;
    cgids.clear();
    time.reserve(num);
    continuations.reserve(num);
    for(i = 0; i < num; ++i){
        const hpx::naming::id_type cgid = packages[i]->get_gid();
        high_resolution_timer t1;
        continuations.push_back(new rc_type(cgid));
        time.push_back(t1.elapsed());
    }
    printout(time, ot, mean, message);
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
    csv = (vm.count("csv") ? true : false);
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
            "the number of arguments each action takes")
        ("csv",
            "output results as csv "
            "(format:count,mean,accurate mean,variance,min,max)");

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
                    empty_actioni0, int> (true, num); break;
                case 1: run_tests<vector<empty_packagei1*>, empty_packagei1, 
                    empty_actioni1, int> (true, num, ivar); break;
                case 2: run_tests<vector<empty_packagei2*>, empty_packagei2, 
                    empty_actioni2, int> (true, num, ivar, ivar); break;
                case 3: run_tests<vector<empty_packagei3*>, empty_packagei3, 
                    empty_actioni3, int> (true, num, ivar, ivar, ivar); break;
                default: run_tests<vector<empty_packagei4*>, empty_packagei4, 
                    empty_actioni4, int> (true, num, ivar, ivar, ivar, ivar); 
            }
        }
        else if(type == 1){
            switch(argc){
                case 0: run_tests<vector<empty_packagel0*>, empty_packagel0, 
                    empty_actionl0, long> (true, num); break;
                case 1: run_tests<vector<empty_packagel1*>, empty_packagel1, 
                    empty_actionl1, long> (true, num, lvar); break;
                case 2: run_tests<vector<empty_packagel2*>, empty_packagel2, 
                    empty_actionl2, long> (true, num, lvar, lvar); break;
                case 3: run_tests<vector<empty_packagel3*>, empty_packagel3, 
                    empty_actionl3, long> (true, num, lvar, lvar, lvar); break;
                default: run_tests<vector<empty_packagel4*>, empty_packagel4, 
                    empty_actionl4, long> (true, num, lvar, lvar, lvar, lvar);
            }
        }
        else if(type == 2){
            switch(argc){
                case 0: run_tests<vector<empty_packagef0*>, empty_packagef0, 
                    empty_actionf0, float> (true, num); break;
                case 1: run_tests<vector<empty_packagef1*>, empty_packagef1, 
                    empty_actionf1, float> (true, num, fvar); break;
                case 2: run_tests<vector<empty_packagef2*>, empty_packagef2, 
                    empty_actionf2, float> (true, num, fvar, fvar); break;
                case 3: run_tests<vector<empty_packagef3*>, empty_packagef3, 
                    empty_actionf3, float> (true, num, fvar, fvar, fvar); break;
                default: run_tests<vector<empty_packagef4*>, empty_packagef4, 
                    empty_actionf4, float> (true, num, fvar, fvar, fvar, fvar); 
            }
        }
        else{
            switch(argc){
                case 0: run_tests<vector<empty_packaged0*>, empty_packaged0, 
                    empty_actiond0, double> (true, num); break;
                case 1: run_tests<vector<empty_packaged1*>, empty_packaged1, 
                    empty_actiond1, double> (true, num, dvar); break;
                case 2: run_tests<vector<empty_packaged2*>, empty_packaged2, 
                    empty_actiond2, double> (true, num, dvar, dvar); break;
                case 3: run_tests<vector<empty_packaged3*>, empty_packaged3, 
                    empty_actiond3, double> (true, num, dvar, dvar, dvar); break;
                default: run_tests<vector<empty_packaged4*>, empty_packaged4, 
                    empty_actiond4, double> (true, num, dvar, dvar, dvar, dvar);
            }
        }
    }
    else{
        if(type == 0){
            switch(argc){
                case 0: run_tests<vector<void_package0*>, void_package0,
                    void_action0, void> (false, num); break;
                case 1: run_tests<vector<void_packagei1*>, void_packagei1, 
                    void_actioni1, int> (false, num, ivar); break;
                case 2: run_tests<vector<void_packagei2*>, void_packagei2, 
                    void_actioni2, int> (false, num, ivar, ivar); break;
                case 3: run_tests<vector<void_packagei3*>, void_packagei3, 
                    void_actioni3, int> (false, num, ivar, ivar, ivar); break;
                default: run_tests<vector<void_packagei4*>, void_packagei4, 
                    void_actioni4, int> (false, num, ivar, ivar, ivar, ivar); 
            }
        }
        else if(type == 1){
            switch(argc){
                case 0: run_tests<vector<void_package0*>, void_package0,
                    void_action0, void> (false, num); break;
                case 1: run_tests<vector<void_packagel1*>, void_packagel1, 
                    void_actionl1, long> (false, num, lvar); break;
                case 2: run_tests<vector<void_packagel2*>, void_packagel2, 
                    void_actionl2, long> (false, num, lvar, lvar); break;
                case 3: run_tests<vector<void_packagel3*>, void_packagel3, 
                    void_actionl3, long> (false, num, lvar, lvar, lvar); break;
                default: run_tests<vector<void_packagel4*>, void_packagel4, 
                    void_actionl4, long> (false, num, lvar, lvar, lvar, lvar); 
            }
        }
        else if(type == 2){
            switch(argc){
                case 0: run_tests<vector<void_package0*>, void_package0,
                    void_action0, void> (false, num); break;
                case 1: run_tests<vector<void_packagef1*>, void_packagef1, 
                    void_actionf1, float> (false, num, fvar); break;
                case 2: run_tests<vector<void_packagef2*>, void_packagef2, 
                    void_actionf2, float> (false, num, fvar, fvar); break;
                case 3: run_tests<vector<void_packagef3*>, void_packagef3, 
                    void_actionf3, float> (false, num, fvar, fvar, fvar); break;
                default: run_tests<vector<void_packagef4*>, void_packagef4, 
                    void_actionf4, float> (false, num, fvar, fvar, fvar, fvar);
            }
        }
        else{
            switch(argc){
                case 0: run_tests<vector<void_package0*>, void_package0,
                    void_action0, void> (false, num); break;
                case 1: run_tests<vector<void_packaged1*>, void_packaged1, 
                    void_actiond1, double> (false, num, dvar); break;
                case 2: run_tests<vector<void_packaged2*>, void_packaged2, 
                    void_actiond2, double> (false, num, dvar, dvar); break;
                case 3: run_tests<vector<void_packaged3*>, void_packaged3, 
                    void_actiond3, double> (false, num, dvar, dvar, dvar); break;
                default: run_tests<vector<void_packaged4*>, void_packaged4, 
                    void_actiond4, double> (false, num, dvar, dvar, dvar, dvar);
            }
        }
    }
}

//this runs a series of tests for packaged_action.apply()
template <typename Vector, typename Package, typename Action, typename T>
void run_tests(bool empty, uint64_t num){
    uint64_t i = 0;
    double ot = timer_overhead(num);
    string message;
    vector<double> time;
    Vector packages;
    create_packages<Vector, Package>(packages, num);
    hpx::naming::id_type lid = hpx::find_here();
    //first measure base case
    high_resolution_timer t;
    for(; i < num; ++i) packages[i]->apply(lid);
    double mean = t.elapsed()/num;
    packages.clear();
    create_packages<Vector, Package>(packages, num);
    time.reserve(num);
    for(i = 0; i < num; ++i){
        high_resolution_timer t1;
        packages[i]->apply(lid);
        time.push_back(t1.elapsed());
    }
    message = "Measuring time required to apply packaged_actions by calling "
              "packaged_action.apply():";
    printout(time, ot, mean, message);
    time.clear();

    //now we begin measuring the individual components
    packages.clear();
    create_packages<Vector, Package>(packages, num);

    //third measures getting gid's from packages
    apply_get_gid<Vector>(packages, num, ot);

    if(empty){
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef 
            typename hpx::actions::extract_action<action_type>::result_type
            result_type;
        typedef hpx::actions::base_lco_continuation<result_type> rc_type;
        vector<rc_type*> continuations;

        //measures creation time of continuations
        apply_create_continuations<Vector, vector<rc_type*>, result_type>
            (packages, continuations, num, ot);

        //finally apply the continuations directly
        high_resolution_timer tt;
        for(i = 0; i < num; i++) 
            hpx::apply<action_type>(lid);
        mean = tt.elapsed()/num;
        time.reserve(num);
        message = "Measuring time required to apply continuations directly:";

        for(i = 0; i < num; i++){
            high_resolution_timer t1;
            hpx::apply<action_type>(lid);
            time.push_back(t1.elapsed());
        }
        printout(time, ot, mean, message);
    }
    else{
        //next test applies actions directly, skipping get_gid stage
        high_resolution_timer tt;
        for(i = 0; i < num; ++i)
            hpx::apply<Action>(lid);
        mean = tt.elapsed()/num;
        time.reserve(num);
        message = "Measuring time required to apply actions directly:";
        for(i = 0; i < num; i++){
            high_resolution_timer t1;
            hpx::apply<Action>(lid);
            time.push_back(t1.elapsed());
        }
        printout(time, ot, mean, message);
    }
}
template <typename Vector, typename Package, typename Action, typename T>
void run_tests(bool empty, uint64_t num, T a1){
    uint64_t i = 0;
    double ot = timer_overhead(num);
    string message;
    vector<double> time;
    Vector packages;
    create_packages<Vector, Package>(packages, num);
    hpx::naming::id_type lid = hpx::find_here();
    //first measure base case
    high_resolution_timer t;
    for(; i < num; ++i) packages[i]->apply(lid, a1);
    double mean = t.elapsed()/num;
    packages.clear();
    create_packages<Vector, Package>(packages, num);
    time.reserve(num);
    for(i = 0; i < num; ++i){
        high_resolution_timer t1;
        packages[i]->apply(lid, a1);
        time.push_back(t1.elapsed());
    }
    message = "Measuring time required to apply packaged_actions by calling "
              "packaged_action.apply():";
    printout(time, ot, mean, message);
    time.clear();

    //now we begin measuring the individual components
    packages.clear();
    create_packages<Vector, Package>(packages, num);

    //third measures getting gid's from packages
    apply_get_gid<Vector>(packages, num, ot);

    if(empty){
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef 
            typename hpx::actions::extract_action<action_type>::result_type
            result_type;
        typedef hpx::actions::base_lco_continuation<result_type> rc_type;
        vector<rc_type*> continuations;

        //measures creation time of continuations
        apply_create_continuations<Vector, vector<rc_type*>, result_type>
            (packages, continuations, num, ot);

        //finally apply the continuations directly
        high_resolution_timer tt;
        for(i = 0; i < num; i++) 
            hpx::apply<action_type>(lid, a1);
        mean = tt.elapsed()/num;
        time.reserve(num);
        message = "Measuring time required to apply continuations directly:";

        for(i = 0; i < num; i++){
            high_resolution_timer t1;
            hpx::apply<action_type>(lid, a1);
            time.push_back(t1.elapsed());
        }
        printout(time, ot, mean, message);
    }
    else{
        //next test applies actions directly, skipping get_gid stage
        high_resolution_timer tt;
        for(i = 0; i < num; ++i)
            hpx::apply<Action>(lid, a1);
        mean = tt.elapsed()/num;
        time.reserve(num);
        message = "Measuring time required to apply actions directly:";
        for(i = 0; i < num; i++){
            high_resolution_timer t1;
            hpx::apply<Action>(lid, a1);
            time.push_back(t1.elapsed());
        }
        printout(time, ot, mean, message);
    }
}
template <typename Vector, typename Package, typename Action, typename T>
void run_tests(bool empty, uint64_t num, T a1, T a2){
    uint64_t i = 0;
    double ot = timer_overhead(num);
    string message;
    vector<double> time;
    Vector packages;
    create_packages<Vector, Package>(packages, num);
    hpx::naming::id_type lid = hpx::find_here();
    //first measure base case
    high_resolution_timer t;
    for(; i < num; ++i) packages[i]->apply(lid, a1, a2);
    double mean = t.elapsed()/num;
    packages.clear();
    create_packages<Vector, Package>(packages, num);
    time.reserve(num);
    for(i = 0; i < num; ++i){
        high_resolution_timer t1;
        packages[i]->apply(lid, a1, a2);
        time.push_back(t1.elapsed());
    }
    message = "Measuring time required to apply packaged_actions by calling "
              "packaged_action.apply():";
    printout(time, ot, mean, message);
    time.clear();

    //now we begin measuring the individual components
    packages.clear();
    create_packages<Vector, Package>(packages, num);

    //third measures getting gid's from packages
    apply_get_gid<Vector>(packages, num, ot);

    if(empty){
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef 
            typename hpx::actions::extract_action<action_type>::result_type
            result_type;
        typedef hpx::actions::base_lco_continuation<result_type> rc_type;
        vector<rc_type*> continuations;

        //measures creation time of continuations
        apply_create_continuations<Vector, vector<rc_type*>, result_type>
            (packages, continuations, num, ot);

        //finally apply the continuations directly
        high_resolution_timer tt;
        for(i = 0; i < num; i++) 
            hpx::apply<action_type>(lid, a1, a2);
        mean = tt.elapsed()/num;
        time.reserve(num);
        message = "Measuring time required to apply continuations directly:";

        for(i = 0; i < num; i++){
            high_resolution_timer t1;
            hpx::apply<action_type>(lid, a1, a2);
            time.push_back(t1.elapsed());
        }
        printout(time, ot, mean, message);
    }
    else{
        //next test applies actions directly, skipping get_gid stage
        high_resolution_timer tt;
        for(i = 0; i < num; ++i)
            hpx::apply<Action>(lid, a1, a2);
        mean = tt.elapsed()/num;
        time.reserve(num);
        message = "Measuring time required to apply actions directly:";
        for(i = 0; i < num; i++){
            high_resolution_timer t1;
            hpx::apply<Action>(lid, a1, a2);
            time.push_back(t1.elapsed());
        }
        printout(time, ot, mean, message);
    }
}
template <typename Vector, typename Package, typename Action, typename T>
void run_tests(bool empty, uint64_t num, T a1, T a2, T a3){
    uint64_t i = 0;
    double ot = timer_overhead(num);
    string message;
    vector<double> time;
    Vector packages;
    create_packages<Vector, Package>(packages, num);
    hpx::naming::id_type lid = hpx::find_here();
    //first measure base case
    high_resolution_timer t;
    for(; i < num; ++i) packages[i]->apply(lid, a1, a2, a3);
    double mean = t.elapsed()/num;
    packages.clear();
    create_packages<Vector, Package>(packages, num);
    time.reserve(num);
    for(i = 0; i < num; ++i){
        high_resolution_timer t1;
        packages[i]->apply(lid, a1, a2, a3);
        time.push_back(t1.elapsed());
    }
    message = "Measuring time required to apply packaged_actions by calling "
              "packaged_action.apply():";
    printout(time, ot, mean, message);
    time.clear();

    //now we begin measuring the individual components
    packages.clear();
    create_packages<Vector, Package>(packages, num);

    //third measures getting gid's from packages
    apply_get_gid<Vector>(packages, num, ot);

    if(empty){
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef 
            typename hpx::actions::extract_action<action_type>::result_type
            result_type;
        typedef hpx::actions::base_lco_continuation<result_type> rc_type;
        vector<rc_type*> continuations;

        //measures creation time of continuations
        apply_create_continuations<Vector, vector<rc_type*>, result_type>
            (packages, continuations, num, ot);

        //finally apply the continuations directly
        high_resolution_timer tt;
        for(i = 0; i < num; i++) 
            hpx::apply<action_type>(lid, a1, a2, a3);
        mean = tt.elapsed()/num;
        time.reserve(num);
        message = "Measuring time required to apply continuations directly:";

        for(i = 0; i < num; i++){
            high_resolution_timer t1;
            hpx::apply<action_type>(lid, a1, a2, a3);
            time.push_back(t1.elapsed());
        }
        printout(time, ot, mean, message);
    }
    else{
        //next test applies actions directly, skipping get_gid stage
        high_resolution_timer tt;
        for(i = 0; i < num; ++i)
            hpx::apply<Action>(lid, a1, a2, a3);
        mean = tt.elapsed()/num;
        time.reserve(num);
        message = "Measuring time required to apply actions directly:";
        for(i = 0; i < num; i++){
            high_resolution_timer t1;
            hpx::apply<Action>(lid, a1, a2, a3);
            time.push_back(t1.elapsed());
        }
        printout(time, ot, mean, message);
    }
}
template <typename Vector, typename Package, typename Action, typename T>
void run_tests(bool empty, uint64_t num, T a1, T a2, T a3, T a4){
    uint64_t i = 0;
    double ot = timer_overhead(num);
    string message;
    vector<double> time;
    Vector packages;
    create_packages<Vector, Package>(packages, num);
    hpx::naming::id_type lid = hpx::find_here();
    //first measure base case
    high_resolution_timer t;
    for(; i < num; ++i) packages[i]->apply(lid, a1, a2, a3, a4);
    double mean = t.elapsed()/num;
    packages.clear();
    create_packages<Vector, Package>(packages, num);
    time.reserve(num);
    for(i = 0; i < num; ++i){
        high_resolution_timer t1;
        packages[i]->apply(lid, a1, a2, a3, a4);
        time.push_back(t1.elapsed());
    }
    message = "Measuring time required to apply packaged_actions by calling "
              "packaged_action.apply():";
    printout(time, ot, mean, message);
    time.clear();

    //now we begin measuring the individual components
    packages.clear();
    create_packages<Vector, Package>(packages, num);

    //third measures getting gid's from packages
    apply_get_gid<Vector>(packages, num, ot);

    if(empty){
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef 
            typename hpx::actions::extract_action<action_type>::result_type
            result_type;
        typedef hpx::actions::base_lco_continuation<result_type> rc_type;
        vector<rc_type*> continuations;

        //measures creation time of continuations
        apply_create_continuations<Vector, vector<rc_type*>, result_type>
            (packages, continuations, num, ot);

        //finally apply the continuations directly
        high_resolution_timer tt;
        for(i = 0; i < num; i++) 
            hpx::apply<action_type>(lid, a1, a2, a3, a4);
        mean = tt.elapsed()/num;
        time.reserve(num);
        message = "Measuring time required to apply continuations directly:";

        for(i = 0; i < num; i++){
            high_resolution_timer t1;
            hpx::apply<action_type>(lid, a1, a2, a3, a4);
            time.push_back(t1.elapsed());
        }
        printout(time, ot, mean, message);
    }
    else{
        //next test applies actions directly, skipping get_gid stage
        high_resolution_timer tt;
        for(i = 0; i < num; ++i)
            hpx::apply<Action>(lid, a1, a2, a3, a4);
        mean = tt.elapsed()/num;
        time.reserve(num);
        message = "Measuring time required to apply actions directly:";
        for(i = 0; i < num; i++){
            high_resolution_timer t1;
            hpx::apply<Action>(lid, a1, a2, a3, a4);
            time.push_back(t1.elapsed());
        }
        printout(time, ot, mean, message);
    }
}
