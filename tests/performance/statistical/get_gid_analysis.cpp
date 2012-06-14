//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "simple_declarations.cpp"
#include "statstd.cpp"

///////////////////////////////////////////////////////////////////////////////
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
void run_tests(uint64_t);

///////////////////////////////////////////////////////////////////////////////
//all of the measured tests are declared in this section

//measure how long it takes to obtain gid from a packaged_action 
template <typename Vector>
void test_get_gid(Vector packages, uint64_t num, double ot){
    uint64_t i = 0;
    double mean;
    string message = "Measuring total time required to get package gids:";
    vector<double> time;
    time.reserve(num);

    high_resolution_timer t;
    for(; i < num; ++i)
        packages[i]->get_gid();
    mean = t.elapsed()/num;
    for(i = 0; i < num; i++){
        high_resolution_timer t1;
        packages[i]->get_gid();
        time.push_back(t1.elapsed());
    }
    printout(time, ot, mean, message);
}

//measure how long it takes to run get_base_gid()
template <typename Vector>
void test_get_base_gid(Vector packages, uint64_t num, double ot){
    uint64_t i = 0;
    double mean;
    string message = "Measuring total time required to get base gids:";
    vector<double> time;
    time.reserve(num);

    high_resolution_timer t;
    for(; i < num; i++)
        packages[i]->get_base_gid();
    mean = t.elapsed()/num;
    for(i = 0; i < num; i++){
        high_resolution_timer t1;
        packages[i]->get_base_gid();
        time.push_back(t1.elapsed());
    }
    printout(time, ot, mean, message);
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm){
    uint64_t num = vm["number-spawned"].as<uint64_t>();
    csv = (vm.count("csv") ? true : false);
    run_tests<vector<void_package0*>, void_package0, void_action0, void> (num);
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
        ("csv",
            "output results as csv "
            "(format:count,mean,accurate mean,variance,min,max)");

    // Initialize and run HPX
    return hpx::init(desc_commandline, argc, argv);
}

///////////////////////////////////////////////////////////////////////////////

//this tests how long it takes to perform get_gid()
template <typename Vector, typename Package, typename Action, typename T>
void run_tests(uint64_t num){
    //uint64_t i = 0;
    double ot = timer_overhead(num);
    string message;
    vector<double> time;
    Vector packages;
    create_packages<Vector, Package>(packages, num);

    test_get_base_gid<Vector>(packages, num, ot);
    test_get_gid<Vector>(packages, num, ot);
}

