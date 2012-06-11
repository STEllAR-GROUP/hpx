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

///////////////////////////////////////////////////////////////////////////////
//all of the measured tests are declared in this section

//measure how long it takes to obtain gid from a packaged_action 
template <typename Vector>
void test_get_gid(Vector packages, uint64_t num, double ot){
    uint64_t i = 0;
    string message = "Measuring total time required to get package gids:";
    vector<double> time;
    time.reserve(num);

    for(; i < num; i++){
        high_resolution_timer t1;
        packages[i]->get_gid();
        time.push_back(t1.elapsed());
    }
    printout(time, ot, message);
}

//measure how long it takes to run get_base_gid()
template <typename Vector>
void test_get_base_gid(Vector packages, uint64_t num, double ot){
    uint64_t i = 0;
    string message = "Measuring total time required to get base gids:";
    vector<double> time;
    time.reserve(num);

    for(; i < num; i++){
        high_resolution_timer t1;
        packages[i]->get_base_gid();
        time.push_back(t1.elapsed());
    }
    printout(time, ot, message);
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm){
    uint64_t num = vm["number-spawned"].as<uint64_t>();
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
            "number of packaged_actions created and run");

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
    create_packages<Vector, Package>(packages, num, ot);

    test_get_base_gid<Vector>(packages, num, ot);
    test_get_gid<Vector>(packages, num, ot);
}

