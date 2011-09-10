//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/actions/continuation_impl.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>
#include <hpx/lcos/future_wait.hpp>

#include <cstdlib>
#include <ctime>

#include "sheneos/sheneos.hpp"

using boost::program_options::variables_map;
using boost::program_options::options_description;

char const* shen_symbolic_name = "/sheneos_test/test";

///////////////////////////////////////////////////////////////////////////////
// This is the test function, which will be invoked on all localities this 
// test is run on.
void test_sheneos(std::size_t num_test_points)
{
    // cubic root
    int num_points = std::exp(std::log(double(num_test_points)) / 3);

    // create a client instance connected to the already existing interpolation 
    // object
    sheneos::sheneos shen;
    shen.connect(shen_symbolic_name);

    double min_ye = 0, max_ye = 0;
    shen.get_dimension(sheneos::dimension::ye, min_ye, max_ye);
    double delta_ye = (max_ye - min_ye) / num_points;

    double min_temp = 0, max_temp = 0;
    shen.get_dimension(sheneos::dimension::temp, min_temp, max_temp);
    double delta_temp = (max_temp - min_temp) / num_points;

    double min_rho = 0, max_rho = 0;
    shen.get_dimension(sheneos::dimension::rho, min_rho, max_rho);
    double delta_rho = (max_rho - min_rho) / num_points;

    // Do the required amount of test calls to the interpolator, using equally 
    // spaced data points. We want to avoid that the evaluation sequence of the
    // data points will be the same on all localities. For that reason we first
    // generate vector's of the values, and a sequence vector, which will be 
    // randomly initialized.
    std::vector<double> values_ye(num_points);
    std::vector<std::size_t> sequence_ye(num_points);
    double ye = min_ye;
    for (std::size_t i = 0; i < num_points; ++i) {
        values_ye[i] = ye;
        sequence_ye[i] = i;
        ye += delta_ye;
    }

    std::vector<double> values_temp(num_points);
    std::vector<std::size_t> sequence_temp(num_points);
    double temp = min_temp;
    for (std::size_t i = 0; i < num_points; ++i) {
        values_temp[i] = temp;
        sequence_temp[i] = i;
        temp += delta_temp;
    }

    std::vector<double> values_rho(num_points);
    std::vector<std::size_t> sequence_rho(num_points);
    double rho = min_rho;
    for (std::size_t i = 0; i < num_points; ++i) {
        values_rho[i] = rho;
        sequence_rho[i] = i;
        rho += delta_rho;
    }

    // randomize the sequences
    std::srand(std::time(0));
    std::random_shuffle(sequence_ye.begin(), sequence_ye.end());
    std::random_shuffle(sequence_temp.begin(), sequence_temp.end());
    std::random_shuffle(sequence_rho.begin(), sequence_rho.end());

    // now, do the actual evaluation, all in parallel
    std::vector<hpx::lcos::future_value<std::vector<double> > > tests;
    for (std::size_t i = 0; i < sequence_ye.size(); ++i)
    {
        std::size_t ii = sequence_ye[i];
        for (std::size_t j = 0; j < sequence_temp.size(); ++j)
        {
            std::size_t jj = sequence_temp[j];
            for (std::size_t k = 0; k < sequence_rho.size(); ++k)
            {
                std::size_t kk = sequence_rho[k];
                tests.push_back(shen.interpolate_async(
                    values_ye[ii], values_temp[jj], values_rho[kk]));
            }
        }
    }
    hpx::lcos::wait(tests);   // wait for all of the tests to finish
}

typedef hpx::actions::plain_action1<std::size_t, test_sheneos> test_action;

HPX_REGISTER_PLAIN_ACTION(test_action);

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    std::string datafilename("myshen_test_220r_180t_50y_extT_analmu_20100322_SVNr28.h5");
    std::size_t num_localities = 27;
    std::size_t num_test_points = 10000;

    if (vm.count("num_tests"))
        num_test_points = vm["num_tests"].as<std::size_t>();

    {
        hpx::util::high_resolution_timer t;

        // create the distributed interpolation object with 'num_localities'
        // partitions
        sheneos::sheneos shen;
        shen.create(datafilename, shen_symbolic_name, num_localities);

        double elapsed = t.elapsed();
        std::cout << "Create interpolator: " << elapsed << " [s]" << std::endl;

        // get list of locality prefixes
        using hpx::components::server::plain_function;
        hpx::components::component_type type = 
            plain_function<test_action>::get_component_type();

        std::vector<hpx::naming::id_type> prefixes =
            hpx::find_all_localities(type);

        // execute tests on all localities
        t.restart();

        std::vector<hpx::lcos::future_value<void> > tests;
        BOOST_FOREACH(hpx::naming::id_type const& id, prefixes)
        {
            hpx::lcos::eager_future<test_action, void> f(id, num_test_points);
            tests.push_back(f);
        }
        hpx::lcos::wait(tests);   // wait for tests to run

        elapsed = t.elapsed();
        std::cout << "Running tests: " << elapsed << " [s]" << std::endl;
    }

    hpx::finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    namespace po = boost::program_options;

    // Configure application-specific options
    po::options_description cmdline("Usage: " HPX_APPLICATION_STRING " [options]");
    cmdline.add_options()
        ("num_tests,n", po::value<std::size_t>(), 
            "number of data points to interpolate (default: 10000)")
    ;

    // Initialize and run HPX
    return hpx::init(cmdline, argc, argv);
}

