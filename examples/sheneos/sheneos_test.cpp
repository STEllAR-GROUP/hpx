//  Copyright (c) 2007-2011 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/lcos.hpp>

#include <boost/dynamic_bitset.hpp>

#include <cstdlib>
#include <ctime>

#include "sheneos/interpolator.hpp"

char const* const shen_symbolic_name = "/sheneos/interpolator_test";

///////////////////////////////////////////////////////////////////////////////
/// This is the test function. It will be invoked on all localities that the 
/// benchmark is being run on.
void test_sheneos(std::size_t num_ye_points, std::size_t num_temp_points,
    std::size_t num_rho_points, std::size_t seed)
{
    // Create a client instance connected to the already existing interpolation
    // object.
    sheneos::interpolator shen;
    shen.connect(shen_symbolic_name);

    ///////////////////////////////////////////////////////////////////////////
    // Compute the minimum, maximum and delta values for each dimension.
    double min_ye = 0, max_ye = 0;
    shen.get_dimension(sheneos::dimension::ye, min_ye, max_ye);
    double const delta_ye = (max_ye - min_ye) / num_ye_points;

    double min_temp = 0, max_temp = 0;
    shen.get_dimension(sheneos::dimension::temp, min_temp, max_temp);
    double const delta_temp = (max_temp - min_temp) / num_temp_points;

    double min_rho = 0, max_rho = 0;
    shen.get_dimension(sheneos::dimension::rho, min_rho, max_rho);
    double const delta_rho = (max_rho - min_rho) / num_rho_points;

    ///////////////////////////////////////////////////////////////////////////
    // Generate the data points, spacing them out equally.
    std::vector<double> values_ye(num_ye_points);
    std::vector<std::size_t> sequence_ye(num_ye_points);
    double ye = min_ye;
    for (std::size_t i = 0; i < num_ye_points; ++i) {
        values_ye[i] = ye;
        sequence_ye[i] = i;
        ye += delta_ye;
    }

    std::vector<double> values_temp(num_temp_points);
    std::vector<std::size_t> sequence_temp(num_temp_points);
    double temp = min_temp;
    for (std::size_t i = 0; i < num_temp_points; ++i) {
        values_temp[i] = temp;
        sequence_temp[i] = i;
        temp += delta_temp;
    }

    std::vector<double> values_rho(num_rho_points);
    std::vector<std::size_t> sequence_rho(num_rho_points);
    double rho = min_rho;
    for (std::size_t i = 0; i < num_rho_points; ++i) {
        values_rho[i] = rho;
        sequence_rho[i] = i;
        rho += delta_rho;
    }

    ///////////////////////////////////////////////////////////////////////////
    // We want to avoid invoking the same evaluation sequence on all localities
    // performing the test, so we randomly shuffle the sequences. We combine
    // the shared seed with the locality id to ensure that each locality has
    // a unique, reproducible seed.
    std::srand(seed + hpx::applier::get_prefix_id());
    std::random_shuffle(sequence_ye.begin(), sequence_ye.end());
    std::random_shuffle(sequence_temp.begin(), sequence_temp.end());
    std::random_shuffle(sequence_rho.begin(), sequence_rho.end());

    // Create the three-dimensional future grid. 
    std::vector<hpx::lcos::promise<std::vector<double> > > tests;
    for (std::size_t i = 0; i < sequence_ye.size(); ++i)
    {
        std::size_t const& ii = sequence_ye[i];
        for (std::size_t j = 0; j < sequence_temp.size(); ++j)
        {
            std::size_t const& jj = sequence_temp[j];
            for (std::size_t k = 0; k < sequence_rho.size(); ++k)
            {
                std::size_t const& kk = sequence_rho[k];
                tests.push_back(shen.interpolate_async(
                    values_ye[ii], values_temp[jj], values_rho[kk]));
            }
        }
    }

    hpx::lcos::wait(tests);
}

typedef hpx::actions::plain_action4<
    std::size_t,
    std::size_t,
    std::size_t,
    std::size_t,
    test_sheneos
> test_action;

HPX_REGISTER_PLAIN_ACTION(test_action);

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    std::string const datafilename = vm["file"].as<std::string>();

    std::size_t const num_ye_points = vm["num-ye-points"].as<std::size_t>();
    std::size_t const num_temp_points = vm["num-ye-points"].as<std::size_t>();
    std::size_t const num_rho_points = vm["num-ye-points"].as<std::size_t>();

    std::size_t const num_partitions = vm["num-partitions"].as<std::size_t>();

    std::size_t num_workers = vm["num-workers"].as<std::size_t>();

    std::size_t seed = vm["seed"].as<std::size_t>();

    if (!seed)
        seed = std::size_t(std::time(0));

    std::cout << "Seed: " << seed << std::endl;

    {
        hpx::util::high_resolution_timer t;

        // Create a distributed interpolation object with the name
        // shen_symbolic_name. The interpolation object will have
        // num_partitions partitions
        sheneos::interpolator shen;
        shen.create(datafilename, shen_symbolic_name, num_partitions);

        std::cout << "Created interpolator: " << t.elapsed() << " [s]"
                  << std::endl;

        // Get the component type of the test_action. A plain action is actually
        // a component action of the special plain_function component.
        using hpx::components::server::plain_function;
        hpx::components::component_type type =
            plain_function<test_action>::get_component_type();

        // Get a list of all localities that support the test action. 
        std::vector<hpx::naming::id_type> prefixes =
            hpx::find_all_localities(type);

        t.restart();

        // Kick off the computation asynchronously. On each locality,
        // num_workers test_actions are created.
        std::vector<hpx::lcos::promise<void> > tests;
        BOOST_FOREACH(hpx::naming::id_type const& id, prefixes)
        {
            using hpx::lcos::async;
            for (std::size_t i = 0; i < num_workers; ++i)
                tests.push_back(async<test_action>(id, num_ye_points,
                    num_temp_points, num_rho_points, seed));
        }

        hpx::lcos::wait(tests, [&](std::size_t i) {
            std::cout << "Finished task " << i << ": " << t.elapsed() << " [s]"
                      << std::endl;
        });

        std::cout << "Completed tests: " << t.elapsed() << " [s]"
                  << std::endl;

    } // Ensure that everything is out of scope before shutdown.

    // Shutdown the runtime system.
    hpx::finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    using boost::program_options::options_description;
    using boost::program_options::value;

    // Configure application-specific options.
    options_description cmdline("Usage: " HPX_APPLICATION_STRING " [options]");

    cmdline.add_options()
        ("file", value<std::string>()->default_value(
                "sheneos_220r_180t_50y_extT_analmu_20100322_SVNr28.h5"),
            "name of HDF5 data file containing the Shen EOS tables")
        ("num-ye-points,YP", value<std::size_t>()->default_value(20),
            "number of points to interpolate on the ye axis")
        ("num-temp-points,TP", value<std::size_t>()->default_value(20),
            "number of points to interpolate on the temp axis")
        ("num-rho-points,RP", value<std::size_t>()->default_value(20),
            "number of points to interpolate on the rho axis")
        ("num-partitions", value<std::size_t>()->default_value(32),
            "number of partitions to create")
        ("num-workers", value<std::size_t>()->default_value(1),
            "number of worker/measurement threads to create")
        ("seed", value<std::size_t>()->default_value(0),
            "seed for the random shuffling of the queries (if 0, std::time(0) "
            "is used")
    ;

    // Initialize and run HPX.
    return hpx::init(cmdline, argc, argv);
}

