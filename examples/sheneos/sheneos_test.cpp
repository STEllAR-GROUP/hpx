//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/util.hpp>
#include <hpx/util/bind.hpp>

#include <boost/dynamic_bitset.hpp>

#include <cstdlib>
#include <ctime>
#include <string>
#include <vector>

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
    std::srand(static_cast<unsigned int>(seed + hpx::get_locality_id()));
    std::random_shuffle(sequence_ye.begin(), sequence_ye.end());
    std::random_shuffle(sequence_temp.begin(), sequence_temp.end());
    std::random_shuffle(sequence_rho.begin(), sequence_rho.end());

    // Create the three-dimensional future grid.
    std::vector<hpx::lcos::future<std::vector<double> > > tests;
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

    hpx::wait_all(tests);
}

HPX_PLAIN_ACTION(test_sheneos, test_action);
HPX_ACTION_USES_MEDIUM_STACK(test_action);

///////////////////////////////////////////////////////////////////////////////
/// This is the test function. It will be invoked on all localities that the
/// benchmark is being run on.
void test_sheneos_one_bulk(std::size_t num_ye_points,
    std::size_t num_temp_points, std::size_t num_rho_points, std::size_t seed)
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
    std::srand(static_cast<unsigned int>(seed + hpx::get_locality_id()));
    std::random_shuffle(sequence_ye.begin(), sequence_ye.end());
    std::random_shuffle(sequence_temp.begin(), sequence_temp.end());
    std::random_shuffle(sequence_rho.begin(), sequence_rho.end());

    // build the array of coordinates we want to get the interpolated values for
    std::vector<sheneos::sheneos_coord> values;
    values.reserve(num_ye_points * num_temp_points * num_rho_points);

    std::vector<hpx::lcos::future<std::vector<double> > > tests;
    for (std::size_t i = 0; i < sequence_ye.size(); ++i)
    {
        std::size_t const& ii = sequence_ye[i];
        for (std::size_t j = 0; j < sequence_temp.size(); ++j)
        {
            std::size_t const& jj = sequence_temp[j];
            for (std::size_t k = 0; k < sequence_rho.size(); ++k)
            {
                std::size_t const& kk = sequence_rho[k];
                values.push_back(sheneos::sheneos_coord(
                    values_ye[ii], values_temp[jj], values_rho[kk]));
            }
        }
    }

    // Execute bulk operation
    hpx::lcos::future<std::vector<double> > bulk_one_tests =
        shen.interpolate_one_bulk_async(values, sheneos::server::partition3d::logpress);

    std::vector<double> results = hpx::util::unwrapped(bulk_one_tests);
}

HPX_PLAIN_ACTION(test_sheneos_one_bulk, test_one_bulk_action);
HPX_ACTION_USES_MEDIUM_STACK(test_one_bulk_action);

///////////////////////////////////////////////////////////////////////////////
/// This is the test function for interpolate_bulk. It will be invoked on all
/// localities that the benchmark is being run on.
void test_sheneos_bulk(std::size_t num_ye_points,
    std::size_t num_temp_points, std::size_t num_rho_points, std::size_t seed)
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
    std::srand(static_cast<unsigned int>(seed + hpx::get_locality_id()));
    std::random_shuffle(sequence_ye.begin(), sequence_ye.end());
    std::random_shuffle(sequence_temp.begin(), sequence_temp.end());
    std::random_shuffle(sequence_rho.begin(), sequence_rho.end());

    // build the array of coordinates we want to get the interpolated values for
    std::vector<sheneos::sheneos_coord> values;
    values.reserve(num_ye_points * num_temp_points * num_rho_points);

    std::vector<hpx::lcos::future<std::vector<double> > > tests;
    for (std::size_t i = 0; i < sequence_ye.size(); ++i)
    {
        std::size_t const& ii = sequence_ye[i];
        for (std::size_t j = 0; j < sequence_temp.size(); ++j)
        {
            std::size_t const& jj = sequence_temp[j];
            for (std::size_t k = 0; k < sequence_rho.size(); ++k)
            {
                std::size_t const& kk = sequence_rho[k];
                values.push_back(sheneos::sheneos_coord(
                    values_ye[ii], values_temp[jj], values_rho[kk]));
            }
        }
    }

    // Execute bulk operation
    hpx::lcos::future<std::vector<std::vector<double> > > bulk_tests =
        shen.interpolate_bulk_async(values);

    std::vector<std::vector<double> > results = hpx::util::unwrapped(bulk_tests);
}

HPX_PLAIN_ACTION(test_sheneos_bulk, test_bulk_action);
HPX_ACTION_USES_MEDIUM_STACK(test_bulk_action);

///////////////////////////////////////////////////////////////////////////////
void wait_for_task(std::size_t i, hpx::util::high_resolution_timer& t)
{
    std::cout << "Finished task " << i << ": " << t.elapsed() << " [s]"
              << std::endl;
}

void wait_for_bulk_one_task(std::size_t i, hpx::util::high_resolution_timer& t)
{
    std::cout << "Finished bulk-one task " << i << ": " << t.elapsed()
        << " [s]" << std::endl;
}

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
        std::vector<hpx::naming::id_type> locality_ids =
            hpx::find_all_localities(type);

        t.restart();

//         // Kick off the computation asynchronously. On each locality,
//         // num_workers test_actions are created.
//         std::vector<hpx::lcos::future<void> > tests;
//         for (hpx::naming::id_type const& id : locality_ids)
//         {
//             using hpx::async;
//             for (std::size_t i = 0; i < num_workers; ++i)
//                 tests.push_back(async<test_action>(id, num_ye_points,
//                     num_temp_points, num_rho_points, seed));
//         }
//
//         hpx::lcos::wait(tests,
//             hpx::util::bind(wait_for_task, ::_1, boost::ref(t)));
//
//         std::cout << "Completed tests: " << t.elapsed() << " [s]" << std::endl;
//
//         t.restart();
//
//         // Kick off the computation asynchronously. On each locality,
//         // num_workers test_actions are created.
//         std::vector<hpx::lcos::future<void> > bulk_one_tests;
//         for (hpx::naming::id_type const& id : locality_ids)
//         {
//             using hpx::async;
//             for (std::size_t i = 0; i < num_workers; ++i)
//                 bulk_one_tests.push_back(async<test_one_bulk_action>(id,
//                     num_ye_points, num_temp_points, num_rho_points, seed));
//         }
//
//         hpx::lcos::wait(bulk_one_tests,
//             hpx::util::bind(wait_for_bulk_one_task, ::_1, boost::ref(t)));
//
//         std::cout << "Completed bulk-one tests: " << t.elapsed() << " [s]"
//             << std::endl;
//
//         t.restart();

        // Kick off the computation asynchronously. On each locality,
        // num_workers test_actions are created.
        std::vector<hpx::future<void> > bulk_tests;
        for (hpx::naming::id_type const& id : locality_ids)
        {
            for (std::size_t i = 0; i < num_workers; ++i)
            {
                bulk_tests.push_back(hpx::async<test_bulk_action>(id,
                    num_ye_points, num_temp_points, num_rho_points, seed));
            }
        }
        hpx::wait_all(bulk_tests);

        std::cout << "Completed bulk tests: " << t.elapsed() << " [s]"
            << std::endl;

    } // Ensure that everything is out of scope before shutdown.

    // Shutdown the runtime system.
    return hpx::finalize();
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
        ("num-ye-points,Y", value<std::size_t>()->default_value(40),
            "number of points to interpolate on the ye axis")
        ("num-temp-points,T", value<std::size_t>()->default_value(40),
            "number of points to interpolate on the temp axis")
        ("num-rho-points,R", value<std::size_t>()->default_value(40),
            "number of points to interpolate on the rho axis")
        ("num-partitions", value<std::size_t>()->default_value(32),
            "number of partitions to create")
        ("num-workers", value<std::size_t>()->default_value(1),
            "number of worker/measurement threads to create")
        ("seed", value<std::size_t>()->default_value(0),
            "the seed for the pseudo random number generator (if 0, a seed "
            "is chosen based on the current system time)")
    ;

    // Initialize and run HPX.
    return hpx::init(cmdline, argc, argv);
}

