//  Copyright (c) 2007-2012 Hartmut Kaiser
//  2012 Matt Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <omp.h>
#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/inlcude/util.hpp>

#include <boost/dynamic_bitset.hpp>

#include <cstdlib>
#include <ctime>
#include <string>

#include "sheneos/interpolator.hpp"
#include "fname.h"

int omp_thread_count() {
    int n = 0;
    #pragma omp parallel reduction(+:n)
    n += 1;
    return n;
}

extern "C" {void FNAME(read_nuc_table)(); }

extern "C" {void FNAME(nuc_eos_short)(double *xrho,double *xtemp,double *xye,
                                      double *xenr,
                                      double *xprs,double *xent,double *xcs2,
                                      double *xdedt,
                                      double *xdpderho,double *xdpdrhoe,double *xmunu,
                                      int *keytemp,int *keyerr,double *rfeps); }

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

    std::size_t nthreads = omp_thread_count();
    //std::size_t nthreads = omp_get_num_threads();
    std::cout << " Number of OMP threads " << nthreads << std::endl;
    std::cout << " Problem Size: Ye " << sequence_ye.size() << " T "
        << sequence_temp.size() << " R " << sequence_rho.size()  << std::endl;

    ///////////////////////////////////////////////////////////////////////////
    // We want to avoid invoking the same evaluation sequence on all localities
    // performing the test, so we randomly shuffle the sequences. We combine
    // the shared seed with the locality id to ensure that each locality has
    // a unique, reproducible seed.
#pragma omp parallel
    std::cout << " HELLO WORLD from thread " << omp_get_thread_num() << std::endl;
    std::srand(static_cast<unsigned int>(omp_get_thread_num() + hpx::get_locality_id()));
    std::random_shuffle(sequence_ye.begin(), sequence_ye.end());
    std::random_shuffle(sequence_temp.begin(), sequence_temp.end());
    std::random_shuffle(sequence_rho.begin(), sequence_rho.end());

    // Create the three-dimensional future grid.
    //std::vector<hpx::lcos::future<std::vector<double> > > tests;
    for (std::size_t i = 0; i < sequence_ye.size(); ++i)
    {
        std::size_t const& ii = sequence_ye[i];
        for (std::size_t j = 0; j < sequence_temp.size(); ++j)
        {
            std::size_t const& jj = sequence_temp[j];
            for (std::size_t k = 0; k < sequence_rho.size(); ++k)
            {
                std::size_t const& kk = sequence_rho[k];
                double xrho = values_rho[kk];
                double xye = values_ye[ii];
                double xtemp = values_temp[jj];
                double rfeps = 1.0e-5;
                double xenr = 0.0;
                double xent = 0.0;
                double xprs = 0.0;
                double xmunu = 0.0;
                double xcs2 = 0.0;
                double xdedt = 0.0;
                double xdpderho = 0.0;
                double xdpdrhoe = 0.0;
                int keyerr = -666;
                int keytemp = 1;
                FNAME(nuc_eos_short)(&xrho,&xtemp,&xye,&xenr,&xprs,&xent,&xcs2,&xdedt,
                                     &xdpderho,&xdpdrhoe,&xmunu,&keytemp,&keyerr,&rfeps);
            }
        }
    }
}

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

    // Execute bulk operation
    for (std::size_t i = 0; i < sequence_ye.size(); ++i)
    {
        std::size_t const& ii = sequence_ye[i];
        for (std::size_t j = 0; j < sequence_temp.size(); ++j)
        {
            std::size_t const& jj = sequence_temp[j];
            for (std::size_t k = 0; k < sequence_rho.size(); ++k)
            {
                std::size_t const& kk = sequence_rho[k];
                double xrho = values_rho[kk];
                double xye = values_ye[ii];
                double xtemp = values_temp[jj];
                double rfeps = 1.0e-5;
                double xenr = 0.0;
                double xent = 0.0;
                double xprs = 0.0;
                double xmunu = 0.0;
                double xcs2 = 0.0;
                double xdedt = 0.0;
                double xdpderho = 0.0;
                double xdpdrhoe = 0.0;
                int keyerr = -666;
                int keytemp = 1;
                FNAME(nuc_eos_short)(&xrho,&xtemp,&xye,&xenr,&xprs,&xent,&xcs2,&xdedt,
                                     &xdpderho,&xdpdrhoe,&xmunu,&keytemp,&keyerr,&rfeps);
            }
        }
    }

    //hpx::lcos::future<std::vector<double> > bulk_one_tests =
    //    shen.interpolate_one_bulk_async(values_ye, values_temp, values_rho,
    //        sheneos::server::partition3d::logpress);

    //std::vector<double> results = hpx::util::unwrapped(bulk_one_tests);
}

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

    // Execute bulk operation
    //hpx::lcos::future<std::vector<std::vector<double> > > bulk_tests =
    //    shen.interpolate_bulk_async(values_ye, values_temp, values_rho);

    for (std::size_t i = 0; i < sequence_ye.size(); ++i)
    {
        std::size_t const& ii = sequence_ye[i];
        for (std::size_t j = 0; j < sequence_temp.size(); ++j)
        {
            std::size_t const& jj = sequence_temp[j];
            for (std::size_t k = 0; k < sequence_rho.size(); ++k)
            {
                std::size_t const& kk = sequence_rho[k];
                double xrho = values_rho[kk];
                double xye = values_ye[ii];
                double xtemp = values_temp[jj];
                double rfeps = 1.0e-5;
                double xenr = 0.0;
                double xent = 0.0;
                double xprs = 0.0;
                double xmunu = 0.0;
                double xcs2 = 0.0;
                double xdedt = 0.0;
                double xdpderho = 0.0;
                double xdpdrhoe = 0.0;
                int keyerr = -666;
                int keytemp = 1;
                FNAME(nuc_eos_short)(&xrho,&xtemp,&xye,&xenr,&xprs,&xent,&xcs2,&xdedt,
                                     &xdpderho,&xdpdrhoe,&xmunu,&keytemp,&keyerr,&rfeps);
            }
        }
    }
    //std::vector<std::vector<double> > results = hpx::util::unwrapped(bulk_tests);
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    std::string const datafilename = vm["file"].as<std::string>();

    std::size_t const num_ye_points = vm["num-ye-points"].as<std::size_t>();
    std::size_t const num_temp_points = vm["num-temp-points"].as<std::size_t>();
    std::size_t const num_rho_points = vm["num-rho-points"].as<std::size_t>();

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
        FNAME(read_nuc_table)();
        std::cout << "Created interpolator: " << t.elapsed() << " [s]"
                  << std::endl;

        // Get the component type of the test_action. A plain action is actually
        // a component action of the special plain_function component.
        //using hpx::components::server::plain_function;
        //hpx::components::component_type type =
        //    plain_function<test_action>::get_component_type();

        // Get a list of all localities that support the test action.
        //std::vector<hpx::naming::id_type> locality_ids =
        //    hpx::find_all_localities(type);

        t.restart();

        test_sheneos(num_ye_points,num_temp_points,num_rho_points,seed);
        std::cout << "Control case: " << t.elapsed() << " [s]" << std::endl;
#if 0
        // Kick off the computation asynchronously. On each locality,
        // num_workers test_actions are created.
        std::vector<hpx::lcos::future<void> > tests;
        for (hpx::naming::id_type const& id : locality_ids)
        {
            using hpx::async;
            for (std::size_t i = 0; i < num_workers; ++i)
                tests.push_back(async<test_action>(id, num_ye_points,
                    num_temp_points, num_rho_points, seed));
        }

        hpx::lcos::wait(tests,
            [&](std::size_t i)
            {
                std::cout << "Finished task " << i << ": " << t.elapsed() << " [s]"
                          << std::endl;
            });

        std::cout << "Completed tests: " << t.elapsed() << " [s]" << std::endl;

        t.restart();

        // Kick off the computation asynchronously. On each locality,
        // num_workers test_actions are created.
        std::vector<hpx::lcos::future<void> > bulk_one_tests;
        for (hpx::naming::id_type const& id : locality_ids)
        {
            using hpx::async;
            for (std::size_t i = 0; i < num_workers; ++i)
                bulk_one_tests.push_back(async<test_one_bulk_action>(id,
                    num_ye_points, num_temp_points, num_rho_points, seed));
        }

        hpx::lcos::wait(bulk_one_tests,
            [&](std::size_t i)
            {
                std::cout << "Finished bulk-one task " << i << ": " << t.elapsed()
                    << " [s]" << std::endl;
            });

        std::cout << "Completed bulk-one tests: " << t.elapsed() << " [s]"
            << std::endl;

        t.restart();

        // Kick off the computation asynchronously. On each locality,
        // num_workers test_actions are created.
        std::vector<hpx::lcos::future<void> > bulk_tests;
        for (hpx::naming::id_type const& id : locality_ids)
        {
            using hpx::async;
            for (std::size_t i = 0; i < num_workers; ++i)
                bulk_tests.push_back(async<test_bulk_action>(id,
                    num_ye_points, num_temp_points, num_rho_points, seed));
        }

        hpx::lcos::wait(bulk_tests,
            [&](std::size_t i)
            {
                std::cout << "Finished bulk task " << i << ": " << t.elapsed()
                    << " [s]" << std::endl;
            });

        std::cout << "Completed bulk tests: " << t.elapsed() << " [s]"
            << std::endl;
#endif
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

