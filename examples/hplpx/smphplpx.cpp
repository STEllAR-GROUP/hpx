//  Copyright (c) 2011 Bryce Lelbach and Dan Kogler
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>

#include <iostream>
#include <vector>

#include <boost/format.hpp>

#include <examples/hplpx/smphplmatrex/smphplmatrex.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::init;
using hpx::finalize;

using hpx::components::smphplmatrex;

using hpx::naming::id_type;

using hpx::applier::get_applier;

/// This small program is used to perform LU decomposition on a randomly
/// generated matrix. Unlike the standard HPL algorithm, partial pivoting
/// is not implemented. This is for simplicity and takes into account
/// the fact that partial pivoting is rarely needed for randomly generated
/// matrices. Upon completion of the computations, the error of the
/// generated solution is calculated and displayed as both cumulative
/// error and average error per part of solution, as well as the total
/// execution time from just before starting the AGAS server to the
/// statistical printout.

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    unsigned int size(0), allocblock(0), blocksize(0);

    size = vm["size"].as<unsigned int>();
    allocblock = vm["allocblock"].as<unsigned int>();
    blocksize = vm["blocksize"].as<unsigned int>();
    {
        id_type prefix = get_applier().get_runtime_support_gid();

        smphplmatrex dat;

        dat.create(prefix);
        dat.construct(size, allocblock, blocksize);

        double r = dat.LUsolve();

        dat.free();

        std::cout << (boost::format("total error   : %1%\n"
                                    "average error : %2%\n")
                     % r % (r / size));
    }

    finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options.
    options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ( "size,S"
        , value<unsigned int>()->default_value(2048)
        , "the height of the NxN+1 matrix generated")

        ( "blocksize,B"
        , value<unsigned int>()->default_value(256)
        , "the amount of work performed by each pxthread during gaussian "
          "elimination")

        ( "allocblock,A"
        , value<unsigned int>()->default_value(512)
        , "amount of work each thread performs during memory allocation "
          "(must be a power of 2)")
        ;
    // Initialize and run HPX.
    return init(desc_commandline, argc, argv);
}

