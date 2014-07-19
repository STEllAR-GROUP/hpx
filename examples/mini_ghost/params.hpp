//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_EXAMPLES_MINI_GHOST_PARAMS_HPP
#define HPX_EXAMPLES_MINI_GHOST_PARAMS_HPP

#include <hpx/hpx_finalize.hpp>
#include <boost/chrono.hpp>
#include <boost/program_options.hpp>

namespace mini_ghost {
    static const std::size_t SCALING_STRONG = 1;
    static const std::size_t SCALING_WEAK   = 2;

    template <typename Real>
    struct params
    {
        static float error_tolerance(float) { return 0.00001f; }
        static double error_tolerance(double) { return 0.00001; }

        params() :
            // Weak scaling is the default (1 for strong scaling)
            scaling(2)
            // Local x dimension (local for weak scaling, global when strong
            // scaling is selected.)
          , nx(100)
            // Local y dimension (local for weak scaling, global when strong
            // scaling is selected.)
          , ny(100)
            // Local z dimension (local for weak scaling, global when strong
            // scaling is selected.)
          , nz(100)
          , nx_block(100)
          , ny_block(100)
          , nz_block(10)
            // Number of variables
          , num_vars(5)
            // (Approximate) number of variables that needs to be summed.
            // This is intented to inject additional work (including a global
            // reduction, but it's not a correctness check.
          , percent_sum(0)
            // Number of source terms to be applied, one per max number of time
            // steps
          , num_spikes(1)
            // Number of time steps to be iterated
          , num_tsteps(100)
            // Stencil selected.
          , stencil(21)
            // Error tolerance
          , error_tol(error_tolerance(Real()))
            // Every report_diffusion time steps, report error. Note that if in
            // debug_grid mode, once the number of time steps exceeds half of the
            // minimum global dimension, an error is expected to occur as the
            // heat dissipates off of the domain.
          , report_diffusion(0)
            // Logical processor grid dimension in x direction
          , npx(1)
            // Logical processor grid dimension in y direction
          , npy(1)
            // Logical processor grid dimension in z direction
          , npz(1)
          , checkpoint_interval(0)
          , checkpoint_file("")
            // Amount of performance data reported
          , report_perf(0)
          , debug_grid(0)
        {}

        void cmd_options(boost::program_options::options_description & desc_commandline)
        {
            desc_commandline.add_options()
                ( "scaling",
                  boost::program_options::value<std::size_t>(&scaling)->default_value(2),
                  "Sets the scaling mode. possible Options: 1 or 2.\n"
                  "    1: Strong Scaling\n"
                  "    2: Weak Scaling");
            desc_commandline.add_options()
                ( "nx",
                  boost::program_options::value<std::size_t>()->default_value(100),
                  "Local x dimension (local for weak scaling, global when strong\n"
                  "scaling is selected.)");
            desc_commandline.add_options()
                ( "ny",
                  boost::program_options::value<std::size_t>()->default_value(100),
                  "Local y dimension (local for weak scaling, global when strong\n"
                  "scaling is selected.)");
            desc_commandline.add_options()
                ( "nz",
                  boost::program_options::value<std::size_t>()->default_value(100),
                  "Local z dimension (local for weak scaling, global when strong\n"
                  "scaling is selected.)");
            desc_commandline.add_options()
                ( "nx_block",
                  boost::program_options::value<std::size_t>(&nx_block)->default_value(100),
                  "Block size in x direction (default: 100)");
            desc_commandline.add_options()
                ( "ny_block",
                  boost::program_options::value<std::size_t>(&ny_block)->default_value(100),
                  "Block size in y direction (default: 100)");
            desc_commandline.add_options()
                ( "nz_block",
                  boost::program_options::value<std::size_t>(&nz_block)->default_value(10),
                  "Block size in z direction (default: 10)");
            desc_commandline.add_options()
                ( "ndim",
                  boost::program_options::value<std::size_t>(),
                  // FIXME: description
                  "");
            desc_commandline.add_options()
                ( "num_vars",
                  boost::program_options::value<std::size_t>(&num_vars)->default_value(5),
                  // FIXME: description
                  "");
            desc_commandline.add_options()
                ( "percent_sum",
                  boost::program_options::value<std::size_t>(&percent_sum)->default_value(0),
                  // FIXME: description
                  "");
            desc_commandline.add_options()
                ( "num_spikes",
                  boost::program_options::value<std::size_t>(&num_spikes)->default_value(1),
                  // FIXME: description
                  "");
            desc_commandline.add_options()
                ( "num_tsteps",
                  boost::program_options::value<std::size_t>(&num_tsteps)->default_value(100),
                  // FIXME: description
                  "");
            desc_commandline.add_options()
                ( "stencil",
                  boost::program_options::value<std::size_t>(&stencil)->default_value(21),
                  // FIXME: description
                  "");
            desc_commandline.add_options()
                ( "error_tol",
                  boost::program_options::value<Real>(&error_tol)->default_value(error_tolerance(Real())),
                  // FIXME: description
                  "");
            desc_commandline.add_options()
                ( "report_diffusion",
                  boost::program_options::value<std::size_t>(&report_diffusion)->default_value(0),
                  // FIXME: description
                  "");
            desc_commandline.add_options()
                ( "npx",
                  boost::program_options::value<std::size_t>(&npx)->default_value(1),
                  // FIXME: description
                  "");
            desc_commandline.add_options()
                ( "npy",
                  boost::program_options::value<std::size_t>(&npy)->default_value(1),
                  // FIXME: description
                  "");
            desc_commandline.add_options()
                ( "npz",
                  boost::program_options::value<std::size_t>(&npz)->default_value(1),
                  // FIXME: description
                  "");
            desc_commandline.add_options()
                ( "npdim",
                  boost::program_options::value<std::size_t>()->default_value(1),
                  // FIXME: description
                  "");
            desc_commandline.add_options()
                ( "checkpoint_interval",
                  boost::program_options::value<std::size_t>(&checkpoint_interval)->default_value(0),
                  // FIXME: description
                  "");
            desc_commandline.add_options()
                ( "checkpoint_file",
                  boost::program_options::value<std::string>(&checkpoint_file)->default_value(std::string("")),
                  // FIXME: description
                  "");
            desc_commandline.add_options()
                ( "report_perf",
                  boost::program_options::value<std::size_t>(&report_perf)->default_value(0),
                  // FIXME: description
                  "");
            desc_commandline.add_options()
                ( "debug_grid",
                  boost::program_options::value<std::size_t>(&debug_grid)->default_value(0),
                  // FIXME: description
                  "");
        }

        void setup(boost::program_options::variables_map& vm)
        {
            std::size_t tmp_nx = 0;
            std::size_t tmp_ny = 0;
            std::size_t tmp_nz = 0;
            if(vm["ndim"].empty())
            {
                tmp_nx = vm["nx"].as<std::size_t>();
                tmp_ny = vm["ny"].as<std::size_t>();
                tmp_nz = vm["nz"].as<std::size_t>();
            }
            else
            {
                tmp_nx = vm["ndim"].as<std::size_t>();
                tmp_ny = tmp_nx;
                tmp_nz = tmp_nx;
            }
            std::size_t remainder = 0;

            if(nranks > 1 && npx == 1 && npy == 1 && npz == 1)
            {
                npx = nranks;
            }

            switch (scaling)
            {
                case SCALING_WEAK:
                    nx = tmp_nx;
                    ny = tmp_ny;
                    nz = tmp_nz;
                    break;
                case SCALING_STRONG:
                    nx = tmp_nx / npx;
                    remainder = tmp_nx & npx;
                    if(rank < remainder)
                        nx++;

                    ny = tmp_ny / npy;
                    remainder = tmp_ny & npy;
                    if(rank < remainder)
                        ny++;

                    nz = tmp_nz / npz;
                    remainder = tmp_nz & npz;
                    if(rank < remainder)
                        nz++;
                    break;
                default:
                    std::cerr << "** Error ** Unknown scaling " << scaling << "; "
                              << "options are weak (" << SCALING_WEAK << ") "
                              << "and strong (" << SCALING_STRONG << ")"
                              << std::endl;
                    hpx::terminate();
            }

            if(report_diffusion == 0)
            {
                report_diffusion = num_tsteps;
            }

            if(debug_grid == 1)
            {
                percent_sum = 100;
            }

            // Validate if we got the correct settings ...
            if(num_tsteps < 1)
            {
                std::cerr << "[rank " << rank << "] ** Input error **: "
                          << "num_tsteps " << num_tsteps << " < 1."
                          << std::endl;
                hpx::terminate();
            }
            if(npx < 1)
            {
                std::cerr << "[rank " << rank << "] ** Input error **: "
                          << "npx " << npx << " < 1."
                          << std::endl;
                hpx::terminate();
            }
            if(npy < 1)
            {
                std::cerr << "[rank " << rank << "] ** Input error **: "
                          << "npy " << npy << " < 1."
                          << std::endl;
                hpx::terminate();
            }
            if(npz < 1)
            {
                std::cerr << "[rank " << rank << "] ** Input error **: "
                          << "npz " << npz << " < 1."
                          << std::endl;
                hpx::terminate();
            }
            if(npx*npy*npz != nranks)
            {
                std::cerr << "[rank " << rank << "] ** Input error **: "
                          << "npx*npy*npz not equal to number of ranks;"
                          << "(npx, npy, npz) = (" << npx << ", " << npy << ", " << npz << ") "
                          << "nranks = " << nranks
                          << std::endl;
                hpx::terminate();
            }
            if(percent_sum > 100)
            {
                std::cerr << "[rank " << rank << "] ** Input error **: "
                          << "percent_sum " << percent_sum << " > 100."
                          << std::endl;
                hpx::terminate();
            }
        }

        void print_header(std::size_t num_sum_grid)
        {
            std::cout
            << "\n"
            << "======================================================================\n"
            << "\n"
            << "        Mantevo miniapp MiniGhost experiment\n"
            << "        HPX port\n"
            << "\n"
            << "======================================================================\n"
            << "\n";
            switch (stencil)
            {
                case 20: //STENCIL_NONE:
                    std::cout << "No computation inserted\n";
                    break;
                case 21: //STENCIL_2D5PT:
                    std::cout << "Computation: 5 pt difference stencil on a 2D grid (STENCIL_2D5PT)\n";
                    break;
                case 22: //STENCIL_2D9PT:
                    std::cout << "Computation: 9 pt difference stencil on a 2D grid (STENCIL_2D9PT)\n";
                    break;
                case 23: //STENCIL_3D7PT:
                    std::cout << "Computation: 7 pt difference stencil on a 3D grid (STENCIL_3D27PT)\n";
                    break;
                case 24: //STENCIL_3D27PT:
                    std::cout << "Computation: 27 pt difference stencil on a 3D grid (STENCIL_3D27PT)\n";
                    break;
                default:
                    std::cout << "** Warning ** Unkown compuation\n";
                    break;
            }
            std::cout << std::endl;

            std::cout << "        Global Grid Dimension: "
                << nx * npx << ", " << ny * npy << ", " << nz * npz << "\n";
            std::cout << "        Local Grid Dimension : "
                << nx << ", " << ny << ", " << nz << "\n";
            std::cout << std::endl;

            std::cout << "Number of variables: " << num_vars << "\n";
            std::cout << std::endl;

            std::cout
                << "Error reported every " << report_diffusion << " time steps. "
                << "Tolerance is " << error_tol << "\n";
            std::cout
                << "Number of variables reduced each time step: " << num_sum_grid
                << "; requested " << percent_sum << "%\n";
            std::cout << std::endl;

            std::cout << "        Time Steps: " << num_tsteps << "\n";
            std::cout << "        Task grid : " << npx << ", " << npy << ", " << npz << "\n";
            std::cout << std::endl;

            switch (scaling)
            {
                case SCALING_WEAK:
                    std::cout << "HPX version, weak scaling\n";
                    break;
                case SCALING_STRONG:
                    std::cout << "HPX version, weak scaling\n";
                    break;
                default:
                    std::cout << "HPX version, unkown scaling\n";
                    hpx::terminate();
                    break;
            }

            if(nranks == 1)
            {
                std::cout << "1 process executing\n";
            }
            else
            {
                std::cout << nranks << " processes executing\n";
            }
            std::cout << std::endl;

            boost::chrono::system_clock::time_point const now = 
                boost::chrono::system_clock::now();
            std::cout << "Program execution date " << now << "\n";
            std::cout << std::endl;
        }

        std::size_t scaling;
        std::size_t nx;
        std::size_t ny;
        std::size_t nz;
        std::size_t nx_block;
        std::size_t ny_block;
        std::size_t nz_block;
        std::size_t num_vars;
        std::size_t percent_sum;
        std::size_t num_spikes;
        std::size_t num_tsteps;
        std::size_t stencil;
        Real error_tol;
        std::size_t report_diffusion;
        std::size_t npx;
        std::size_t npy;
        std::size_t npz;
        std::size_t checkpoint_interval;
        std::string checkpoint_file;
        std::size_t report_perf;
        std::size_t debug_grid;
        std::size_t rank;
        std::size_t nranks;
    };
}

#endif
