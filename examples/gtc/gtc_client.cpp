//  Copyright (c) 2007-2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/iostreams.hpp>

#include "point/point.hpp"
#include "particle/particle.hpp"
#include "parameter.hpp"

#include <hpx/components/distributing_factory/distributing_factory.hpp>

using boost::lexical_cast;

using hpx::bad_parameter;

using hpx::applier::get_applier;
using hpx::naming::id_type;

using hpx::util::section;
using hpx::util::high_resolution_timer;

using hpx::components::gtc::parameter;

/// This function initializes a vector of \a gtc::point clients, 
/// connecting them to components created with
/// \a hpx::components::distributing_factory.
inline void
init(hpx::components::server::distributing_factory::iterator_range_type const& r,
    std::vector<gtc::point>& point)
{
    BOOST_FOREACH(hpx::naming::id_type const& id, r)
    {
        point.push_back(gtc::point(id));
    }
}

/// This function initializes a vector of \a gtc::particle clients, 
/// connecting them to components created with
/// \a hpx::components::distributing_factory.
inline void
init(hpx::components::server::distributing_factory::iterator_range_type const& r,
    std::vector<gtc::particle>& particle)
{
    BOOST_FOREACH(hpx::naming::id_type const& id, r)
    {
        particle.push_back(gtc::particle(id));
    }
}

void appconfig_option(std::string const& name, section const& pars,
                      std::string& data)
{
    if (pars.has_entry(name))
        data = pars.get_entry(name);
}

template <typename T>
void appconfig_option(std::string const& name, section const& pars, T& data)
{
    try {
        if (pars.has_entry(name))
            data = lexical_cast<T>(pars.get_entry(name));
    } catch (...) {
        std::string msg = boost::str(boost::format(
            "\"%1%\" is not a valid value for %2%")
            % pars.get_entry(name) % name);
        HPX_THROW_IN_CURRENT_FUNC(bad_parameter, msg);
    }
}


///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map &vm)
{
    {
        // Start a high resolution timer to record the execution time of this
        // example.
        hpx::util::high_resolution_timer t;

        parameter par;

        // Default parameters
        par->irun = false;
        par->mstep = 1500;
        par->msnap = 1;
        par->ndiag = 4;
        par->nonlinear = 1.0;
        par->nhybrid = 0;
        par->paranl = 0.0;
        par->mode00 = true;

        par->tstep = 0.2;
        par->micell = 2;
        par->mecell = 2;
        par->mpsi = 90;
        par->mthetamax = 640;
        par->mzetamax = 64;
        par->npartdom = 1;
        par->ncycle = 5;

        par->a = 0.358;
        par->a0 = 0.1;
        par->a1 = 0.9;
        par->q0 = 0.854;
        par->q1 = 0.0;
        par->q2 = 2.184;
        par->rc = 0.5;
        par->rw = 0.35;

        par->aion = 1.0;
        par->qion = 1.0;
        par->aelectron = 1.0/1837.0;
        par->qelectron = -1.0;

        par->kappati = 6.9;
        par->kappate = 6.9;
        par->fixed_Tprofile = 1;
        par->tite = 1.0;
        par->flow0 = 0.0;
        par->flow1 = 0.0;
        par->flow2 = 0.0;

        par->r0 = 93.4;
        par->b0 = 19100.0;
        par->temperature = 2500.0;
        par->edensity0 = 0.46e14;

        par->output = 6;
        par->nbound = true;
        par->umax = 4.0;
        par->iload = false;
        par->tauii = -1.0;
        par->track_particles = false;
        par->nptrack = false;
        par->rng_control = false;

        par->nmode.push_back(5);
        par->nmode.push_back(7);
        par->nmode.push_back(9);
        par->nmode.push_back(11);
        par->nmode.push_back(13);
        par->nmode.push_back(15);
        par->nmode.push_back(18);
        par->nmode.push_back(20);

        par->mmode.push_back(7);
        par->mmode.push_back(10);
        par->mmode.push_back(13);
        par->mmode.push_back(15);
        par->mmode.push_back(18);
        par->mmode.push_back(21);
        par->mmode.push_back(25);
        par->mmode.push_back(28);

        // number of "processors"
        par->numberpe = 40;

        id_type rt_id = get_applier().get_runtime_support_gid();

        section root;
        hpx::components::stubs::runtime_support::get_config(rt_id, root);
        if (root.has_section("gtc"))
        { 
          section pars = *(root.get_section("gtc"));
           
          appconfig_option<bool>("irun", pars, par->irun);
          appconfig_option<std::size_t>("mstep", pars, par->mstep);
          appconfig_option<std::size_t>("msnap", pars, par->msnap);
          appconfig_option<std::size_t>("ndiag", pars, par->ndiag);
          appconfig_option<double>("nonlinear", pars, par->nonlinear);
          appconfig_option<std::size_t>("nhybrid", pars, par->nhybrid);
          appconfig_option<double>("paranl", pars, par->paranl);
          appconfig_option<bool>("mode00", pars, par->mode00);

          appconfig_option<double>("tstep", pars, par->tstep);
          appconfig_option<std::size_t>("micell", pars, par->micell);
          appconfig_option<std::size_t>("mecell", pars, par->mecell);
          appconfig_option<std::size_t>("mpsi", pars, par->mpsi);
          appconfig_option<std::size_t>("mthetamax", pars, par->mthetamax);
          appconfig_option<std::size_t>("mzetamax", pars, par->mzetamax);
          appconfig_option<std::size_t>("npartdom", pars, par->npartdom);
          appconfig_option<std::size_t>("ncycle", pars, par->ncycle);
          appconfig_option<double>("a", pars, par->a);
          appconfig_option<double>("a0", pars, par->a0);
          appconfig_option<double>("a1", pars, par->a1);
          appconfig_option<double>("q0", pars, par->q0);
          appconfig_option<double>("q1", pars, par->q1);
          appconfig_option<double>("q2", pars, par->q2);
          appconfig_option<double>("rc", pars, par->rc);
          appconfig_option<double>("rw", pars, par->rw);
          appconfig_option<double>("aion", pars, par->aion);
          appconfig_option<double>("qion", pars, par->qion);
          appconfig_option<double>("aelectron", pars, par->aelectron);
          appconfig_option<double>("qelectron", pars, par->qelectron);
          appconfig_option<double>("kappati", pars, par->kappati);
          appconfig_option<double>("kappate", pars, par->kappate);
          appconfig_option<bool>("fixed_Tprofile", pars, par->fixed_Tprofile);
          appconfig_option<double>("tite", pars, par->tite);
          appconfig_option<double>("flow0", pars, par->flow0);
          appconfig_option<double>("flow1", pars, par->flow1);
          appconfig_option<double>("flow2", pars, par->flow2);
          appconfig_option<double>("r0", pars, par->r0);
          appconfig_option<double>("b0", pars, par->b0);
          appconfig_option<double>("temperature", pars, par->temperature);
          appconfig_option<double>("edensity0", pars, par->edensity0);
          appconfig_option<std::size_t>("output", pars, par->output);
          appconfig_option<bool>("nbound", pars, par->nbound);
          appconfig_option<double>("umax", pars, par->umax);
          appconfig_option<bool>("iload", pars, par->iload);
          appconfig_option<double>("tauii", pars, par->tauii);
          appconfig_option<bool>("track_particles", pars, par->track_particles);
          appconfig_option<bool>("nptrack", pars, par->nptrack);
          appconfig_option<bool>("rng_control", pars, par->rng_control);
          appconfig_option<std::size_t>("numberpe", pars, par->numberpe);
        }

        // Derived parameters
        par->kappan = par->kappati*0.319;

        // Changing the units of a0 and a1 from units of "a" to units of "R_0"
        par->a0 = par->a0*par->a;
        par->a1 = par->a1*par->a;

        // ----- First we verify the consistency of ntoroidal and npartdom ------
        // The number of toroidal domains (ntoroidal) times the number of particle
        // "domains" (npartdom) needs to be equal to the number of processor "numberpe".
        // numberpe must be a multiple of npartdom so change npartdom accordingly
        while ( (par->numberpe%par->npartdom) != 0 ) {
          par->npartdom -= 1;
          if ( par->npartdom == 1 ) break;
        } 
        par->ntoroidal = par->numberpe/par->npartdom;

        // make sure that mzetamax is a multiple of ntoroidal
        double tmp1 = (double) par->mzetamax;
        double tmp2 = (double) par->ntoroidal;
        int tmp = tmp1/tmp2 + 0.5;
        par->mzetamax = par->ntoroidal*std::max(1,tmp);

        // ensure that "mpsi", the total number of flux surfaces, is an even
        // number since this quantity will be used in Fast Fourier Transforms
        par->mpsi = 2*(par->mpsi/2);

        hpx::cout << ( boost::format("GTC parameters \n")  ) << hpx::flush;
        hpx::cout << ( boost::format("----------------------------\n")  ) << hpx::flush;
        hpx::cout << ( boost::format("irun           : %1%\n") % par->irun) << hpx::flush;
        hpx::cout << ( boost::format("mstep          : %1%\n") % par->mstep) << hpx::flush;
        hpx::cout << ( boost::format("msnap          : %1%\n") % par->msnap) << hpx::flush;
        hpx::cout << ( boost::format("ndiag          : %1%\n") % par->ndiag) << hpx::flush;
        hpx::cout << ( boost::format("nonlinear      : %1%\n") % par->nonlinear) << hpx::flush;
        hpx::cout << ( boost::format("nhybrid        : %1%\n") % par->nhybrid) << hpx::flush;
        hpx::cout << ( boost::format("paranl         : %1%\n") % par->paranl) << hpx::flush;
        hpx::cout << ( boost::format("mode00         : %1%\n") % par->mode00) << hpx::flush;
        hpx::cout << ( boost::format("tstep          : %1%\n") % par->tstep) << hpx::flush;
        hpx::cout << ( boost::format("micell         : %1%\n") % par->micell) << hpx::flush;
        hpx::cout << ( boost::format("mecell         : %1%\n") % par->mecell) << hpx::flush;
        hpx::cout << ( boost::format("mpsi           : %1%\n") % par->mpsi) << hpx::flush;
        hpx::cout << ( boost::format("mthetamax      : %1%\n") % par->mthetamax) << hpx::flush;
        hpx::cout << ( boost::format("mzetamax       : %1%\n") % par->mzetamax) << hpx::flush;
        hpx::cout << ( boost::format("npartdom       : %1%\n") % par->npartdom) << hpx::flush;
        hpx::cout << ( boost::format("ncycle         : %1%\n") % par->ncycle) << hpx::flush;
        hpx::cout << ( boost::format("a              : %1%\n") % par->a) << hpx::flush;
        hpx::cout << ( boost::format("a0             : %1%\n") % par->a0) << hpx::flush;
        hpx::cout << ( boost::format("a1             : %1%\n") % par->a1) << hpx::flush;
        hpx::cout << ( boost::format("q0             : %1%\n") % par->q0) << hpx::flush;
        hpx::cout << ( boost::format("q1             : %1%\n") % par->q1) << hpx::flush;
        hpx::cout << ( boost::format("q2             : %1%\n") % par->q2) << hpx::flush;
        hpx::cout << ( boost::format("rc             : %1%\n") % par->rc) << hpx::flush;
        hpx::cout << ( boost::format("rw             : %1%\n") % par->rw) << hpx::flush;
        hpx::cout << ( boost::format("aion           : %1%\n") % par->aion) << hpx::flush;
        hpx::cout << ( boost::format("qion           : %1%\n") % par->qion) << hpx::flush;
        hpx::cout << ( boost::format("aelectron      : %1%\n") % par->aelectron) << hpx::flush;
        hpx::cout << ( boost::format("qelectron      : %1%\n") % par->qelectron) << hpx::flush;
        hpx::cout << ( boost::format("kappati        : %1%\n") % par->kappati) << hpx::flush;
        hpx::cout << ( boost::format("kappate        : %1%\n") % par->kappate) << hpx::flush;
        hpx::cout << ( boost::format("kappan         : %1%\n") % par->kappan) << hpx::flush;
        hpx::cout << ( boost::format("fixed_Tprofile : %1%\n") % par->fixed_Tprofile) << hpx::flush;
        hpx::cout << ( boost::format("tite           : %1%\n") % par->tite) << hpx::flush;
        hpx::cout << ( boost::format("flow0          : %1%\n") % par->flow0) << hpx::flush;
        hpx::cout << ( boost::format("flow1          : %1%\n") % par->flow1) << hpx::flush;
        hpx::cout << ( boost::format("flow2          : %1%\n") % par->flow2) << hpx::flush;
        hpx::cout << ( boost::format("r0             : %1%\n") % par->r0) << hpx::flush;
        hpx::cout << ( boost::format("b0             : %1%\n") % par->b0) << hpx::flush;
        hpx::cout << ( boost::format("temperature    : %1%\n") % par->temperature) << hpx::flush;
        hpx::cout << ( boost::format("edensity0      : %1%\n") % par->edensity0) << hpx::flush;
        hpx::cout << ( boost::format("output         : %1%\n") % par->output) << hpx::flush;
        hpx::cout << ( boost::format("nbound         : %1%\n") % par->nbound) << hpx::flush;
        hpx::cout << ( boost::format("umax           : %1%\n") % par->umax) << hpx::flush;
        hpx::cout << ( boost::format("iload          : %1%\n") % par->iload) << hpx::flush;
        hpx::cout << ( boost::format("tauii          : %1%\n") % par->tauii) << hpx::flush;
        hpx::cout << ( boost::format("track_particles: %1%\n") % par->track_particles) << hpx::flush;
        hpx::cout << ( boost::format("nptrack        : %1%\n") % par->nptrack) << hpx::flush;
        hpx::cout << ( boost::format("rng_control    : %1%\n") % par->rng_control) << hpx::flush;
        hpx::cout << ( boost::format("numberpe       : %1%\n") % par->numberpe) << hpx::flush;
        BOOST_FOREACH(std::size_t i, par->nmode)
        {
          hpx::cout << ( boost::format("nmode              : %1% \n") % i) << hpx::flush;
        }
        BOOST_FOREACH(std::size_t i, par->mmode)
        {
          hpx::cout << ( boost::format("mmode              : %1% \n") % i) << hpx::flush;
        }
        hpx::cout << ( boost::format("**************************************\n")  ) << hpx::flush;
        hpx::cout << ( boost::format("Using npartdom %1% and ntoroidal %2%\n") % par->npartdom % par->ntoroidal) << hpx::flush;
        hpx::cout << ( boost::format("**************************************\n")  ) << hpx::flush;

        ///////////////////////////////////////////////////////////////////////
        // Retrieve the command line options. 
        std::size_t const num_gridpoints = vm["n"].as<std::size_t>();
        std::size_t const num_particles = vm["np"].as<std::size_t>();

        std::string const meshfile = vm["mesh"].as<std::string>();
        std::string const particlefile = vm["particles"].as<std::string>();

        ///////////////////////////////////////////////////////////////////////
        // Create a distributing factory locally. The distributing factory can
        // be used to create blocks of components that are distributed across
        // all localities that support that component type. 
        hpx::components::distributing_factory factory;
        factory.create(hpx::find_here());

        // Get the global component type of our point component.
        hpx::components::component_type block_type_points =
            hpx::components::get_component_type<gtc::server::point>();

        // Create num_gridpoints point components with distributing factory.
        // These components will be evenly distributed among all available
        // localities supporting the component type.
        hpx::components::distributing_factory::result_type blocks_points =
            factory.create_components(block_type_points, num_gridpoints);

        // Get the component type for our particle component.
        hpx::components::component_type block_type_particles =
            hpx::components::get_component_type<gtc::server::particle>();

        // Create num_gridpoints particle components with distributing factory.
        // These components will be evenly distributed among all available
        // localities supporting the component type.
        hpx::components::distributing_factory::result_type blocks_particles =
            factory.create_components(block_type_particles, num_particles);

        ///////////////////////////////////////////////////////////////////////
        // These two vectors will hold client classes referring to all of the
        // components we just created.
        std::vector<gtc::point> points;
        std::vector<gtc::particle> particles;

        // Populate the client vectors. 
        init(hpx::components::server::locality_results(blocks_points), points);
        init(hpx::components::server::locality_results(blocks_particles), particles);

        ///////////////////////////////////////////////////////////////////////
        // Initialize the particles and points with the data from the input
        // files. 
        std::vector<hpx::lcos::promise<void> > initial_phase;

        for (std::size_t i=0;i<num_gridpoints;i++) {
          initial_phase.push_back(points[i].init_async(i,par));
        }

        for (std::size_t i=0;i<num_particles;i++) {
          initial_phase.push_back(particles[i].init_async(i,particlefile));
        }

        // While we're waiting for the initialization phase to complete, we 
        // build a vector of all of the particle GIDs. This will be used as the
        // input for the next phase.
        std::vector<hpx::naming::id_type> particle_components;
        for (std::size_t i=0;i<num_particles;i++) {
          particle_components.push_back(particles[i].get_gid());
        }

        // We have to wait for the initialization to complete before we begin
        // the next phase of computation. 
        hpx::lcos::wait(initial_phase);

        ///////////////////////////////////////////////////////////////////////
        // Start the search/charge depositing phase.
        std::vector<hpx::lcos::promise<void> > charge_phase;

        // We use the vector of particle component GIDS that we created during
        // the initialization phase as the input to the search action on each
        // point. 
        for (std::size_t i=0;i<num_gridpoints;i++) {
          charge_phase.push_back(points[i].search_async(particle_components));
        }

        // Wait for the search/charge depositing phase to complete.
        hpx::lcos::wait(charge_phase);

        // Print the total walltime that the computation took.
        std::cout << "Elapsed time: " << t.elapsed() << " [s]" << std::endl;

    } // Ensure things go out of scope before hpx::finalize is called.

    hpx::finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    using boost::program_options::value;

    // Configure application-specific options.
    boost::program_options::options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ("n", value<std::size_t>()->default_value(5),
            "the number of gridpoints")
        ("np", value<std::size_t>()->default_value(5),
            "the number of particles")
        ("mesh", value<std::string>()->default_value("mesh.txt"),
            "the file containing the mesh")
        ("particles", value<std::string>()->default_value("particles.txt"),
            "the file containing the particles");

    return hpx::init(desc_commandline, argc, argv); // Initialize and run HPX.
}

