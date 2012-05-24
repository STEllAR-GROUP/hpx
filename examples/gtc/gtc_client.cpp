//  Copyright (c) 2007-2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/iostreams.hpp>

#include "point/point.hpp"
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
        // TEST
        //par->mstep = 1500;
        par->mstep = 4;
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
        par->tauii = -1.0;
        par->temperature = 2500.0;
        par->edensity0 = 0.46e14;

        par->mflux = 5;
        par->num_mode = 8;
        par->m_poloidal = 9;

        par->output = 6;
        par->nbound = 4;
        par->umax = 4.0;
        par->iload = false;
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
        //par->numberpe = 40;
        par->numberpe = 1;

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
          appconfig_option<double>("tauii", pars, par->tauii);
          appconfig_option<double>("temperature", pars, par->temperature);
          appconfig_option<double>("edensity0", pars, par->edensity0);
          appconfig_option<std::size_t>("output", pars, par->output);
          appconfig_option<std::size_t>("nbound", pars, par->nbound);
          appconfig_option<double>("umax", pars, par->umax);
          appconfig_option<bool>("iload", pars, par->iload);
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
        // TEST
        //std::size_t two = 2;
        //par->mstep = (std::max)(two,par->mstep);

        // I don't think this is right
        //  par->msnap = (std::min)(par->msnap,par->mstep/par->ndiag);

        par->isnap = par->mstep/par->msnap;
        par->idiag1 = par->mpsi/2;
        par->idiag2 = par->mpsi/2;
        if ( par->nonlinear < 0.5 ) {
          par->paranl = 0.0;
          par->mode00 = false;
          par->idiag1 = 1;
          par->idiag2 = par->mpsi;
        }
        par->rc = par->rc*(par->a0 + par->a1);
        par->rw = 1.0/(par->rw*(par->a1-par->a0));

        // equilibrium unit: length (unit=cm) and time (unit=second) unit
        double ulength=par->r0;
        par->utime=1.0/(9580.0*par->b0); // time unit = inverse gyrofrequency of proton
        // primary ion thermal gyroradius in equilibrium unit, vthermal=sqrt(T/m)
        par->gyroradius=102.0*sqrt(par->aion*par->temperature)/
                            (abs(par->qion)*par->b0)/ulength;
        par->tstep = par->tstep*par->aion/(abs(par->qion)*par->gyroradius*par->kappati);

        // basic ion-ion collision time, Braginskii definition
        bool collision = false;
        if ( par->tauii > 0.0 ) {
          double zeff = par->qion;
          double tau_vth = 23.0-log(sqrt(zeff*zeff*par->edensity0)/pow(par->temperature,1.5));
          tau_vth=2.09e7*pow(par->temperature,1.5)*sqrt(par->aion)/
                             (par->edensity0*tau_vth*par->utime*zeff);
          par->tauii *= tau_vth;
          collision = true;
        }

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
        int tmp = static_cast<int>(tmp1/tmp2 + 0.5);
        par->mzetamax = par->ntoroidal*(std::max)(1,tmp);

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
        if ( collision ) {
          double r = 0.5*(par->a0 + par->a1);
          double q = par->q0 + par->q1*r/par->a + par->q2*r*r/(par->a*par->a);
          double tmp = q/(par->tauii*par->gyroradius*pow(r,1.5));
          hpx::cout << ( boost::format("Collision time tauii=%1%   nu_star=%2%  q=%3% \n") %
                 par->tauii % tmp % q ) << hpx::flush;
        }
        hpx::cout << ( boost::format("mflux=%1%  num_mode=%2%  m_poloidal=%3% \n") %
             par->mflux % par->num_mode % par->m_poloidal ) << hpx::flush;

        ///////////////////////////////////////////////////////////////////////
        // Create a distributing factory locally. The distributing factory can
        // be used to create blocks of components that are distributed across
        // all localities that support that component type.
        hpx::components::distributing_factory factory;
        factory.create(hpx::find_here());

        // Get the global component type of our point component.
        hpx::components::component_type block_type_points =
            hpx::components::get_component_type<gtc::server::point>();

        // Create ntoroidal point components with distributing factory.
        // These components will be evenly distributed among all available
        // localities supporting the component type.
        hpx::components::distributing_factory::result_type blocks_points =
            factory.create_components(block_type_points, par->ntoroidal);

        ///////////////////////////////////////////////////////////////////////
        // This will hold client classes referring to all of the
        // components we just created.
        std::vector<gtc::point> points;

        // Populate the client vectors.
        init(hpx::util::locality_results(blocks_points), points);

        ///////////////////////////////////////////////////////////////////////
        { // SETUP
          std::vector<hpx::lcos::future<void> > initial_phase;
          for (std::size_t i=0;i<par->ntoroidal;i++) {
            initial_phase.push_back(points[i].init_async(i,par));
          }
          hpx::lcos::wait(initial_phase);
        }

        bool do_collision;
        if ( par->tauii > 0.0 ) {
          do_collision = true;
        } else {
          do_collision = false;
        }

        { // LOAD
          std::vector<hpx::lcos::future<void> > load_phase;
          for (std::size_t i=0;i<par->ntoroidal;i++) {
            load_phase.push_back(points[i].load_async(i,par));
          }
          hpx::lcos::wait(load_phase);
        }

        std::vector<hpx::naming::id_type> point_components;
        for (std::size_t i=0;i<par->ntoroidal;i++) {
          point_components.push_back(points[i].get_gid());
        }

        std::size_t istep = 0;
        { // CHARGEI
          std::vector<hpx::lcos::future<void> > chargei_phase;
          for (std::size_t i=0;i<par->ntoroidal;i++) {
            chargei_phase.push_back(points[i].chargei_async(istep,
                                point_components,par));
          }
          hpx::lcos::wait(chargei_phase);
        }

        // main time loop
        for (istep=1;istep<=par->mstep;istep++) {
          std::cout << " Step : " << istep << std::endl;
          for (std::size_t irk=1;irk<=2;irk++) {
            // idiag=0: do time history diagnosis
            std::size_t idiag = ((irk+1)%2) + (istep%par->ndiag);

            {  // SMOOTH(3)
              std::vector<hpx::lcos::future<void> > smooth_phase;
              std::size_t iflag = 3;
              for (std::size_t i=0;i<par->ntoroidal;i++) {
                smooth_phase.push_back(points[i].smooth_async(iflag,
                                      point_components,idiag,par));
              }
              hpx::lcos::wait(smooth_phase);
            }

            {  // FIELD
              std::vector<hpx::lcos::future<void> > field_phase;
              for (std::size_t i=0;i<par->ntoroidal;i++) {
                field_phase.push_back(points[i].field_async(
                                      point_components,par));
              }
              hpx::lcos::wait(field_phase);
            }

            // push ion
            {  // PUSHI
              std::vector<hpx::lcos::future<void> > pushi_phase;
              for (std::size_t i=0;i<par->ntoroidal;i++) {
                pushi_phase.push_back(points[i].pushi_async(irk,istep,idiag,
                                      point_components,par));
              }
              hpx::lcos::wait(pushi_phase);
            }

            // redistribute ion across PEs
            {  // SHIFTI
              std::vector<hpx::lcos::future<void> > shifti_phase;
              for (std::size_t i=0;i<par->ntoroidal;i++) {
                shifti_phase.push_back(points[i].shifti_async(
                                      point_components,par));
              }
              hpx::lcos::wait(shifti_phase);
            }

            if ( irk == 2 && do_collision ) { 
              std::cerr << " Collision not implemented yet.  Not a default parameter " << std::endl;
            }

            // ion perturbed density
            { // CHARGEI
              std::vector<hpx::lcos::future<void> > chargei_phase;
              for (std::size_t i=0;i<par->ntoroidal;i++) {
                chargei_phase.push_back(points[i].chargei_async(istep,
                                    point_components,par));
              }
              hpx::lcos::wait(chargei_phase);
            }

            // smooth ion density
            {  // SMOOTH(0)
              std::vector<hpx::lcos::future<void> > smooth_phase;
              std::size_t iflag = 0;
              for (std::size_t i=0;i<par->ntoroidal;i++) {
                smooth_phase.push_back(points[i].smooth_async(iflag,
                                      point_components,idiag,par));
              }
              hpx::lcos::wait(smooth_phase);
            }

            // solve GK Poisson equation using adiabatic electron
            {  // POISSON(0)
              std::vector<hpx::lcos::future<void> > poisson_phase;
              std::size_t iflag = 0;
              for (std::size_t i=0;i<par->ntoroidal;i++) {
                poisson_phase.push_back(points[i].poisson_async(
                                      iflag,istep,irk,
                                      point_components,par));
              }
              hpx::lcos::wait(poisson_phase);
            }

            for (std::size_t ihybrid=1;ihybrid<=par->nhybrid;ihybrid++) {
              std::cerr << " par->nybrid > 0 not supported yet " << std::endl;
            }
          }
        }
        ///////////////////////////////////////////////////////////////////////

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

    return hpx::init(desc_commandline, argc, argv); // Initialize and run HPX.
}

