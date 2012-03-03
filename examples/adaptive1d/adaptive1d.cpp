//  Copyright (c) 2009-2011 Matthew Anderson
//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <examples/adaptive1d/dataflow/dynamic_stencil_value.hpp>
#include <examples/adaptive1d/dataflow/functional_component.hpp>
#include <examples/adaptive1d/dataflow/dataflow_stencil.hpp>
#include <examples/adaptive1d/stencil/stencil.hpp>
#include <examples/adaptive1d/stencil/stencil_data.hpp>
#include <examples/adaptive1d/stencil/stencil_functions.hpp>
#include <examples/adaptive1d/stencil/logging.hpp>
#include <examples/adaptive1d/refine.hpp>

#include <boost/format.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;
using boost::lexical_cast;

using hpx::bad_parameter;

using hpx::applier::get_applier;

using hpx::components::component_type;
using hpx::components::component_invalid;
using hpx::components::get_component_type;

using hpx::naming::id_type;

using hpx::util::section;
using hpx::util::high_resolution_timer;

using hpx::init;
using hpx::finalize;

using hpx::components::adaptive1d::parameter;
using hpx::components::adaptive1d::stencil;
using hpx::components::adaptive1d::dataflow_stencil;
using hpx::components::adaptive1d::server::logging;

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
int hpx_main(variables_map& vm)
{

    parameter par;

    // default pars
    par->loglevel       = 2;

    par->nt0            = 100;
    par->nx0            = 101;
    par->grain_size     = 10;
    par->allowedl       = 0;
    par->num_neighbors  = 2;
    par->out_every      = 5.0;
    par->refine_every   = 100;
    par->ethreshold     = 1.e-4;
    par->ghostwidth     = 9;

    // application specific parameters
    par->cfl    = 0.01;
    par->disip  = 0.005;
    par->Rmin   = -5.0;
    par->Rout   = 15.0;
    par->tau    = 1.0;
    par->lambda = 1.0;
    par->v      = 0.1;
    par->amp    = 0.0001;
    par->x0     = 10.0;
    par->id_sigma = 0.9;
    par->outdir = "./";

    id_type rt_id = get_applier().get_runtime_support_gid();

    section root;
    hpx::components::stubs::runtime_support::get_config(rt_id, root);

    if (root.has_section("adaptive1d"))
    {
        section pars = *(root.get_section("adaptive1d"));

        appconfig_option<std::size_t>("loglevel", pars, par->loglevel);
        appconfig_option<std::size_t>("nx0", pars, par->nx0);
        appconfig_option<std::size_t>("nt0", pars, par->nt0);
        appconfig_option<std::size_t>("allowedl", pars, par->allowedl);
        appconfig_option<std::size_t>("grain_size", pars, par->grain_size);
        appconfig_option<std::size_t>("num_neighbors", pars, par->num_neighbors);
        appconfig_option<double>("out_every", pars, par->out_every);
        appconfig_option<std::string>("output_directory", pars, par->outdir);
        appconfig_option<std::size_t>("refine_every", pars, par->refine_every);
        appconfig_option<double>("ethreshold", pars, par->ethreshold);
        appconfig_option<std::size_t>("ghostwidth", pars, par->ghostwidth);

        // Application parameters
        appconfig_option<double>("cfl", pars, par->cfl);
        appconfig_option<double>("disip", pars, par->disip);
        appconfig_option<double>("Rmin", pars, par->Rmin);
        appconfig_option<double>("Rout", pars, par->Rout);
        appconfig_option<double>("tau", pars, par->tau);
        appconfig_option<double>("lambda", pars, par->lambda);
        appconfig_option<double>("v", pars, par->v);
        appconfig_option<double>("amp", pars, par->amp);
        appconfig_option<double>("x0", pars, par->x0);
        appconfig_option<double>("id_sigma", pars, par->id_sigma);
    }

    // derived parameters
    if ( (par->nx0 % 2) == 0 )
    {
        std::string msg = boost::str(boost::format(
            "mesh dimensions (%1%) must be odd "
            ) % par->nx0 );
        HPX_THROW_IN_CURRENT_FUNC(bad_parameter, msg);
    }
    if ( (par->nt0 % 2) != 0 )
    {
        std::string msg = boost::str(boost::format(
            "nt0 needs to be even: (%1%) "
            ) % par->nt0);
        HPX_THROW_IN_CURRENT_FUNC(bad_parameter, msg);
    }
    if ( par->grain_size <= 3*par->num_neighbors ) {
        std::string msg = boost::str(boost::format(
            "Increase grain size (%1%) or decrease the num_neighbors (%2%) "
            ) % par->grain_size % par->num_neighbors);
        HPX_THROW_IN_CURRENT_FUNC(bad_parameter, msg);
    }
    if ( par->refine_every < 1 ) {
      // no adaptivity
      par->refine_every = par->nt0;
    }

    if ( (par->refine_every % 2) != 0 )
    {
        std::string msg = boost::str(boost::format(
            "refine_every needs to be even: (%1%) "
            ) % par->refine_every);
        HPX_THROW_IN_CURRENT_FUNC(bad_parameter, msg);
    }

    if ( (par->nt0 % par->refine_every ) != 0 )
    {
        std::string msg = boost::str(boost::format(
            "nt0 (%1%) needs to be evenly divisible by refine_every (%2%) "
            ) % par->nt0 % par->refine_every);
        HPX_THROW_IN_CURRENT_FUNC(bad_parameter, msg);
    }

    std::size_t time_grain_size = par->nt0/par->refine_every;

    std::cout << " Parameters    : " << std::endl;
    std::cout << " nx0           : " << par->nx0 << std::endl;
    std::cout << " nt0           : " << par->nt0 << std::endl;
    std::cout << " refine_every  : " << par->refine_every << std::endl;
    std::cout << " allowedl      : " << par->allowedl << std::endl;
    std::cout << " grain_size    : " << par->grain_size << std::endl;
    std::cout << " num_neighbors : " << par->num_neighbors << std::endl;
    std::cout << " out_every     : " << par->out_every << std::endl;
    std::cout << " output_directory : " << par->outdir << std::endl;
    std::cout << " --------------: " << std::endl;
    std::cout << " loglevel      : " << par->loglevel << std::endl;
    std::cout << " --------------: " << std::endl;
    std::cout << " Application     " << std::endl;
    std::cout << " --------------: " << std::endl;
    std::cout << " cfl           : " << par->cfl << std::endl;
    std::cout << " disip         : " << par->disip << std::endl;
    std::cout << " Rmin          : " << par->Rmin << std::endl;
    std::cout << " Rout          : " << par->Rout << std::endl;
    std::cout << " tau           : " << par->tau << std::endl;
    std::cout << " lambda        : " << par->lambda << std::endl;
    std::cout << " v             : " << par->v << std::endl;
    std::cout << " amp           : " << par->amp << std::endl;
    std::cout << " x0            : " << par->x0 << std::endl;
    std::cout << " id_sigma      : " << par->id_sigma << std::endl;

    // number stencils
    std::size_t number_stencils = par->nx0/par->grain_size;

    // compute derived parameters
    par->minx0 = par->Rmin;
    par->maxx0 = par->Rout;
    par->h = (par->maxx0 - par->minx0)/(par->nx0-1);

    par->levelp.resize(par->allowedl+1);
    par->levelp[0] = 0;

    // Compute the grid sizes
    boost::shared_ptr<std::vector<id_type> > placeholder;
    double initial_time = 0.0;
    int rc = level_refine(-1,par,placeholder,initial_time);
    for (std::size_t i=0;i<par->allowedl;i++) {
      rc = level_refine(i,par,placeholder,initial_time);
    }

    for (std::size_t i=1;i<=par->allowedl;i++) {
      rc = level_bbox(i,par);
    }

    // TEST
    // Check grid structure
    //for (std::size_t i=0;i<=par->allowedl;i++) {
    //  int gi = level_return_start(i,par);
    //  while ( grid_return_existence(gi,par) ) {
    //    std::cout << " TEST level : " << i << " nx : " << par->gr_nx[gi] << " minx : " << par->gr_minx[gi] << " maxx : " << par->gr_maxx[gi] << " test dx : " << (par->gr_maxx[gi]-par->gr_minx[gi])/(par->gr_nx[gi]-1) << " actual h " << par->gr_h[gi] << " end: " << par->gr_minx[gi] + (par->gr_nx[gi]-1)*par->gr_h[gi] << std::endl;
    //    std::cout << " TEST bbox : " << par->gr_lbox[gi] << " " << par->gr_rbox[gi] << std::endl;
    //    gi = par->gr_sibling[gi];
    //  }
    //}
    // END TEST

    rc = compute_numrows(par);
    rc = compute_rowsize(par);
    //std::cout << " num_rows " << par->num_rows << std::endl;
    //std::cout << " rowsize " << par->rowsize[0] << " number stencils " << number_stencils << std::endl;

    // get component types needed below
    component_type function_type = get_component_type<stencil>();
    component_type logging_type = get_component_type<logging>();

    {
        id_type here = get_applier().get_runtime_support_gid();

        high_resolution_timer t;

        dataflow_stencil um;
        um.create(here);
        int numsteps = par->refine_every/2;
        boost::shared_ptr<std::vector<id_type> > result_data(new std::vector<id_type>);

        for (std::size_t j=0;j<time_grain_size;j++) {
          double time = j*par->refine_every*par->cfl*par->h;

          result_data = um.init_execute(*result_data,time,function_type,
                                        par->rowsize[0], numsteps,
                                par->loglevel ? logging_type : component_invalid, par);

          // Regrid
          time = (j+1)*par->refine_every*par->cfl*par->h;
          std::cout << " Completed time: " << time << std::endl;
        }
        std::cout << "Elapsed time: " << t.elapsed() << " [s]" << std::endl;

        for (std::size_t i = 0; i < result_data->size(); ++i)
            hpx::components::stubs::memory_block::free((*result_data)[i]);
    } // mesh needs to go out of scope before shutdown
    // initiate shutdown of the runtime systems on all localities

    finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    // Initialize and run HPX
    return init(desc_commandline, argc, argv);
}

