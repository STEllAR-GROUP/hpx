//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c)      2011 Matt Anderson
//  Copyright (c)      2011 Bryce Lelbach
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/format.hpp>

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/util/util.hpp>

#include <examples/smp_amr3d/amr/dynamic_stencil_value.hpp>
#include <examples/smp_amr3d/amr/functional_component.hpp>
#include <examples/smp_amr3d/amr/unigrid_mesh.hpp>
#include <examples/smp_amr3d/amr_c/stencil.hpp>
#include <examples/smp_amr3d/amr_c/logging.hpp>

#include <examples/smp_amr3d/amr_c_test/rand.hpp>

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

using hpx::components::amr::parameter;
using hpx::components::amr::stencil;
using hpx::components::amr::unigrid_mesh;
using hpx::components::amr::server::logging;

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
    bool do_logging = false;

    if (vm.count("verbose"))
        do_logging = true;

    std::size_t timesteps = 0;
    
    if (vm.count("timesteps"))
        timesteps = vm["timesteps"].as<std::size_t>();
        
    std::size_t grain_size = 3;
    if (vm.count("grain-size"))
        grain_size = vm["grain-size"].as<std::size_t>();
    
    std::size_t allowedl = 0;
    if (vm.count("refinement"))
        allowedl = vm["refinement"].as<std::size_t>();
    
    std::size_t nx0 = 0;
    if (vm.count("dimensions"))
        nx0 = vm["dimensions"].as<std::size_t>();

    parameter par;

    // default pars
    par->allowedl       = allowedl;
    par->loglevel       = 2;
    par->output         = 1.0;
    par->output_stdout  = 1;
    par->lambda         = 0.15;
    par->nt0            = timesteps;
    par->minx0          = -15.0;
    par->maxx0          = 15.0;
    par->ethreshold     = 0.005;
    par->R0             = 8.0;
    par->amp            = 0.1;
    par->amp_dot        = 0.1;
    par->delta          = 1.0;
    par->gw             = 5;
    par->eps            = 0.0;
    par->output_level   = 0;
    par->granularity    = grain_size;

    for (std::size_t i = 0; i < HPX_SMP_AMR3D_MAX_LEVELS; i++)
        par->refine_level[i] = 1.5;

    id_type rt_id = get_applier().get_runtime_support_gid();

    section root;
    hpx::components::stubs::runtime_support::get_config(rt_id, root);

    if (root.has_section("smp_amr3d"))
    {
        section pars = *(root.get_section("smp_amr3d"));

        appconfig_option<double>("lambda", pars, par->lambda);
        appconfig_option<std::size_t>("refinement", pars, par->allowedl);
        appconfig_option<std::size_t>("loglevel", pars, par->loglevel);
        appconfig_option<double>("output", pars, par->output);
        appconfig_option<std::size_t>("output_stdout", pars, par->output_stdout);
        appconfig_option<std::size_t>("output_level", pars, par->output_level);
        appconfig_option<std::size_t>("dimensions", pars, nx0);
        appconfig_option<std::size_t>("timesteps", pars, par->nt0);
        appconfig_option<double>("maxx0", pars, par->maxx0);
        appconfig_option<double>("minx0", pars, par->minx0);
        appconfig_option<double>("ethreshold", pars, par->ethreshold);
        appconfig_option<double>("R0", pars, par->R0);
        appconfig_option<double>("delta", pars, par->delta);
        appconfig_option<double>("amp", pars, par->amp);
        appconfig_option<double>("amp_dot", pars, par->amp_dot);
        appconfig_option<std::size_t>("ghostwidth", pars, par->gw);
        appconfig_option<double>("eps", pars, par->eps);
        appconfig_option<std::size_t>("grain_size", pars, par->granularity);

        for (std::size_t i = 0; i < par->allowedl; i++)
        {
            std::string ref_level = "refine_level_"
                                  + lexical_cast<std::string>(i);
            appconfig_option<double>(ref_level, pars, par->refine_level[i]);
        }
    }

    // derived parameters
    if ((nx0 % par->granularity) != 0)
    {
        std::string msg = boost::str(boost::format(
            "dimensions (%1%) must be cleanly divisible by the grain-size "
            "(%2%)") % nx0 % par->granularity);
        HPX_THROW_IN_CURRENT_FUNC(bad_parameter, msg);
    }

    // the number of timesteps each px thread can take independent of
    // communication
    par->time_granularity = par->granularity / 3;
    
    // set up refinement centered around the middle of the grid
    par->nx[0] = nx0 / par->granularity;

    for (std::size_t i = 1; i < (par->allowedl + 1); ++i)
    {
        par->nx[i] = std::size_t(par->refine_level[i - 1] * par->nx[i - 1]);

        if ((par->nx[i] % 2) == 0)
            par->nx[i] += 1;

        if (par->nx[i] > (2 * par->nx[i - 1] - 5))
            par->nx[i] = 2 * par->nx[i - 1] - 5;
    }

    // for each row, record what the lowest level on the row is
    std::size_t num_rows = 1 << par->allowedl;

    // account for prolongation and restriction (which is done every other step)
    if (par->allowedl > 0)
        num_rows += (1 << par->allowedl) / 2;

    num_rows *= 2; // we take two timesteps in the mesh
    par->num_rows = num_rows;

    int ii = -1; 
    for (std::size_t i = 0; i < num_rows; ++i)
    {
        if (((i + 5) % 3) != 0 || (par->allowedl == 0))
            ii++;

        int level = -1;
        for (std::size_t j = par->allowedl; j>=0; --j)
        {
            if ((ii % (1 << j)) == 0)
            {
                level = par->allowedl - j;
                par->level_row.push_back(level);
                break;
            }
        }
    }

    par->dx0 = (par->maxx0 - par->minx0) / (nx0 - 1);
    par->dt0 = par->lambda * par->dx0;

    par->min.resize(par->allowedl + 1);
    par->max.resize(par->allowedl + 1);
    par->min[0] = par->minx0;
    par->max[0] = par->maxx0;
    for (std::size_t i = par->allowedl; i > 0; --i)
    {
        par->min[i] = 0.5 * (par->maxx0 - par->minx0) + par->minx0
            - (par->nx[i] - 1) / 2 * par->dx0 / (1 << i) * par->granularity;
        par->max[i] = par->min[i] + ((par->nx[i] - 1) * par->granularity
                    + par->granularity - 1) * par->dx0 / (1 << i);
    }

    par->rowsize.resize(par->allowedl + 1);
    for (int i = 0; i <= par->allowedl; ++i)
    {
        par->rowsize[i] = 0;
        for (int j = par->allowedl; j >= i; --j)
            par->rowsize[i] += par->nx[j] * par->nx[j] * par->nx[j];
    }
    
    // get component types needed below
    component_type function_type = get_component_type<stencil>();
    component_type logging_type = get_component_type<logging>();

    {
        id_type here = get_applier().get_runtime_support_gid();

        high_resolution_timer t;

        // we are in spherical symmetry, r=0 is the smallest radial domain point
        unigrid_mesh um;
        um.create(here);
        boost::shared_ptr<std::vector<id_type> > result_data =
            um.init_execute(function_type, par->rowsize[0], par->nt0,
                par->loglevel ? logging_type : component_invalid, par);

        std::cout << "Elapsed time: " << t.elapsed() << " [s]" << std::endl;

        for (std::size_t i = 0; i < result_data->size(); ++i)
            hpx::components::stubs::memory_block::free((*result_data)[i]);
    } // unigrid_mesh needs to go out of scope before shutdown

    // initiate shutdown of the runtime systems on all localities
    finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description
       desc_commandline("usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ("timesteps,s", value<std::size_t>()->default_value(10), 
            "the number of time steps to use for the computation")
        ("grain-size,g", value<std::size_t>()->default_value(10),
            "the grain-size of the data")
        ("dimensions,i", value<std::size_t>()->default_value(100),
            "the dimensions of the search space")
        ("refinement,e", value<std::size_t>()->default_value(0),
            "the number of refinment levels that should be computed")
        ("verbose,v", "print calculated values after each time step")
        ;

    // Initialize and run HPX
    return init(desc_commandline, argc, argv);
}

