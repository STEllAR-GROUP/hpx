//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c)      2011 Matt Anderson
//  Copyright (c)      2011 Bryce Lelbach
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/format.hpp>

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <examples/marduk/mesh/dynamic_stencil_value.hpp>
#include <examples/marduk/mesh/functional_component.hpp>
#include <examples/marduk/mesh/unigrid_mesh.hpp>
#include <examples/marduk/amr_c/stencil.hpp>
#include <examples/marduk/amr_c/logging.hpp>

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

    parameter par;

    // default pars
    par->loglevel       = 2;
    par->output         = 1.0;
    par->output_stdout  = 1;
    par->output_level   = 0;

    par->nx0            = 39;
    par->ny0            = 39;
    par->nz0            = 39;
    par->allowedl       = 0;
    par->lambda         = 0.15;
    par->nt0            = 10;
    par->minx0          = -4.0;
    par->maxx0          =  4.0;
    par->miny0          = -4.0;
    par->maxy0          =  4.0;
    par->minz0          = -4.0;
    par->maxz0          =  4.0;
    par->ethreshold     = 1.e-4;
    par->ghostwidth     = 9;
    par->bound_width    = 5;
    par->shadow         = 0;
    par->refine_factor  = 2;
    par->minefficiency  = 0.9;
    par->clusterstyle   = 0;
    par->mindim         = 6;

    par->num_px_threads = 40;
    par->refine_every   = 0;

    for (std::size_t i = 0; i < HPX_SMP_AMR3D_MAX_LEVELS; i++)
        par->refine_level[i] = 1.0;

    id_type rt_id = get_applier().get_runtime_support_gid();

    section root;
    hpx::components::stubs::runtime_support::get_config(rt_id, root);

    if (root.has_section("marduk"))
    {
        section pars = *(root.get_section("marduk"));

        appconfig_option<double>("lambda", pars, par->lambda);
        appconfig_option<std::size_t>("allowedl", pars, par->allowedl);
        appconfig_option<std::size_t>("loglevel", pars, par->loglevel);
        appconfig_option<double>("output", pars, par->output);
        appconfig_option<std::size_t>("output_stdout", pars, par->output_stdout);
        appconfig_option<std::size_t>("output_level", pars, par->output_level);
        appconfig_option<std::size_t>("nx0", pars, par->nx0);
        appconfig_option<std::size_t>("ny0", pars, par->ny0);
        appconfig_option<std::size_t>("nz0", pars, par->nz0);
        appconfig_option<std::size_t>("timesteps", pars, par->nt0);
        appconfig_option<double>("maxx0", pars, par->maxx0);
        appconfig_option<double>("minx0", pars, par->minx0);
        appconfig_option<double>("maxy0", pars, par->maxy0);
        appconfig_option<double>("miny0", pars, par->miny0);
        appconfig_option<double>("maxz0", pars, par->maxz0);
        appconfig_option<double>("minz0", pars, par->minz0);
        appconfig_option<double>("ethreshold", pars, par->ethreshold);
        appconfig_option<std::size_t>("ghostwidth", pars, par->ghostwidth);
        appconfig_option<std::size_t>("bound_width", pars, par->bound_width);
        appconfig_option<std::size_t>("num_px_threads", pars, par->num_px_threads);
        appconfig_option<std::size_t>("refine_every", pars, par->refine_every);
        appconfig_option<double>("minefficiency", pars, par->minefficiency);
        appconfig_option<std::size_t>("mindim", pars, par->mindim);
        appconfig_option<std::size_t>("shadow", pars, par->shadow);

        for (std::size_t i = 0; i < par->allowedl; i++)
        {
            std::string ref_level = "refine_level_"
                                  + lexical_cast<std::string>(i);
            appconfig_option<double>(ref_level, pars, par->refine_level[i]);
        }
    }

    // derived parameters
    if ( (par->nx0 % 2) == 0 || (par->ny0 % 2) == 0 || (par->nz0 % 2) == 0 ) 
    {
        std::string msg = boost::str(boost::format(
            "coarse mesh dimensions (%1%) (%2%) (%3%) must each be odd "
            ) % par->nx0 % par->ny0 % par->nz0);
        HPX_THROW_IN_CURRENT_FUNC(bad_parameter, msg);
    }
#if 0
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

        std::size_t level = 0;
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
#endif
#if 0
    // get component types needed below
    component_type function_type = get_component_type<stencil>();
    component_type logging_type = get_component_type<logging>();

    {
        id_type here = get_applier().get_runtime_support_gid();

        high_resolution_timer t;

        unigrid_mesh um;
        um.create(here);
        boost::shared_ptr<std::vector<id_type> > result_data =
            um.init_execute(function_type, par->rowsize[0], par->nt0,
                par->loglevel ? logging_type : component_invalid, par);

        std::cout << "Elapsed time: " << t.elapsed() << " [s]" << std::endl;

        for (std::size_t i = 0; i < result_data->size(); ++i)
            hpx::components::stubs::memory_block::free((*result_data)[i]);
    } // mesh needs to go out of scope before shutdown

    // initiate shutdown of the runtime systems on all localities
#endif
    finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description
       desc_commandline("usage: " HPX_APPLICATION_STRING " [options]");

    // Initialize and run HPX
    return init(desc_commandline, argc, argv);
}

