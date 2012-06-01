//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Matt Anderson
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <examples/dataflow/mesh/dynamic_stencil_value.hpp>
#include <examples/dataflow/mesh/functional_component.hpp>
#include <examples/dataflow/mesh/unigrid_mesh.hpp>
#include <examples/dataflow/amr_c/stencil.hpp>
#include <examples/dataflow/amr_c/stencil_data.hpp>
#include <examples/dataflow/amr_c/stencil_functions.hpp>
#include <examples/dataflow/amr_c/logging.hpp>

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

    par->nt0            = 10;
    par->nx0            = 3;
    par->grain_size     = 5;

    id_type rt_id = get_applier().get_runtime_support_gid();

    section root;
    hpx::components::stubs::runtime_support::get_config(rt_id, root);

    if (root.has_section("dataflow"))
    {
        section pars = *(root.get_section("dataflow"));

        appconfig_option<std::size_t>("loglevel", pars, par->loglevel);
        appconfig_option<std::size_t>("nx0", pars, par->nx0);
        appconfig_option<std::size_t>("nt0", pars, par->nt0);
        appconfig_option<std::size_t>("grain_size", pars, par->grain_size);
    }

    // derived parameters
    if ( par->nx0 < 3 )
    {
        std::string msg = boost::str(boost::format(
            "mesh dimension (%1%) must be 3 or greater "
            ) % par->nx0 );
        HPX_THROW_IN_CURRENT_FUNC(bad_parameter, msg);
    }
    //if ( (par->nt0 % 5) != 0 )
    //{
    //    std::string msg = boost::str(boost::format(
    //        "nt0 needs to be even: (%1%) "
    //        ) % par->nt0);
    //    HPX_THROW_IN_CURRENT_FUNC(bad_parameter, msg);
    //}

    std::cout << " Parameters    : " << std::endl;
    std::cout << " nx0           : " << par->nx0 << std::endl;
    std::cout << " nt0           : " << par->nt0 << std::endl;
    std::cout << " grain_size    : " << par->grain_size << std::endl;
    std::cout << " --------------: " << std::endl;
    std::cout << " loglevel      : " << par->loglevel << std::endl;
    std::cout << " --------------: " << std::endl;

    // get component types needed below
    component_type function_type = get_component_type<stencil>();
    component_type logging_type = get_component_type<logging>();

    {
        id_type here = get_applier().get_runtime_support_gid();

        high_resolution_timer t;

        unigrid_mesh um;
        um.create(here);
        int numsteps = 1;
        boost::shared_ptr<std::vector<id_type> > result_data(new std::vector<id_type>);

        double time = 0.0;
        result_data = um.init_execute(*result_data,time,function_type,
                                      par->nx0, numsteps,
                              par->loglevel ? logging_type : component_invalid, par);

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

