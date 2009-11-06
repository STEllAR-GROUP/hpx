//  Copyright (c) 2007-2009 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/lcos/future_wait.hpp>
#include <hpx/runtime/components/component_factory.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>
#include <boost/assign/std/vector.hpp>

#include "../dynamic_stencil_value.hpp"
#include "../functional_component.hpp"
#include "../../parameter.hpp"

#include "amr_mesh_tapered.hpp"

///////////////////////////////////////////////////////////////////////////////
typedef hpx::components::amr::server::amr_mesh_tapered had_amr_mesh_tapered_type;

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the actions
HPX_REGISTER_ACTION_EX(had_amr_mesh_tapered_type::init_execute_action, had_amr_mesh_tapered_init_execute_action);
HPX_REGISTER_ACTION_EX(had_amr_mesh_tapered_type::execute_action, had_amr_mesh_tapered_execute_action);

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::simple_component<had_amr_mesh_tapered_type>, had_amr_mesh_tapered);
HPX_DEFINE_GET_COMPONENT_TYPE(had_amr_mesh_tapered_type);

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr { namespace server 
{
    amr_mesh_tapered::amr_mesh_tapered()
      : numvalues_(0)
    {}

    ///////////////////////////////////////////////////////////////////////////////
    // Initialize functional components by setting the logging component to use
    void amr_mesh_tapered::init(distributed_iterator_range_type const& functions,
        distributed_iterator_range_type const& logging, std::size_t numsteps)
    {
        components::distributing_factory::iterator_type function = functions.first;
        naming::id_type log = naming::invalid_id;

        if (logging.first != logging.second)
            log = *logging.first;

        for (/**/; function != functions.second; ++function)
        {
            components::amr::stubs::functional_component::
                init(*function, numsteps, log);
        }
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Create functional components, one for each data point, use those to 
    // initialize the stencil value instances
    void amr_mesh_tapered::init_stencils(distributed_iterator_range_type const& stencils,
        distributed_iterator_range_type const& functions, int static_step, 
        int numvalues, Parameter const& par)
    {
        components::distributing_factory::iterator_type stencil = stencils.first;
        components::distributing_factory::iterator_type function = functions.first;

        BOOST_ASSERT(par.coarsestencilsize == 5);

        for (int column = 0; stencil != stencils.second; ++stencil, ++function, ++column)
        {
            namespace stubs = components::amr::stubs;
            BOOST_ASSERT(function != functions.second);
            if ( numvalues == 8 ) {
              if (column == 0 || column == 7) {
                stubs::dynamic_stencil_value::set_functional_component(*stencil,
                                                *function, static_step, column, 1,2, par);
              } else if (column == 1 || column == 6) {
                stubs::dynamic_stencil_value::set_functional_component(*stencil,
                                                *function, static_step, column, 1,3, par);
              } else if (column == 2 || column == 3 || column == 4 || column == 5) {
                stubs::dynamic_stencil_value::set_functional_component(*stencil,
                                                *function, static_step, column, 1,4, par);
              } else {
                // this shouldn't happen
                BOOST_ASSERT(false);
              }
            } else if ( numvalues == 6 ) {
              if (column == 0 || column == 5) {
                stubs::dynamic_stencil_value::set_functional_component(*stencil,
                                                *function, static_step, column, 3,1, par);
              } else if ( column == 1 || column == 2 || column == 3 || column == 4 ) {
                stubs::dynamic_stencil_value::set_functional_component(*stencil,
                                                *function, static_step, column, 5,2, par);
              } else {
                // this shouldn't happen
                BOOST_ASSERT(false);
              }
            } else if ( numvalues == 2 ) {
              if (column == 0 || column == 1) {
                stubs::dynamic_stencil_value::set_functional_component(*stencil,
                                                *function, static_step, column, 5,4, par);
              } else {
                // this shouldn't happen
                BOOST_ASSERT(false);
              }
            } else {
              // this shouldn't happen
              BOOST_ASSERT(false);
            }
        }
       // BOOST_ASSERT(function == functions.second);
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Get gids of output ports of all functions
    void amr_mesh_tapered::get_output_ports(
        distributed_iterator_range_type const& stencils,
        std::vector<std::vector<naming::id_type> >& outputs)
    {
        typedef components::distributing_factory::result_type result_type;
        typedef 
            std::vector<lcos::future_value<std::vector<naming::id_type> > >
        lazyvals_type;

        // start an asynchronous operation for each of the stencil value instances
        lazyvals_type lazyvals;
        components::distributing_factory::iterator_type stencil = stencils.first;
        for (/**/; stencil != stencils.second; ++stencil)
        {
            lazyvals.push_back(components::amr::stubs::dynamic_stencil_value::
                get_output_ports_async(*stencil));
        }

        // now wait for the results
        lazyvals_type::iterator lend = lazyvals.end();
        for (lazyvals_type::iterator lit = lazyvals.begin(); lit != lend; ++lit) 
        {
            outputs.push_back((*lit).get());
        }
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Connect the given output ports with the correct input ports, creating the 
    // required static data-flow structure.
    //
    // Currently we have exactly one stencil_value instance per data point, where 
    // the output ports of a stencil_value are connected to the input ports of the 
    // direct neighbors of itself.
    inline std::size_t mod(int idx, std::size_t maxidx)
    {
        return (idx < 0) ? (idx + maxidx) % maxidx : idx % maxidx;
    }

    void amr_mesh_tapered::connect_input_ports(
        components::distributing_factory::result_type const* stencils,
        std::vector<std::vector<std::vector<naming::id_type> > > const& outputs,
        Parameter const& par)
    {
        BOOST_ASSERT(par.coarsestencilsize == 5);

        typedef components::distributing_factory::result_type result_type;

        int steps = (int)outputs.size();
        for (int step = 0; step < steps; ++step) 
        {
            std::size_t numvals = outputs[0].size();
            components::distributing_factory::iterator_range_type r = 
                locality_results(stencils[step]);
            components::distributing_factory::iterator_type stencil = r.first;
            for (int i = 0; stencil != r.second; ++stencil, ++i)
            {
                using namespace boost::assign;

                std::vector<naming::id_type> output_ports;
                if (step==0) {
                    if (i==0) {
                      output_ports += outputs[step+2][0][0];  
                    } else if (i==1) {
                      output_ports += outputs[step+2][0][1];  
                    } else if (i==2) {
                      output_ports += outputs[step+2][0][2];  
                    } else if (i==3) {
                      output_ports += outputs[step+2][0][3];  
                    } else if (i==4) {
                      output_ports += outputs[step+2][1][0];  
                    } else if (i==5) {
                      output_ports += outputs[step+2][1][1];  
                    } else if (i==6) {
                      output_ports += outputs[step+2][1][2];  
                    } else if (i==7) {
                      output_ports += outputs[step+2][1][3];  
                    } else {
                      BOOST_ASSERT(false);
                    }
                } else if (step==1) {
                    if (i==0) {
                      output_ports += outputs[step-1][0][0],  
                                      outputs[step-1][1][0],
                                      outputs[step-1][2][0];
                    } else if (i==1) {
                      output_ports += outputs[step-1][0][1],  
                                      outputs[step-1][1][1],
                                      outputs[step-1][2][1],
                                      outputs[step-1][3][0],
                                      outputs[step-1][4][0];
                    } else if (i==2) {
                      output_ports += outputs[step-1][1][2],  
                                      outputs[step-1][2][2],
                                      outputs[step-1][3][1],
                                      outputs[step-1][4][1],
                                      outputs[step-1][5][0];
                    } else if (i==3) {
                      output_ports += outputs[step-1][2][3],  
                                      outputs[step-1][3][2],  
                                      outputs[step-1][4][2],  
                                      outputs[step-1][5][1],  
                                      outputs[step-1][6][0];  
                    } else if (i==4) {
                      output_ports += outputs[step-1][3][3],  
                                      outputs[step-1][4][3],  
                                      outputs[step-1][5][2],  
                                      outputs[step-1][6][2],  
                                      outputs[step-1][7][1];  
                    } else if (i==5) {
                      output_ports += outputs[step-1][5][3],  
                                      outputs[step-1][6][1],  
                                      outputs[step-1][7][0];  
                    } else {
                      BOOST_ASSERT(false);
                    }
                } else if (step==2) {
                    if (i==0) {
                      output_ports += outputs[step-1][0][0],  
                                      outputs[step-1][1][0],  
                                      outputs[step-1][2][0],  
                                      outputs[step-1][3][0],  
                                      outputs[step-1][4][0];
                  } else if (i==1) {
                      output_ports += outputs[step-1][1][1],  
                                      outputs[step-1][2][1],  
                                      outputs[step-1][3][1],  
                                      outputs[step-1][4][1],  
                                      outputs[step-1][5][0];
                  } else {
                    BOOST_ASSERT(false);
                  }
                } else {
                  BOOST_ASSERT(false);
                }

                components::amr::stubs::dynamic_stencil_value::
                    connect_input_ports(*stencil, output_ports);
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////////
    void amr_mesh_tapered::prepare_initial_data(
        distributed_iterator_range_type const& functions, 
        std::vector<naming::id_type>& initial_data,
        Parameter const& par)
    {
        typedef std::vector<lcos::future_value<naming::id_type> > lazyvals_type;

        // create a data item value type for each of the functions
        lazyvals_type lazyvals;
        components::distributing_factory::iterator_type function = functions.first;

        for (std::size_t i = 0; function != functions.second; ++function, ++i)
        {
            lazyvals.push_back(components::amr::stubs::functional_component::
                alloc_data_async(*function, i, numvalues_, 0, par));
        }

        // now wait for the results
        lazyvals_type::iterator lend = lazyvals.end();
        for (lazyvals_type::iterator lit = lazyvals.begin(); lit != lend; ++lit) 
        {
            initial_data.push_back((*lit).get());
        }
    }

    ///////////////////////////////////////////////////////////////////////////////
    // do actual work
    void amr_mesh_tapered::execute(
        components::distributing_factory::iterator_range_type const& stencils, 
        std::vector<naming::id_type> const& initial_data, 
        std::vector<naming::id_type>& result_data)
    {
        // start the execution of all stencil stencils (data items)
        typedef std::vector<lcos::future_value<naming::id_type> > lazyvals_type;

        lazyvals_type lazyvals;
        components::distributing_factory::iterator_type stencil = stencils.first;
        for (std::size_t i = 0; stencil != stencils.second; ++stencil, ++i)
        {
            BOOST_ASSERT(i < initial_data.size());
            lazyvals.push_back(components::amr::stubs::dynamic_stencil_value::
                call_async(*stencil, initial_data[i]));
          //  lazyvals.push_back(components::amr::stubs::stencil_value<3>::
          //      call_async(*stencil, initial_data[i]));
        }

        // now wait for the results
        lazyvals_type::iterator lend = lazyvals.end();
        for (lazyvals_type::iterator lit = lazyvals.begin(); lit != lend; ++lit) 
        {
            result_data.push_back((*lit).get());
        }
    }
    
    ///////////////////////////////////////////////////////////////////////////////
    // 
    void amr_mesh_tapered::start_row(
        components::distributing_factory::iterator_range_type const& stencils)
    {
        // start the execution of all stencil stencils (data items)
        typedef std::vector<lcos::future_value< void > > lazyvals_type;

        lazyvals_type lazyvals;
        components::distributing_factory::iterator_type stencil = stencils.first;
        for (std::size_t i = 0; stencil != stencils.second; ++stencil, ++i)
        {
            lazyvals.push_back(components::amr::stubs::dynamic_stencil_value::
                start_async(*stencil));
        }

        // now wait for the results
        lazyvals_type::iterator lend = lazyvals.end();
        for (lazyvals_type::iterator lit = lazyvals.begin(); lit != lend; ++lit) 
        {
            (*lit).get();
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    /// This is the main entry point of this component. 
    std::vector<naming::id_type> amr_mesh_tapered::init_execute(
        components::component_type function_type, std::size_t numvalues, 
        std::size_t numsteps,
        components::component_type logging_type, Parameter const& par)
    {
        BOOST_ASSERT(numvalues == 8);
        BOOST_ASSERT(numsteps == 2);

        std::vector<naming::id_type> result_data;

        components::component_type stencil_type = 
            components::get_component_type<components::amr::server::dynamic_stencil_value >();

        typedef components::distributing_factory::result_type result_type;

        // create a distributing factory locally
        components::distributing_factory factory(
            components::distributing_factory::create(
                applier::get_applier().get_runtime_support_gid(), true));

        // create a couple of stencil (functional) components and twice the 
        // amount of stencil_value components
        numvalues_ = numvalues;
        result_type functions = factory.create_components(function_type, numvalues);
        result_type stencils[3] = 
        {
            factory.create_components(stencil_type, 8),
            factory.create_components(stencil_type, 6),
            factory.create_components(stencil_type, 2)
        };

        // initialize logging functionality in functions
        result_type logging;
        if (logging_type != components::component_invalid)
            logging = factory.create_components(logging_type);

        init(locality_results(functions), locality_results(logging), numsteps);

        // initialize stencil_values using the stencil (functional) components
        init_stencils(locality_results(stencils[0]), locality_results(functions), 
            0, 8, par);
        init_stencils(locality_results(stencils[1]), locality_results(functions), 
            1, 6, par);
        init_stencils(locality_results(stencils[2]), locality_results(functions), 
            2, 2, par);

        // ask stencil instances for their output gids
        std::vector<std::vector<std::vector<naming::id_type> > > outputs(3);
        get_output_ports(locality_results(stencils[0]), outputs[0]);
        get_output_ports(locality_results(stencils[1]), outputs[1]);
        get_output_ports(locality_results(stencils[2]), outputs[2]);

        // connect output gids with corresponding stencil inputs
        connect_input_ports(stencils, outputs,par);

        // for loop over second row ; call start for each
        start_row(locality_results(stencils[1]));
        start_row(locality_results(stencils[2]));

        // prepare initial data
        std::vector<naming::id_type> initial_data;
        prepare_initial_data(locality_results(functions), initial_data, par);

        // do actual work
        execute(locality_results(stencils[0]), initial_data, result_data);

        // free all allocated components (we can do that synchronously)
        if (!logging.empty())
            factory.free_components_sync(logging);
        factory.free_components_sync(stencils[2]);
        factory.free_components_sync(stencils[1]);
        factory.free_components_sync(stencils[0]);
        factory.free_components_sync(functions);

        return result_data;
    }

    ///////////////////////////////////////////////////////////////////////////
    /// This the other entry point of this component. 
    std::vector<naming::id_type> amr_mesh_tapered::execute(
        std::vector<naming::id_type> const& initial_data,
        components::component_type function_type, std::size_t numvalues, 
        std::size_t numsteps,
        components::component_type logging_type, Parameter const& par)
    {
        BOOST_ASSERT(numvalues == 8);
        BOOST_ASSERT(numsteps == 2);

        std::vector<naming::id_type> result_data;

        components::component_type stencil_type = 
            components::get_component_type<components::amr::server::dynamic_stencil_value >();

        typedef components::distributing_factory::result_type result_type;

        // create a distributing factory locally
        components::distributing_factory factory(
            components::distributing_factory::create(
                applier::get_applier().get_runtime_support_gid(), true));

        // create a couple of stencil (functional) components and twice the 
        // amount of stencil_value components
        numvalues_ = numvalues;
        result_type functions = factory.create_components(function_type, numvalues);
        result_type stencils[3] = 
        {
            factory.create_components(stencil_type, 8),
            factory.create_components(stencil_type, 6),
            factory.create_components(stencil_type, 2)
        };

        // initialize logging functionality in functions
        result_type logging;
        if (logging_type != components::component_invalid)
            logging = factory.create_components(logging_type);

        init(locality_results(functions), locality_results(logging), numsteps);

        // initialize stencil_values using the stencil (functional) components
        init_stencils(locality_results(stencils[0]), locality_results(functions), 
            0, 8, par);
        init_stencils(locality_results(stencils[1]), locality_results(functions), 
            1, 6, par);
        init_stencils(locality_results(stencils[2]), locality_results(functions), 
            2, 2, par);

        // ask stencil instances for their output gids
        std::vector<std::vector<std::vector<naming::id_type> > > outputs(3);
        get_output_ports(locality_results(stencils[0]), outputs[0]);
        get_output_ports(locality_results(stencils[1]), outputs[1]);
        get_output_ports(locality_results(stencils[2]), outputs[2]);

        // connect output gids with corresponding stencil inputs
        connect_input_ports(stencils, outputs,par);

        // for loop over second row ; call start for each
        start_row(locality_results(stencils[1]));
        start_row(locality_results(stencils[2]));

        // do actual work
        execute(locality_results(stencils[0]), initial_data, result_data);

        // free all allocated components (we can do that synchronously)
        if (!logging.empty())
            factory.free_components_sync(logging);
        factory.free_components_sync(stencils[2]);
        factory.free_components_sync(stencils[1]);
        factory.free_components_sync(stencils[0]);
        factory.free_components_sync(functions);

        return result_data;
    }

}}}}
