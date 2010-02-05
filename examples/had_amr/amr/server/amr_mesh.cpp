//  Copyright (c) 2007-2010 Hartmut Kaiser
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

#include "amr_mesh.hpp"

///////////////////////////////////////////////////////////////////////////////
typedef hpx::components::amr::server::amr_mesh had_amr_mesh_type;

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the actions
HPX_REGISTER_ACTION_EX(had_amr_mesh_type::init_execute_action, had_amr_mesh_init_execute_action);
HPX_REGISTER_ACTION_EX(had_amr_mesh_type::execute_action, had_amr_mesh_execute_action);

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::simple_component<had_amr_mesh_type>, had_amr_mesh);
HPX_DEFINE_GET_COMPONENT_TYPE(had_amr_mesh_type);

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr { namespace server 
{
    amr_mesh::amr_mesh()
      : numvalues_(0)
    {}

    ///////////////////////////////////////////////////////////////////////////////
    // Initialize functional components by setting the logging component to use
    void amr_mesh::init(distributed_iterator_range_type const& functions,
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
    void amr_mesh::init_stencils(distributed_iterator_range_type const& stencils,
        distributed_iterator_range_type const& functions, int static_step, 
        int numvalues, Parameter const& par)
    {
        components::distributing_factory::iterator_type stencil = stencils.first;
        components::distributing_factory::iterator_type function = functions.first;

        if ( par.coarsestencilsize == 5 ) BOOST_ASSERT(numvalues > 9);
        if ( par.coarsestencilsize == 9 ) BOOST_ASSERT(numvalues > 9);

        for (int column = 0; stencil != stencils.second; ++stencil, ++function, ++column)
        {
            namespace stubs = components::amr::stubs;
            BOOST_ASSERT(function != functions.second);
            if ( par.coarsestencilsize == 3 ) {
              if (column == 0 || column == numvalues-1) {
                  // boundary value
                  stubs::dynamic_stencil_value::set_functional_component(*stencil, 
                      *function, static_step, column, 
                      par.coarsestencilsize-1, par.coarsestencilsize-1, par);
              } 
              else {
                  // 'normal' value
                  stubs::dynamic_stencil_value::set_functional_component(*stencil, 
                      *function, static_step, column, 
                      par.coarsestencilsize, par.coarsestencilsize, par);
              }
            } else if ( par.coarsestencilsize == 5 ) {
              if (column == 0 ) {
                  // boundary value
                  stubs::dynamic_stencil_value::set_functional_component(*stencil, 
                      *function, static_step, column, 2,3, par);
              } else if ( column == 1) {
                  stubs::dynamic_stencil_value::set_functional_component(*stencil, 
                      *function, static_step, column,3,4, par);
              } else if ( column == 2 || column == 3) {
                  stubs::dynamic_stencil_value::set_functional_component(*stencil, 
                      *function, static_step, column,5,4, par);
              } else if ( column == numvalues-4 || column == numvalues-3) {
                  stubs::dynamic_stencil_value::set_functional_component(*stencil, 
                      *function, static_step, column,5,4,par);
              } else if ( column == numvalues-2) {
                  stubs::dynamic_stencil_value::set_functional_component(*stencil, 
                      *function, static_step, column,3,4,par);
              } else if ( column == numvalues-1) {
                  stubs::dynamic_stencil_value::set_functional_component(*stencil, 
                      *function, static_step, column, 2,3, par);
              } else {
                  // 'normal' value
                  stubs::dynamic_stencil_value::set_functional_component(*stencil, 
                      *function, static_step, column, 5,5, par);
              }
            } else if ( par.coarsestencilsize == 9 ) {
              if (column == 0 ) {
                  // boundary value
                  stubs::dynamic_stencil_value::set_functional_component(*stencil, 
                      *function, static_step, column, 2,3, par);
              } else if ( column == 1) {
                  stubs::dynamic_stencil_value::set_functional_component(*stencil, 
                      *function, static_step, column,3,5, par);
              } else if ( column == 2 || column == 3) {
                  stubs::dynamic_stencil_value::set_functional_component(*stencil, 
                      *function, static_step, column,3,6, par);
              } else if ( column == 4 || column == 5) {
                  stubs::dynamic_stencil_value::set_functional_component(*stencil, 
                      *function, static_step, column,9,6, par);
              } else if ( column == 6 ) {
                  stubs::dynamic_stencil_value::set_functional_component(*stencil, 
                      *function, static_step, column,9,7, par);
              } else if ( column == 7 ) {
                  stubs::dynamic_stencil_value::set_functional_component(*stencil, 
                      *function, static_step, column,9,8, par);
              } else if ( column == numvalues-8 ) {
                  stubs::dynamic_stencil_value::set_functional_component(*stencil, 
                      *function, static_step, column,9,8,par);
              } else if ( column == numvalues-7 ) {
                  stubs::dynamic_stencil_value::set_functional_component(*stencil, 
                      *function, static_step, column,9,7,par);
              } else if ( column == numvalues-6 || column == numvalues-5) {
                  stubs::dynamic_stencil_value::set_functional_component(*stencil, 
                      *function, static_step, column,9,6,par);
              } else if ( column == numvalues-4 || column == numvalues-3) {
                  stubs::dynamic_stencil_value::set_functional_component(*stencil, 
                      *function, static_step, column,3,6,par);
              } else if ( column == numvalues-2) {
                  stubs::dynamic_stencil_value::set_functional_component(*stencil, 
                      *function, static_step, column,3,5,par);
              } else if ( column == numvalues-1) {
                  stubs::dynamic_stencil_value::set_functional_component(*stencil, 
                      *function, static_step, column, 2,3, par);
              } else {
                  // 'normal' value
                  stubs::dynamic_stencil_value::set_functional_component(*stencil, 
                      *function, static_step, column, 9,9, par);
              }
            } else {
               BOOST_ASSERT (false);
            }
        }
        BOOST_ASSERT(function == functions.second);
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Get gids of output ports of all functions
    void amr_mesh::get_output_ports(
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

    void amr_mesh::connect_input_ports(
        components::distributing_factory::result_type const* stencils,
        std::vector<std::vector<std::vector<naming::id_type> > > const& outputs,
        Parameter const& par)
    {
        typedef components::distributing_factory::result_type result_type;

        int j;
        int *dst_ports;
        int *dst_src;
        int *dst_size;
        std::size_t numvals = outputs[0].size();
        if ( par.coarsestencilsize == 9 ) {
          prep_ports_nine(dst_ports,dst_src,dst_size,numvals); 
        }

        int steps = (int)outputs.size();
        for (int step = 0; step < steps; ++step) 
        {
          //  std::size_t numvals = outputs[0].size();
            components::distributing_factory::iterator_range_type r = 
                locality_results(stencils[step]);
            components::distributing_factory::iterator_type stencil = r.first;
            for (int i = 0; stencil != r.second; ++stencil, ++i)
            {
                using namespace boost::assign;
                int outstep = mod(step-1, steps);

                std::vector<naming::id_type> output_ports;
                if ( par.coarsestencilsize == 3 ) {
                  if ( i==0 ) {
                    output_ports += 
                          outputs[outstep][0][0],  
                          outputs[outstep][1][0];
                  } else if ( i==numvals-1 ) {
                    output_ports += 
                          outputs[outstep][numvals-2][2],  
                          outputs[outstep][numvals-1][1];
                  } else if ( i==1 ) {
                    output_ports += 
                          outputs[outstep][i-1][1], // there are only two inputs at i=0 
                          outputs[outstep][i][1],  
                          outputs[outstep][i+1][0];
                  } else {
                    output_ports += 
                          outputs[outstep][i-1][2],    // sw input is connected to the ne output of the sw element
                          outputs[outstep][i][1],    // s input is connected to the n output of the s element 
                          outputs[outstep][i+1][0]    // se input is connected to the nw output of the se element
                      ;
                  }
                } else if ( par.coarsestencilsize == 5 ) {
                  if ( i==0 ) {
                    output_ports += 
                          outputs[outstep][0][1],  
                          outputs[outstep][1][3];
                  } else if ( i==1 ) {
                    output_ports += 
                          outputs[outstep][i-1][2], 
                          outputs[outstep][i][2],  
                          outputs[outstep][i+1][3];
                  } else if ( i==numvals-6 ) {
                    output_ports += 
                          outputs[outstep][i-2][0], 
                          outputs[outstep][i-1][1], 
                          outputs[outstep][i][2],  
                          outputs[outstep][i+1][3],
                          outputs[outstep][i+2][0];
                  } else if ( i==numvals-5 ) {
                    output_ports += 
                          outputs[outstep][i-2][0], 
                          outputs[outstep][i-1][1], 
                          outputs[outstep][i][2],  
                          outputs[outstep][i+1][3],
                          outputs[outstep][i+2][3];
                  } else if ( i==numvals-4 ) {
                    output_ports += 
                          outputs[outstep][i-2][0], 
                          outputs[outstep][i-1][1], 
                          outputs[outstep][i][2],  
                          outputs[outstep][i+1][2],
                          outputs[outstep][i+2][3];
                  } else if ( i==numvals-3 ) {
                    output_ports += 
                          outputs[outstep][i-2][0], 
                          outputs[outstep][i-1][1], 
                          outputs[outstep][i][1],  
                          outputs[outstep][i+1][2],
                          outputs[outstep][i+2][2];
                  } else if ( i==numvals-2 ) {
                    output_ports += 
                          outputs[outstep][i-1][0],
                          outputs[outstep][i][0],  
                          outputs[outstep][i+1][0];
                  } else if ( i==numvals-1 ) {
                    output_ports += 
                          outputs[outstep][i-1][1],
                          outputs[outstep][i][1];  
                  } else {
                    output_ports += 
                          outputs[outstep][i-2][0],
                          outputs[outstep][i-1][1],  
                          outputs[outstep][i][2],
                          outputs[outstep][i+1][3],
                          outputs[outstep][i+2][4];
                  }
                } else if ( par.coarsestencilsize == 9 ) {
                  // get src ports in ascending order
                  for (j=0;j<dst_size[i];j++) {
                    output_ports.push_back(outputs[outstep][dst_src[i+numvals*j]][dst_ports[i+numvals*j]]);
                  }
                } else {
                  BOOST_ASSERT(false);
                }

                components::amr::stubs::dynamic_stencil_value::
                    connect_input_ports(*stencil, output_ports);
             //     components::amr::stubs::stencil_value<3>::
             //         connect_input_ports(*stencil, output_ports);
            }
        }
        if ( par.coarsestencilsize == 9 ) {
          delete [] dst_ports;
          delete [] dst_src;
          delete [] dst_size;
        }
    }

    ///////////////////////////////////////////////////////////////////////////////
    void amr_mesh::prepare_initial_data(
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
                alloc_data_async(*function, i, numvalues_, 0, 0, 0.0, par));
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
    void amr_mesh::execute(
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
    void amr_mesh::start_row(
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
    std::vector<naming::id_type> amr_mesh::init_execute(
        components::component_type function_type, std::size_t numvalues, 
        std::size_t numsteps,
        components::component_type logging_type, Parameter const& par)
    {
        std::vector<naming::id_type> result_data;

        components::component_type stencil_type = 
            components::get_component_type<components::amr::server::dynamic_stencil_value >();

        typedef components::distributing_factory::result_type result_type;

        // create a distributing factory locally
        components::distributing_factory factory;
        factory.create(applier::get_applier().get_runtime_support_gid(), true);

        // create a couple of stencil (functional) components and twice the 
        // amount of stencil_value components
        numvalues_ = numvalues;
        result_type functions = factory.create_components(function_type, numvalues);
        result_type stencils[2] = 
        {
            factory.create_components(stencil_type, numvalues),
            factory.create_components(stencil_type, numvalues)
        };

        // initialize logging functionality in functions
        result_type logging;
        if (logging_type != components::component_invalid)
            logging = factory.create_components(logging_type);

        init(locality_results(functions), locality_results(logging), numsteps);

        // initialize stencil_values using the stencil (functional) components
        init_stencils(locality_results(stencils[0]), locality_results(functions), 
            0, numvalues, par);
        init_stencils(locality_results(stencils[1]), locality_results(functions), 
            1, numvalues, par);

        // ask stencil instances for their output gids
        std::vector<std::vector<std::vector<naming::id_type> > > outputs(2);
        get_output_ports(locality_results(stencils[0]), outputs[0]);
        get_output_ports(locality_results(stencils[1]), outputs[1]);

        // connect output gids with corresponding stencil inputs
        connect_input_ports(stencils, outputs,par);

        // for loop over second row ; call start for each
        start_row(locality_results(stencils[1]));

        // prepare initial data
        std::vector<naming::id_type> initial_data;
        prepare_initial_data(locality_results(functions), initial_data, par);

        // do actual work
        execute(locality_results(stencils[0]), initial_data, result_data);

        // free all allocated components (we can do that synchronously)
        if (!logging.empty())
            factory.free_components_sync(logging);
        factory.free_components_sync(stencils[1]);
        factory.free_components_sync(stencils[0]);
        factory.free_components_sync(functions);

        return result_data;
    }

    ///////////////////////////////////////////////////////////////////////////
    /// This the other entry point of this component. 
    std::vector<naming::id_type> amr_mesh::execute(
        std::vector<naming::id_type> const& initial_data,
        components::component_type function_type, std::size_t numvalues, 
        std::size_t numsteps,
        components::component_type logging_type, Parameter const& par)
    {
        std::vector<naming::id_type> result_data;

        components::component_type stencil_type = 
            components::get_component_type<components::amr::server::dynamic_stencil_value >();

        typedef components::distributing_factory::result_type result_type;

        // create a distributing factory locally
        components::distributing_factory factory;
        factory.create(applier::get_applier().get_runtime_support_gid(), true);

        // create a couple of stencil (functional) components and twice the 
        // amount of stencil_value components
        numvalues_ = numvalues;
        result_type functions = factory.create_components(function_type, numvalues);
        result_type stencils[2] = 
        {
            factory.create_components(stencil_type, numvalues),
            factory.create_components(stencil_type, numvalues)
        };

        // initialize logging functionality in functions
        result_type logging;
        if (logging_type != components::component_invalid)
            logging = factory.create_components(logging_type);

        init(locality_results(functions), locality_results(logging), numsteps);

        // initialize stencil_values using the stencil (functional) components
        init_stencils(locality_results(stencils[0]), locality_results(functions), 
            0, numvalues, par);
        init_stencils(locality_results(stencils[1]), locality_results(functions), 
            1, numvalues, par);

        // ask stencil instances for their output gids
        std::vector<std::vector<std::vector<naming::id_type> > > outputs(2);
        get_output_ports(locality_results(stencils[0]), outputs[0]);
        get_output_ports(locality_results(stencils[1]), outputs[1]);

        // connect output gids with corresponding stencil inputs
        connect_input_ports(stencils, outputs,par);

        // for loop over second row ; call start for each
        start_row(locality_results(stencils[1]));

        // do actual work
        execute(locality_results(stencils[0]), initial_data, result_data);

        // free all allocated components (we can do that synchronously)
        if (!logging.empty())
            factory.free_components_sync(logging);
        factory.free_components_sync(stencils[1]);
        factory.free_components_sync(stencils[0]);
        factory.free_components_sync(functions);

        return result_data;
    }

    void amr_mesh::prep_ports_nine(int *dst_ports,int *dst_src,int *dst_size,std::size_t numvals ) { 
      int j,k;
      dst_ports = new int[numvals*9];
      dst_src = new int[numvals*9];
      dst_size = new int[numvals];
      // initialize size to zero
      for (j=0;j<numvals;j++) {
        dst_size[j] = 0;
      }

      for (j=0;j<numvals;j++) {
        if ( j == 0 ) {
          dst_ports[j + numvals*dst_size[j]] = 0;
            dst_src[j + numvals*dst_size[j]] = j;
           dst_size[j]++;

          dst_ports[j+1 + numvals*dst_size[j+1]] = 1;
            dst_src[j+1 + numvals*dst_size[j+1]] = j;
           dst_size[j+1]++;

          dst_ports[j+4 + numvals*dst_size[j+4]] = 2;
            dst_src[j+4 + numvals*dst_size[j+4]] = j;
           dst_size[j+4]++;
        } else if ( j == 1 ) {
          dst_ports[j-1 + numvals*dst_size[j-1]] = 0;
            dst_src[j-1 + numvals*dst_size[j-1]] = j;
           dst_size[j-1]++;

          dst_ports[j + numvals*dst_size[j]] = 1;
            dst_src[j + numvals*dst_size[j]] = j;
           dst_size[j]++;

          dst_ports[j+1 + numvals*dst_size[j+1]] = 2;
            dst_src[j+1 + numvals*dst_size[j+1]] = j;
           dst_size[j+1]++;
          
          dst_ports[j+3 + numvals*dst_size[j+3]] = 3;
            dst_src[j+3 + numvals*dst_size[j+3]] = j;
           dst_size[j+3]++;

          dst_ports[j+4 + numvals*dst_size[j+4]] = 4;
            dst_src[j+4 + numvals*dst_size[j+4]] = j;
           dst_size[j+4]++;
        } else if ( j == 2 || j == 3 || j == 4 || j == 5) {
          dst_ports[j-1 + numvals*dst_size[j-1]] = 0;
            dst_src[j-1 + numvals*dst_size[j-1]] = j;
           dst_size[j-1]++;

          dst_ports[j + numvals*dst_size[j]] = 1;
            dst_src[j + numvals*dst_size[j]] = j;
           dst_size[j]++;

          dst_ports[j+1 + numvals*dst_size[j+1]] = 2;
            dst_src[j+1 + numvals*dst_size[j+1]] = j;
           dst_size[j+1]++;

          dst_ports[j+2 + numvals*dst_size[j+2]] = 3;
            dst_src[j+2 + numvals*dst_size[j+2]] = j;
           dst_size[j+2]++;

          dst_ports[j+3 + numvals*dst_size[j+3]] = 4;
            dst_src[j+3 + numvals*dst_size[j+3]] = j;
           dst_size[j+3]++;

          dst_ports[j+4 + numvals*dst_size[j+4]] = 5;
            dst_src[j+4 + numvals*dst_size[j+4]] = j;
           dst_size[j+4]++;
        } else if ( j == 6 ) {
          dst_ports[j-2 + numvals*dst_size[j-2]] = 0;
            dst_src[j-2 + numvals*dst_size[j-2]] = j;
           dst_size[j-2]++;

          dst_ports[j-1 + numvals*dst_size[j-1]] = 1;
            dst_src[j-1 + numvals*dst_size[j-1]] = j;
           dst_size[j-1]++;

          dst_ports[j + numvals*dst_size[j]] = 2;
            dst_src[j + numvals*dst_size[j]] = j;
           dst_size[j]++;

          dst_ports[j+1 + numvals*dst_size[j+1]] = 3;
            dst_src[j+1 + numvals*dst_size[j+1]] = j;
           dst_size[j+1]++;

          dst_ports[j+2 + numvals*dst_size[j+2]] = 4;
            dst_src[j+2 + numvals*dst_size[j+2]] = j;
           dst_size[j+2]++;

          dst_ports[j+3 + numvals*dst_size[j+3]] = 5;
            dst_src[j+3 + numvals*dst_size[j+3]] = j;
           dst_size[j+3]++;

          dst_ports[j+4 + numvals*dst_size[j+4]] = 6;
            dst_src[j+4 + numvals*dst_size[j+4]] = j;
           dst_size[j+4]++;
        } else if ( j == 7 ) {
          dst_ports[j-3 + numvals*dst_size[j-3]] = 0;
            dst_src[j-3 + numvals*dst_size[j-3]] = j;
           dst_size[j-3]++;

          dst_ports[j-2 + numvals*dst_size[j-2]] = 1;
            dst_src[j-2 + numvals*dst_size[j-2]] = j;
           dst_size[j-2]++;

          dst_ports[j-1 + numvals*dst_size[j-1]] = 2;
            dst_src[j-1 + numvals*dst_size[j-1]] = j;
           dst_size[j-1]++;

          dst_ports[j + numvals*dst_size[j]] = 3;
            dst_src[j + numvals*dst_size[j]] = j;
           dst_size[j]++;

          dst_ports[j+1 + numvals*dst_size[j+1]] = 4;
            dst_src[j+1 + numvals*dst_size[j+1]] = j;
           dst_size[j+1]++;

          dst_ports[j+2 + numvals*dst_size[j+2]] = 5;
            dst_src[j+2 + numvals*dst_size[j+2]] = j;
           dst_size[j+2]++;

          dst_ports[j+3 + numvals*dst_size[j+3]] = 6;
            dst_src[j+3 + numvals*dst_size[j+3]] = j;
           dst_size[j+3]++;

          dst_ports[j+4 + numvals*dst_size[j+4]] = 7;
            dst_src[j+4 + numvals*dst_size[j+4]] = j;
           dst_size[j+4]++;
        } else if ( j == numvals-8 ) {
          dst_ports[j-4 + numvals*dst_size[j-4]] = 0;
            dst_src[j-4 + numvals*dst_size[j-4]] = j;
           dst_size[j-4]++;

          dst_ports[j-3 + numvals*dst_size[j-3]] = 1;
            dst_src[j-3 + numvals*dst_size[j-3]] = j;
           dst_size[j-3]++;

          dst_ports[j-2 + numvals*dst_size[j-2]] = 2;
            dst_src[j-2 + numvals*dst_size[j-2]] = j;
           dst_size[j-2]++;

          dst_ports[j-1 + numvals*dst_size[j-1]] = 3;
            dst_src[j-1 + numvals*dst_size[j-1]] = j;
           dst_size[j-1]++;

          dst_ports[j + numvals*dst_size[j]] = 4;
            dst_src[j + numvals*dst_size[j]] = j;
           dst_size[j]++;

          dst_ports[j+1 + numvals*dst_size[j+1]] = 5;
            dst_src[j+1 + numvals*dst_size[j+1]] = j;
           dst_size[j+1]++;

          dst_ports[j+2 + numvals*dst_size[j+2]] = 6;
            dst_src[j+2 + numvals*dst_size[j+2]] = j;
           dst_size[j+2]++;

          dst_ports[j+3 + numvals*dst_size[j+3]] = 7;
            dst_src[j+3 + numvals*dst_size[j+3]] = j;
           dst_size[j+3]++;
        } else if ( j == numvals-7 ) {
          dst_ports[j-4 + numvals*dst_size[j-4]] = 0;
            dst_src[j-4 + numvals*dst_size[j-4]] = j;
           dst_size[j-4]++;

          dst_ports[j-3 + numvals*dst_size[j-3]] = 1;
            dst_src[j-3 + numvals*dst_size[j-3]] = j;
           dst_size[j-3]++;

          dst_ports[j-2 + numvals*dst_size[j-2]] = 2;
            dst_src[j-2 + numvals*dst_size[j-2]] = j;
           dst_size[j-2]++;

          dst_ports[j-1 + numvals*dst_size[j-1]] = 3;
            dst_src[j-1 + numvals*dst_size[j-1]] = j;
           dst_size[j-1]++;

          dst_ports[j + numvals*dst_size[j]] = 4;
            dst_src[j + numvals*dst_size[j]] = j;
           dst_size[j]++;

          dst_ports[j+1 + numvals*dst_size[j+1]] = 5;
            dst_src[j+1 + numvals*dst_size[j+1]] = j;
           dst_size[j+1]++;

          dst_ports[j+2 + numvals*dst_size[j+2]] = 6;
            dst_src[j+2 + numvals*dst_size[j+2]] = j;
           dst_size[j+2]++;
        } else if ( j == numvals-6 || j == numvals-5 || j == numvals-4 || j == numvals-3 ) {
          dst_ports[j-4 + numvals*dst_size[j-4]] = 0;
            dst_src[j-4 + numvals*dst_size[j-4]] = j;
           dst_size[j-4]++;

          dst_ports[j-3 + numvals*dst_size[j-3]] = 1;
            dst_src[j-3 + numvals*dst_size[j-3]] = j;
           dst_size[j-3]++;

          dst_ports[j-2 + numvals*dst_size[j-2]] = 2;
            dst_src[j-2 + numvals*dst_size[j-2]] = j;
           dst_size[j-2]++;

          dst_ports[j-1 + numvals*dst_size[j-1]] = 3;
            dst_src[j-1 + numvals*dst_size[j-1]] = j;
           dst_size[j-1]++;

          dst_ports[j + numvals*dst_size[j]] = 4;
            dst_src[j + numvals*dst_size[j]] = j;
           dst_size[j]++;

          dst_ports[j+1 + numvals*dst_size[j+1]] = 5;
            dst_src[j+1 + numvals*dst_size[j+1]] = j;
           dst_size[j+1]++;
        } else if ( j == numvals-2 ) {
          dst_ports[j-4 + numvals*dst_size[j-4]] = 0;
            dst_src[j-4 + numvals*dst_size[j-4]] = j;
           dst_size[j-4]++;

          dst_ports[j-3 + numvals*dst_size[j-3]] = 1;
            dst_src[j-3 + numvals*dst_size[j-3]] = j;
           dst_size[j-3]++;

          dst_ports[j-1 + numvals*dst_size[j-1]] = 2;
            dst_src[j-1 + numvals*dst_size[j-1]] = j;
           dst_size[j-1]++;

          dst_ports[j + numvals*dst_size[j]] = 3;
            dst_src[j + numvals*dst_size[j]] = j;
           dst_size[j]++;

          dst_ports[j+1 + numvals*dst_size[j+1]] = 4;
            dst_src[j+1 + numvals*dst_size[j+1]] = j;
           dst_size[j+1]++;
        } else if ( j == numvals-1 ) {
          dst_ports[j-4 + numvals*dst_size[j-4]] = 0;
            dst_src[j-4 + numvals*dst_size[j-4]] = j;
           dst_size[j-4]++;

          dst_ports[j-1 + numvals*dst_size[j-1]] = 1;
            dst_src[j-1 + numvals*dst_size[j-1]] = j;
           dst_size[j-1]++;

          dst_ports[j + numvals*dst_size[j]] = 2;
            dst_src[j + numvals*dst_size[j]] = j;
           dst_size[j]++;
        } else {
          dst_ports[j-4 + numvals*dst_size[j-4]] = 0;
            dst_src[j-4 + numvals*dst_size[j-4]] = j;
           dst_size[j-4]++;

          dst_ports[j-3 + numvals*dst_size[j-3]] = 1;
            dst_src[j-3 + numvals*dst_size[j-3]] = j;
           dst_size[j-3]++;

          dst_ports[j-2 + numvals*dst_size[j-2]] = 2;
            dst_src[j-2 + numvals*dst_size[j-2]] = j;
           dst_size[j-2]++;

          dst_ports[j-1 + numvals*dst_size[j-1]] = 3;
            dst_src[j-1 + numvals*dst_size[j-1]] = j;
           dst_size[j-1]++;

          dst_ports[j + numvals*dst_size[j]] = 4;
            dst_src[j + numvals*dst_size[j]] = j;
           dst_size[j]++;

          dst_ports[j+1 + numvals*dst_size[j+1]] = 5;
            dst_src[j+1 + numvals*dst_size[j+1]] = j;
           dst_size[j+1]++;

          dst_ports[j+2 + numvals*dst_size[j+2]] = 6;
            dst_src[j+2 + numvals*dst_size[j+2]] = j;
           dst_size[j+2]++;

          dst_ports[j+3 + numvals*dst_size[j+3]] = 7;
            dst_src[j+3 + numvals*dst_size[j+3]] = j;
           dst_size[j+3]++;

          dst_ports[j+4 + numvals*dst_size[j+4]] = 8;
            dst_src[j+4 + numvals*dst_size[j+4]] = j;
           dst_size[j+4]++;
        }
      }

      int t1;
      for (j=0;j<numvals;j++) {
        // Now sort the src ports in ascending order for each destination port
        for (k=0;k<dst_size[j]-1;k++) {
          if (dst_src[j+numvals*k] > dst_src[j+numvals*(k+1)]) {
            // swap
            t1 = dst_src[j+numvals*k];
            dst_src[j+numvals*k] = dst_src[j+numvals*(k+1)];
            dst_src[j+numvals*(k+1)] = t1;

            // swap the port too
            t1 = dst_ports[j+numvals*k];
            dst_ports[j+numvals*k] = dst_ports[j+numvals*(k+1)];
            dst_ports[j+numvals*(k+1)] = t1;

          }
        }
      }

    }

}}}}
