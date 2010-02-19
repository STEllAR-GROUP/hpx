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

#include "rk_left.hpp"

///////////////////////////////////////////////////////////////////////////////
typedef hpx::components::amr::server::rk_left had_rk_left_type;

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the actions
HPX_REGISTER_ACTION_EX(had_rk_left_type::init_execute_action, had_rk_left_init_execute_action);
HPX_REGISTER_ACTION_EX(had_rk_left_type::execute_action, had_rk_left_execute_action);

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::simple_component<had_rk_left_type>, had_rk_left);
HPX_DEFINE_GET_COMPONENT_TYPE(had_rk_left_type);

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr { namespace server 
{
    rk_left::rk_left()
      : numvalues_(0)
    {}

    ///////////////////////////////////////////////////////////////////////////////
    // Initialize functional components by setting the logging component to use
    void rk_left::init(distributed_iterator_range_type const& functions,
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
    void rk_left::init_stencils(distributed_iterator_range_type const& stencils,
        distributed_iterator_range_type const& functions, int static_step, 
        int numvalues, Parameter const& par)
    {
        components::distributing_factory::iterator_type stencil = stencils.first;
        components::distributing_factory::iterator_type function = functions.first;

        BOOST_ASSERT(par.stencilsize == 3);
        BOOST_ASSERT(par.coarsestencilsize == 9);

        for (int column = 0; stencil != stencils.second; ++stencil, ++function, ++column)
        {
            namespace stubs = components::amr::stubs;
            BOOST_ASSERT(function != functions.second);
            if ( numvalues == 17 ) {
              if (column == 0 || column == 16) {
                stubs::dynamic_stencil_value::set_functional_component(*stencil,
                                                *function, static_step, column, 1,2, par);
              } else if (column == 1 || column == 15) {
                stubs::dynamic_stencil_value::set_functional_component(*stencil,
                                                *function, static_step, column, 1,4, par);
              } else if (column == 2 || column == 14) {
                stubs::dynamic_stencil_value::set_functional_component(*stencil,
                                                *function, static_step, column, 1,6, par);
              } else if (column == 3 || column == 13) {
                stubs::dynamic_stencil_value::set_functional_component(*stencil,
                                                *function, static_step, column, 1,7, par);
              } else if (column == 4 || column == 12) {
                stubs::dynamic_stencil_value::set_functional_component(*stencil,
                                                *function, static_step, column, 1,8, par);
              } else if (column == 5 || column == 11) {
                stubs::dynamic_stencil_value::set_functional_component(*stencil,
                                                *function, static_step, column, 1,9, par);
              } else if (column == 6 || column == 10) {
                stubs::dynamic_stencil_value::set_functional_component(*stencil,
                                                *function, static_step, column, 1,10, par);
              } else if (column == 7 || column == 9) {
                stubs::dynamic_stencil_value::set_functional_component(*stencil,
                                                *function, static_step, column, 1,11, par);
              } else if (column == 8 ) {
                stubs::dynamic_stencil_value::set_functional_component(*stencil,
                                                *function, static_step, column, 1,12, par);
              } else {
                // this shouldn't happen
                BOOST_ASSERT(false);
              }
            } else if ( numvalues == 15 ) {
              if (column == 0 || column == 14) {
                stubs::dynamic_stencil_value::set_functional_component(*stencil,
                                                *function, static_step, column, 3,1, par);
              } else if ( column == 1 || column == 13 ) {
                stubs::dynamic_stencil_value::set_functional_component(*stencil,
                                                *function, static_step, column, 3,2, par);
              } else if ( column > 1 && column < 13 ) {
                stubs::dynamic_stencil_value::set_functional_component(*stencil,
                                                *function, static_step, column, 3,3, par);
              } else {
                // this shouldn't happen
                BOOST_ASSERT(false);
              }
            } else if ( numvalues == 13 ) {
              if (column == 0 || column == 12) {
                stubs::dynamic_stencil_value::set_functional_component(*stencil,
                                                *function, static_step, column, 3,1, par);
              } else if (column == 1 || column == 11) {
                stubs::dynamic_stencil_value::set_functional_component(*stencil,
                                                *function, static_step, column, 3,2, par);
              } else if (column > 1 && column < 11) {
                stubs::dynamic_stencil_value::set_functional_component(*stencil,
                                                *function, static_step, column, 3,3, par);
              } else {
                // this shouldn't happen
                BOOST_ASSERT(false);
              }
            } else if ( numvalues == 11 ) {
              if (column == 0 || column == 10) {
                stubs::dynamic_stencil_value::set_functional_component(*stencil,
                                                *function, static_step, column, 3,1, par);
              } else if (column == 1 || column == 9) {
                stubs::dynamic_stencil_value::set_functional_component(*stencil,
                                                *function, static_step, column, 12,3, par);
              } else if (column == 2 || column == 8) {
                stubs::dynamic_stencil_value::set_functional_component(*stencil,
                                                *function, static_step, column, 12,5, par);
              } else if (column > 2 && column < 8) {
                stubs::dynamic_stencil_value::set_functional_component(*stencil,
                                                *function, static_step, column, 12,6, par);
              } else {
                // this shouldn't happen
                BOOST_ASSERT(false);
              }
            } else if ( numvalues == 7 ) {
              if (column == 0 || column == 6) {
                stubs::dynamic_stencil_value::set_functional_component(*stencil,
                                                *function, static_step, column, 3,1, par);
              } else if (column == 1 || column == 5) {
                stubs::dynamic_stencil_value::set_functional_component(*stencil,
                                                *function, static_step, column, 3,2, par);
              } else if (column > 1 && column < 5) {
                stubs::dynamic_stencil_value::set_functional_component(*stencil,
                                                *function, static_step, column, 3,3, par);
              } else {
                // this shouldn't happen
                BOOST_ASSERT(false);
              }
            } else if ( numvalues == 5 ) {
              if (column == 0 || column == 4) {
                stubs::dynamic_stencil_value::set_functional_component(*stencil,
                                                *function, static_step, column, 3,1, par);
              } else if (column == 1 || column == 3) {
                stubs::dynamic_stencil_value::set_functional_component(*stencil,
                                                *function, static_step, column, 3,2, par);
              } else if (column == 2 ) {
                stubs::dynamic_stencil_value::set_functional_component(*stencil,
                                                *function, static_step, column, 3,3, par);
              } else {
                // this shouldn't happen
                BOOST_ASSERT(false);
              }
            } else if ( numvalues == 3 ) {
              if (column == 0 || column == 2) {
                stubs::dynamic_stencil_value::set_functional_component(*stencil,
                                                *function, static_step, column, 12,6, par);
              } else if ( column == 1 ) {
                stubs::dynamic_stencil_value::set_functional_component(*stencil,
                                                *function, static_step, column, 12,5, par);
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
    void rk_left::get_output_ports(
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

    void rk_left::connect_input_ports(
        components::distributing_factory::result_type const* stencils,
        std::vector<std::vector<std::vector<naming::id_type> > > const& outputs,
        Parameter const& par)
    {
        BOOST_ASSERT(par.stencilsize == 3);
        BOOST_ASSERT(par.coarsestencilsize == 9);

        typedef components::distributing_factory::result_type result_type;

        std::size_t numvals = outputs[0].size();
        Array3D dst_port(7,numvals,12);
        Array3D dst_src(7,numvals,12);
        Array3D dst_step(7,numvals,12);
        Array3D dst_size(7,numvals,1);
        prep_ports(dst_port,dst_src,dst_step,dst_size);

        int steps = (int)outputs.size();
        BOOST_ASSERT(steps == 7);

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
                components::amr::stubs::dynamic_stencil_value::
                    connect_input_ports(*stencil, output_ports);
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////////
    void rk_left::prepare_initial_data(
        distributed_iterator_range_type const& functions, 
        std::vector<naming::id_type>& initial_data,
        std::size_t level, double x, 
        Parameter const& par)
    {
        typedef std::vector<lcos::future_value<naming::id_type> > lazyvals_type;

        // create a data item value type for each of the functions
        lazyvals_type lazyvals;
        components::distributing_factory::iterator_type function = functions.first;

        for (std::size_t i = 0; function != functions.second; ++function, ++i)
        {
            lazyvals.push_back(components::amr::stubs::functional_component::
                alloc_data_async(*function, i, numvalues_, 0, level, x, par));
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
    void rk_left::execute(
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
    void rk_left::start_row(
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
    std::vector<naming::id_type> rk_left::init_execute(
        components::component_type function_type,
        components::component_type logging_type, 
        std::size_t level, double x, Parameter const& par)
    {
        std::size_t numvalues = 17;
        std::size_t numsteps = 6;
        int i;

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
        result_type stencils[7] = 
        {
            factory.create_components(stencil_type, 17),
            factory.create_components(stencil_type, 15),
            factory.create_components(stencil_type, 13),
            factory.create_components(stencil_type, 11),
            factory.create_components(stencil_type, 7),
            factory.create_components(stencil_type, 5),
            factory.create_components(stencil_type, 3)
        };

        // initialize logging functionality in functions
        result_type logging;
        if (logging_type != components::component_invalid)
            logging = factory.create_components(logging_type);

        init(locality_results(functions), locality_results(logging), numsteps);

        // initialize stencil_values using the stencil (functional) components
        init_stencils(locality_results(stencils[0]), locality_results(functions), 0, 17, par);
        init_stencils(locality_results(stencils[1]), locality_results(functions), 1, 15, par);
        init_stencils(locality_results(stencils[2]), locality_results(functions), 2, 13, par);
        init_stencils(locality_results(stencils[3]), locality_results(functions), 3, 11, par);
        init_stencils(locality_results(stencils[4]), locality_results(functions), 4, 7, par);
        init_stencils(locality_results(stencils[5]), locality_results(functions), 5, 5, par);
        init_stencils(locality_results(stencils[6]), locality_results(functions), 6, 3, par);

        // ask stencil instances for their output gids
        std::vector<std::vector<std::vector<naming::id_type> > > outputs(7);
        for (i=0;i<=6;i++) get_output_ports(locality_results(stencils[i]), outputs[i]);

        // connect output gids with corresponding stencil inputs
        connect_input_ports(stencils, outputs,par);

        // for loop over second row ; call start for each
        for (i=1;i<=6;i++) start_row(locality_results(stencils[i]));

        // prepare initial data
        std::vector<naming::id_type> initial_data;
        prepare_initial_data(locality_results(functions), initial_data, level, x, par);

        // do actual work
        execute(locality_results(stencils[0]), initial_data, result_data);

        // free all allocated components (we can do that synchronously)
        if (!logging.empty())
            factory.free_components_sync(logging);
        for (i=6;i>=0;i--) factory.free_components_sync(stencils[i]);
        factory.free_components_sync(functions);

        return result_data;
    }

    ///////////////////////////////////////////////////////////////////////////
    /// This the other entry point of this component. 
    std::vector<naming::id_type> rk_left::execute(
        std::vector<naming::id_type> const& initial_data,
        components::component_type function_type,
        components::component_type logging_type, Parameter const& par)
    {
        std::size_t numvalues = 17;
        std::size_t numsteps = 6;
        int i;

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
        result_type stencils[7] = 
        {
            factory.create_components(stencil_type, 17),
            factory.create_components(stencil_type, 15),
            factory.create_components(stencil_type, 13),
            factory.create_components(stencil_type, 11),
            factory.create_components(stencil_type, 7),
            factory.create_components(stencil_type, 5),
            factory.create_components(stencil_type, 3)
        };

        // initialize logging functionality in functions
        result_type logging;
        if (logging_type != components::component_invalid)
            logging = factory.create_components(logging_type);

        init(locality_results(functions), locality_results(logging), numsteps);

        // initialize stencil_values using the stencil (functional) components
        init_stencils(locality_results(stencils[0]), locality_results(functions), 0, 17, par);
        init_stencils(locality_results(stencils[1]), locality_results(functions), 1, 15, par);
        init_stencils(locality_results(stencils[2]), locality_results(functions), 2, 13, par);
        init_stencils(locality_results(stencils[3]), locality_results(functions), 3, 11, par);
        init_stencils(locality_results(stencils[4]), locality_results(functions), 4, 7, par);
        init_stencils(locality_results(stencils[5]), locality_results(functions), 5, 5, par);
        init_stencils(locality_results(stencils[6]), locality_results(functions), 6, 3, par);

        // ask stencil instances for their output gids
        std::vector<std::vector<std::vector<naming::id_type> > > outputs(7);
        for (i=0;i<=6;i++) get_output_ports(locality_results(stencils[i]), outputs[i]);

        // connect output gids with corresponding stencil inputs
        connect_input_ports(stencils, outputs,par);

        // for loop over second row ; call start for each
        for (i=1;i<=6;i++) start_row(locality_results(stencils[i]));

        // do actual work
        execute(locality_results(stencils[0]), initial_data, result_data);

        // free all allocated components (we can do that synchronously)
        if (!logging.empty())
            factory.free_components_sync(logging);
        for (i=6;i>=0;i--) factory.free_components_sync(stencils[i]);
        factory.free_components_sync(functions);

        return result_data;
    }

    void rk_left::prep_ports(Array3D &dst_port,Array3D &dst_src,
                             Array3D &dst_step,Array3D &dst_size)
    {
      int i,j;

      // vcolumn is the destination column number
      // vstep is the destination step (or row) number
      // vsrc_column is the source column number
      // vsrc_step is the source step number
      // vport is the output port number; increases consecutively
      std::vector<int> vcolumn,vstep,vsrc_column,vsrc_step,vport;

      // connect outputs for the zeroth row (the zeroth row outputs to the first row *and* the third row)
      vsrc_step.push_back(0);vsrc_column.push_back(0);vstep.push_back(1);vcolumn.push_back(0);vport.push_back(0);
      vsrc_step.push_back(0);vsrc_column.push_back(0);vstep.push_back(3);vcolumn.push_back(1);vport.push_back(1);

      vsrc_step.push_back(0);vsrc_column.push_back(1);vstep.push_back(1);vcolumn.push_back(0);vport.push_back(0);
      vsrc_step.push_back(0);vsrc_column.push_back(1);vstep.push_back(1);vcolumn.push_back(1);vport.push_back(1);
      vsrc_step.push_back(0);vsrc_column.push_back(1);vstep.push_back(3);vcolumn.push_back(1);vport.push_back(2);
      vsrc_step.push_back(0);vsrc_column.push_back(1);vstep.push_back(3);vcolumn.push_back(2);vport.push_back(3);

      for (i=2;i<15;i++) {
        vsrc_step.push_back(0);vsrc_column.push_back(i);vstep.push_back(1);vcolumn.push_back(i-2);vport.push_back(0);
        vsrc_step.push_back(0);vsrc_column.push_back(i);vstep.push_back(1);vcolumn.push_back(i-1);vport.push_back(1);
        vsrc_step.push_back(0);vsrc_column.push_back(i);vstep.push_back(1);vcolumn.push_back(i  );vport.push_back(2);
      }

      int counter;
      for (i=2;i<15;i++) {
        counter = 3;  // counter starts at 3 because this the first three output ports were already used in the lines above
        for (j=i-6;j<i+2;j++) {
          if ( j > 0 && j < 10 ) {
            vsrc_step.push_back(0);vsrc_column.push_back(i);vstep.push_back(3);vcolumn.push_back(j);vport.push_back(counter);
            counter++;
          }
        }
      }

      vsrc_step.push_back(0);vsrc_column.push_back(15);vstep.push_back(1);vcolumn.push_back(13);vport.push_back(0);
      vsrc_step.push_back(0);vsrc_column.push_back(15);vstep.push_back(1);vcolumn.push_back(14);vport.push_back(1);
      vsrc_step.push_back(0);vsrc_column.push_back(15);vstep.push_back(3);vcolumn.push_back(8);vport.push_back(2);
      vsrc_step.push_back(0);vsrc_column.push_back(15);vstep.push_back(3);vcolumn.push_back(9);vport.push_back(3);

      vsrc_step.push_back(0);vsrc_column.push_back(16);vstep.push_back(1);vcolumn.push_back(14);vport.push_back(0);
      vsrc_step.push_back(0);vsrc_column.push_back(16);vstep.push_back(3);vcolumn.push_back(9);vport.push_back(1);

      // connect outputs for the first row (the first row only outputs to the second row)
      for (i=0;i<15;i++) {
        counter = 0;  
        for (j=i-2;j<i+1;j++) {
          if ( j >= 0 && j <= 12 ) {
            vsrc_step.push_back(1);vsrc_column.push_back(i);vstep.push_back(2);vcolumn.push_back(j);vport.push_back(counter);
            counter++;
          }
        }
      }

      // connect outputs for the second row (the second row only outputs to the third row)
      for (i=0;i<13;i++) {
        counter = 0;  
        for (j=i-2;j<i+1;j++) {
          if ( j >= 0 && j <= 10 ) {
            vsrc_step.push_back(2);vsrc_column.push_back(i);vstep.push_back(3);vcolumn.push_back(j);vport.push_back(counter);
            counter++;
          }
        }
      }

      // connect outputs for the third row (the third row outputs to the fourth row *and* the sixth row)
      vsrc_step.push_back(3);vsrc_column.push_back(0);vstep.push_back(6);vcolumn.push_back(0);vport.push_back(0);

      vsrc_step.push_back(3);vsrc_column.push_back(1);vstep.push_back(4);vcolumn.push_back(0);vport.push_back(0);
      vsrc_step.push_back(3);vsrc_column.push_back(1);vstep.push_back(6);vcolumn.push_back(0);vport.push_back(1);
      vsrc_step.push_back(3);vsrc_column.push_back(1);vstep.push_back(6);vcolumn.push_back(1);vport.push_back(2);

      vsrc_step.push_back(3);vsrc_column.push_back(2);vstep.push_back(4);vcolumn.push_back(0);vport.push_back(0);
      vsrc_step.push_back(3);vsrc_column.push_back(2);vstep.push_back(4);vcolumn.push_back(1);vport.push_back(1);
      vsrc_step.push_back(3);vsrc_column.push_back(2);vstep.push_back(6);vcolumn.push_back(0);vport.push_back(2);
      vsrc_step.push_back(3);vsrc_column.push_back(2);vstep.push_back(6);vcolumn.push_back(1);vport.push_back(3);
      vsrc_step.push_back(3);vsrc_column.push_back(2);vstep.push_back(6);vcolumn.push_back(2);vport.push_back(4);

      for (i=3;i<8;i++) {
        vsrc_step.push_back(3);vsrc_column.push_back(i);vstep.push_back(4);vcolumn.push_back(i-3);vport.push_back(0);
        vsrc_step.push_back(3);vsrc_column.push_back(i);vstep.push_back(4);vcolumn.push_back(i-2);vport.push_back(1);
        vsrc_step.push_back(3);vsrc_column.push_back(i);vstep.push_back(4);vcolumn.push_back(i-1);vport.push_back(2);

        vsrc_step.push_back(3);vsrc_column.push_back(i);vstep.push_back(6);vcolumn.push_back(0);vport.push_back(3);
        vsrc_step.push_back(3);vsrc_column.push_back(i);vstep.push_back(6);vcolumn.push_back(1);vport.push_back(4);
        vsrc_step.push_back(3);vsrc_column.push_back(i);vstep.push_back(6);vcolumn.push_back(2);vport.push_back(5);
      }

      vsrc_step.push_back(3);vsrc_column.push_back(8);vstep.push_back(4);vcolumn.push_back(5);vport.push_back(0);
      vsrc_step.push_back(3);vsrc_column.push_back(8);vstep.push_back(4);vcolumn.push_back(6);vport.push_back(1);
      vsrc_step.push_back(3);vsrc_column.push_back(8);vstep.push_back(6);vcolumn.push_back(0);vport.push_back(2);
      vsrc_step.push_back(3);vsrc_column.push_back(8);vstep.push_back(6);vcolumn.push_back(1);vport.push_back(3);
      vsrc_step.push_back(3);vsrc_column.push_back(8);vstep.push_back(6);vcolumn.push_back(2);vport.push_back(4);

      vsrc_step.push_back(3);vsrc_column.push_back(9);vstep.push_back(4);vcolumn.push_back(6);vport.push_back(0);
      vsrc_step.push_back(3);vsrc_column.push_back(9);vstep.push_back(6);vcolumn.push_back(0);vport.push_back(1);
      vsrc_step.push_back(3);vsrc_column.push_back(9);vstep.push_back(6);vcolumn.push_back(1);vport.push_back(2);

      vsrc_step.push_back(3);vsrc_column.push_back(0);vstep.push_back(6);vcolumn.push_back(2);vport.push_back(0);

      // connect outputs for the fourth row (the fourth row only outputs to the fifth row)
      for (i=0;i<6;i++) {
        counter = 0;  
        for (j=i-2;j<i+1;j++) {
          if ( j >= 0 && j <= 4 ) {
            vsrc_step.push_back(4);vsrc_column.push_back(i);vstep.push_back(5);vcolumn.push_back(j);vport.push_back(counter);
            counter++;
          }
        }
      }

      // connect outputs for the fifth row (the fifth row only outputs to the sixth row)
      for (i=0;i<5;i++) {
        counter = 0;  
        for (j=i-2;j<i+1;j++) {
          if ( j >= 0 && j <= 2 ) {
            vsrc_step.push_back(5);vsrc_column.push_back(i);vstep.push_back(6);vcolumn.push_back(j);vport.push_back(counter);
            counter++;
          }
        }
      }

      // connect outputs for the sixth row (the sixth row only outputs to the first row)
      for (i=0;i<6;i++) {
                     vsrc_step.push_back(6);vsrc_column.push_back(0);vstep.push_back(1);vcolumn.push_back(i);vport.push_back(i);
        if ( i < 5 ) vsrc_step.push_back(6);vsrc_column.push_back(1);vstep.push_back(1);vcolumn.push_back(i+6);vport.push_back(i);
                     vsrc_step.push_back(6);vsrc_column.push_back(2);vstep.push_back(1);vcolumn.push_back(i+11);vport.push_back(i);
      }

      // Create a ragged 3D array
      for (j=0;j<vsrc_step.size();j++) {
        int column,step,src_column,src_step,port;
        src_column = vsrc_column[j]; src_step = vsrc_step[j];
        column = vcolumn[j]; step = vstep[j];
        port = vport[j];
        dst_port( step,column,dst_size(step,column,0) ) = port;
        dst_src(  step,column,dst_size(step,column,0) ) = src_column;
        dst_step( step,column,dst_size(step,column,0) ) = src_step;
        dst_size(step,column,0) += 1;
      }
    }


}}}}
