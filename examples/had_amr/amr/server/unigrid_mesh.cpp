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

#include "unigrid_mesh.hpp"

///////////////////////////////////////////////////////////////////////////////
typedef hpx::components::amr::server::unigrid_mesh had_unigrid_mesh_type;

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the actions
HPX_REGISTER_ACTION_EX(had_unigrid_mesh_type::init_execute_action, had_unigrid_mesh_init_execute_action);
HPX_REGISTER_ACTION_EX(had_unigrid_mesh_type::execute_action, had_unigrid_mesh_execute_action);

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::simple_component<had_unigrid_mesh_type>, had_unigrid_mesh);
HPX_DEFINE_GET_COMPONENT_TYPE(had_unigrid_mesh_type);

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr { namespace server 
{
    unigrid_mesh::unigrid_mesh()
      : numvalues_(0)
    {}

    ///////////////////////////////////////////////////////////////////////////////
    // Initialize functional components by setting the logging component to use
    void unigrid_mesh::init(distributed_iterator_range_type const& functions,
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
    void unigrid_mesh::init_stencils(distributed_iterator_range_type const& stencils,
        distributed_iterator_range_type const& functions, int static_step, 
        int numvalues, Parameter const& par)
    {
        components::distributing_factory::iterator_type stencil = stencils.first;
        components::distributing_factory::iterator_type function = functions.first;

        int memsize;
        memsize = 6;
        Array3D dst_port(6,numvalues,memsize);
        Array3D dst_src(6,numvalues,memsize);
        Array3D dst_step(6,numvalues,memsize);
        Array3D dst_size(6,numvalues,1);
        Array3D src_size(6,numvalues,1);
        prep_ports(dst_port,dst_src,dst_step,dst_size,src_size,numvalues,par);

        // start an asynchronous operation for each of the stencil value instances
        typedef std::vector<lcos::future_value<void> > lazyvals_type;
        lazyvals_type lazyvals;

        for (int column = 0; stencil != stencils.second; ++stencil, ++function, ++column)
        {
            namespace stubs = components::amr::stubs;
            BOOST_ASSERT(function != functions.second);

            //std::cout << " row " << static_step << " column " << column << " in " << dst_size(static_step,column,0) << " out " << src_size(static_step,column,0) << std::endl;
            lazyvals.push_back(
                stubs::dynamic_stencil_value::set_functional_component_async(
                    *stencil, *function, static_step, column, 
                    dst_size(static_step, column, 0), 
                    src_size(static_step, column, 0), par));
        }
        BOOST_ASSERT(function == functions.second);

        wait(lazyvals);   // now wait for the results
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Get gids of output ports of all functions
    void unigrid_mesh::get_output_ports(
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

        wait(lazyvals, outputs);      // now wait for the results
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

    void unigrid_mesh::connect_input_ports(
        components::distributing_factory::result_type const* stencils,
        std::vector<std::vector<std::vector<naming::id_type> > > const& outputs,
        Parameter const& par)
    {
        typedef components::distributing_factory::result_type result_type;

        int j;
        std::size_t numvals = outputs[0].size();

        int memsize;
        memsize = 6;

        Array3D dst_port(6,numvals,memsize);
        Array3D dst_src(6,numvals,memsize);
        Array3D dst_step(6,numvals,memsize);
        Array3D dst_size(6,numvals,1);
        Array3D src_size(6,numvals,1);
        prep_ports(dst_port,dst_src,dst_step,dst_size,src_size,numvals,par);


        int steps = (int)outputs.size();
        for (int step = 0; step < steps; ++step) 
        {
            components::distributing_factory::iterator_range_type r = 
                locality_results(stencils[step]);
            components::distributing_factory::iterator_type stencil = r.first;
            for (int i = 0; stencil != r.second; ++stencil, ++i)
            {
                using namespace boost::assign;

                std::vector<naming::id_type> output_ports;

                for (j = 0; j < dst_size(step, i, 0); ++j) {
                    output_ports.push_back(
                        outputs[dst_step(step,i,j)][dst_src(step,i,j)][dst_port( step,i,j)]);
                }

                components::amr::stubs::dynamic_stencil_value::
                    connect_input_ports(*stencil, output_ports);
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////////
    void unigrid_mesh::prepare_initial_data(
        distributed_iterator_range_type const& functions, 
        std::vector<naming::id_type>& initial_data,
        std::size_t level, had_double_type xmin,
        Parameter const& par)
    {
        typedef std::vector<lcos::future_value<naming::id_type> > lazyvals_type;

        // create a data item value type for each of the functions
        lazyvals_type lazyvals;
        components::distributing_factory::iterator_type function = functions.first;

        for (std::size_t i = 0; function != functions.second; ++function, ++i)
        {
            lazyvals.push_back(components::amr::stubs::functional_component::
                alloc_data_async(*function, i, numvalues_, 0, level, xmin, par));
        }

        wait (lazyvals, initial_data);      // now wait for the results
    }

    ///////////////////////////////////////////////////////////////////////////////
    // do actual work
    void unigrid_mesh::execute(
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

        wait (lazyvals, result_data);      // now wait for the results
    }
    
    ///////////////////////////////////////////////////////////////////////////////
    // 
    void unigrid_mesh::start_row(
        components::distributing_factory::iterator_range_type const& stencils)
    {
        // start the execution of all stencil stencils (data items)
        typedef std::vector<lcos::future_value<void> > lazyvals_type;

        lazyvals_type lazyvals;
        components::distributing_factory::iterator_type stencil = stencils.first;
        for (std::size_t i = 0; stencil != stencils.second; ++stencil, ++i)
        {
            lazyvals.push_back(components::amr::stubs::dynamic_stencil_value::
                start_async(*stencil));
        }

        wait (lazyvals);      // now wait for the results
    }

    ///////////////////////////////////////////////////////////////////////////
    /// This is the main entry point of this component. 
    std::vector<naming::id_type> unigrid_mesh::init_execute(
        components::component_type function_type, std::size_t numvalues, 
        std::size_t numsteps,
        components::component_type logging_type,
        std::size_t level, had_double_type xmin,
        Parameter const& par)
    {
        std::vector<naming::id_type> result_data;

        components::component_type stencil_type = 
            components::get_component_type<components::amr::server::dynamic_stencil_value>();

        typedef components::distributing_factory::result_type result_type;

        // create a distributing factory locally
        components::distributing_factory factory;
        factory.create(applier::get_applier().get_runtime_support_gid());

        // create a couple of stencil (functional) components and twice the 
        // amount of stencil_value components
        numvalues_ = numvalues;
        result_type functions = factory.create_components(function_type, numvalues);
        result_type stencils[6] = 
        {
            factory.create_components(stencil_type, numvalues),
            factory.create_components(stencil_type, numvalues),
            factory.create_components(stencil_type, numvalues),
            factory.create_components(stencil_type, numvalues),
            factory.create_components(stencil_type, numvalues),
            factory.create_components(stencil_type, numvalues)
        };

        // initialize logging functionality in functions
        result_type logging;
        if (logging_type != components::component_invalid)
            logging = factory.create_components(logging_type);

        init(locality_results(functions), locality_results(logging), numsteps);

        int i;
        // initialize stencil_values using the stencil (functional) components
        for (i = 0; i < 6; ++i) 
            init_stencils(locality_results(stencils[i]), locality_results(functions), i, numvalues, par);

        // ask stencil instances for their output gids
        std::vector<std::vector<std::vector<naming::id_type> > > outputs(6);
        for (i = 0; i < 6; ++i) 
            get_output_ports(locality_results(stencils[i]), outputs[i]);

        // connect output gids with corresponding stencil inputs
        connect_input_ports(stencils, outputs,par);

        // for loop over second row ; call start for each
        for (i = 1; i < 6; ++i) 
            start_row(locality_results(stencils[i]));

        // prepare initial data
        std::vector<naming::id_type> initial_data;
        prepare_initial_data(locality_results(functions), initial_data, level, xmin, par);

        // do actual work
        execute(locality_results(stencils[0]), initial_data, result_data);

        // free all allocated components (we can do that synchronously)
        if (!logging.empty())
            factory.free_components_sync(logging);

        for (i = 5; i >= 0; --i) 
            factory.free_components_sync(stencils[i]);
        factory.free_components_sync(functions);

        return result_data;
    }

    ///////////////////////////////////////////////////////////////////////////
    /// This the other entry point of this component. 
    std::vector<naming::id_type> unigrid_mesh::execute(
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
        factory.create(applier::get_applier().get_runtime_support_gid());

        // create a couple of stencil (functional) components and twice the 
        // amount of stencil_value components
        numvalues_ = numvalues;
        result_type functions = factory.create_components(function_type, numvalues);
        result_type stencils[6] = 
        {
            factory.create_components(stencil_type, numvalues),
            factory.create_components(stencil_type, numvalues),
            factory.create_components(stencil_type, numvalues),
            factory.create_components(stencil_type, numvalues),
            factory.create_components(stencil_type, numvalues),
            factory.create_components(stencil_type, numvalues)
        };

        // initialize logging functionality in functions
        result_type logging;
        if (logging_type != components::component_invalid)
            logging = factory.create_components(logging_type);

        init(locality_results(functions), locality_results(logging), numsteps);

        int i;
        // initialize stencil_values using the stencil (functional) components
        for (i=0;i<6;i++) init_stencils(locality_results(stencils[i]), locality_results(functions), i, numvalues, par);

        // ask stencil instances for their output gids
        std::vector<std::vector<std::vector<naming::id_type> > > outputs(6);
        for (i=0;i<6;i++) get_output_ports(locality_results(stencils[i]), outputs[i]);

        // connect output gids with corresponding stencil inputs
        connect_input_ports(stencils, outputs,par);

        // for loop over second row ; call start for each
        for (i=1;i<6;i++) start_row(locality_results(stencils[i]));

        // do actual work
        execute(locality_results(stencils[0]), initial_data, result_data);

        // free all allocated components (we can do that synchronously)
        if (!logging.empty())
            factory.free_components_sync(logging);
        for (i=5;i>=0;i--) factory.free_components_sync(stencils[i]);
        factory.free_components_sync(functions);

        return result_data;
    }

    void unigrid_mesh::prep_ports(Array3D &dst_port,Array3D &dst_src,
                                  Array3D &dst_step,Array3D &dst_size,Array3D &src_size,int numvalues,Parameter const& par)
    {
      int i,j;
      
      // vcolumn is the destination column number
      // vstep is the destination step (or row) number
      // vsrc_column is the source column number
      // vsrc_step is the source step number
      // vport is the output port number; increases consecutively
      std::vector<int> vcolumn,vstep,vsrc_column,vsrc_step,vport;

      //using namespace boost::assign;

      int counter;
      int step,dst,dst2;

      for (step=0;step<6;step = step + 1) {
        dst = step+1;
        if ( dst == 6 ) dst = 0;

        for (i=0;i<numvalues;i++) {
          counter = 0;

          // three 
          for (j=i-1;j<i+2;j++) {
            if ( j >=0 && j < numvalues ) {
              vsrc_step.push_back(step);vsrc_column.push_back(i);vstep.push_back(dst);vcolumn.push_back(j);vport.push_back(counter);
              counter++;
            }
          }

          // extra output for step 0 and step 6
          if ( step == 0 || step == 3 ) {
            if (step == 0 ) dst2 = 3;
            else dst2 = 0;

            // three 
            for (j=i-1;j<i+2;j++) {
              if ( j >=0 && j < numvalues ) {
                vsrc_step.push_back(step);vsrc_column.push_back(i);vstep.push_back(dst2);vcolumn.push_back(j);vport.push_back(counter);
                counter++;
              }
            }

          }
        }
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
        src_size(src_step,src_column,0) += 1;
      }

      // sort the src step (or row) in descending order
      int t1,k,kk;
      int column;
      for (j=0;j<vsrc_step.size();j++) {
        step = vstep[j];
        column = vcolumn[j];

        for (kk=dst_size(step,column,0);kk>=0;kk--) {
          for (k=0;k<kk-1;k++) {
            if (dst_step( step,column,k) < dst_step( step,column,k+1) ) {
              // swap
              t1 = dst_step( step,column,k);
              dst_step( step,column,k) = dst_step( step,column,k+1);
              dst_step( step,column,k+1) = t1;
  
              // swap the src, port info too
              t1 = dst_src( step,column,k);
              dst_src( step,column,k) = dst_src( step,column,k+1);
              dst_src( step,column,k+1) = t1;
  
              t1 = dst_port( step,column,k);
              dst_port( step,column,k) = dst_port( step,column,k+1);
              dst_port( step,column,k+1) = t1;
            } else if ( dst_step( step,column,k) == dst_step( step,column,k+1) ) {
              //sort the src column in ascending order if the step is the same
              if (dst_src( step,column,k) > dst_src( step,column,k+1) ) {
                t1 = dst_src( step,column,k);
                dst_src( step,column,k) = dst_src( step,column,k+1);
                dst_src( step,column,k+1) = t1;

                // swap the src, port info too
                t1 = dst_port( step,column,k);
                dst_port( step,column,k) = dst_port( step,column,k+1);
                dst_port( step,column,k+1) = t1;
              }

            }
          }
        }
      }

    }

}}}}
