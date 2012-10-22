//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2009-2011 Matt Anderson
//  Copyright (c)      2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/include/serialization.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/util.hpp>

#include <boost/serialization/shared_ptr.hpp>
#include <boost/assign/std/vector.hpp>

#include "../dynamic_stencil_value.hpp"
#include "../functional_component.hpp"
#include "../../parameter.hpp"
#include "../../array3d.hpp"

#include "dataflow_stencil.hpp"

///////////////////////////////////////////////////////////////////////////////
typedef hpx::components::adaptive1d::server::dataflow_stencil dataflow_dataflow_stencil_type;

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the actions
HPX_REGISTER_ACTION(dataflow_dataflow_stencil_type::init_execute_action,
    adaptive1d_dataflow_dataflow_stencil_init_execute_action);
HPX_REGISTER_ACTION(dataflow_dataflow_stencil_type::execute_action,
    adaptive1d_dataflow_dataflow_stencil_execute_action);

typedef hpx::lcos::base_lco_with_value<
    boost::shared_ptr<std::vector<hpx::naming::id_type> > > dataflow_lco_gid_vector_ptr;

HPX_REGISTER_ACTION(dataflow_lco_gid_vector_ptr::set_value_action,
    adaptive1d_dataflow_set_value_action_gid_vector_ptr);
HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(dataflow_lco_gid_vector_ptr,
                                hpx::components::component_base_lco_with_value);

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::simple_component<dataflow_dataflow_stencil_type>,
    adaptive1d_dataflow_dataflow_stencil3d);

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace adaptive1d { namespace server
{
    dataflow_stencil::dataflow_stencil()
    {}

    inline bool dataflow_stencil::floatcmp(double_type const& x1, double_type const& x2) {
      // compare two floating point numbers
      static double_type const epsilon = 1.e-8;
      if ( x1 + epsilon >= x2 && x1 - epsilon <= x2 ) {
        // the numbers are close enough for coordinate comparison
        return true;
      } else {
        return false;
      }
    }

    inline bool dataflow_stencil::floatcmp_ge(double_type const& x1, double_type const& x2) {
      // compare two floating point numbers
      static double_type const epsilon = 1.e-8;

      if ( x1 > x2 ) return true;

      if ( x1 + epsilon >= x2 && x1 - epsilon <= x2 ) {
        // the numbers are close enough for coordinate comparison
        return true;
      } else {
        return false;
      }
    }

    inline bool dataflow_stencil::floatcmp_le(double_type const& x1, double_type const& x2) {
      // compare two floating point numbers
      static double_type const epsilon = 1.e-8;

      if ( x1 < x2 ) return true;

      if ( x1 + epsilon >= x2 && x1 - epsilon <= x2 ) {
        // the numbers are close enough for coordinate comparison
        return true;
      } else {
        return false;
      }
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Initialize functional components by setting the logging component to use
    void dataflow_stencil::init(distributed_iterator_range_type const& functions,
        distributed_iterator_range_type const& logging, std::size_t numsteps)
    {
        components::distributing_factory::iterator_type function = functions.first;
        naming::id_type log = naming::invalid_id;

        if (logging.first != logging.second)
            log = *logging.first;

        typedef std::vector<lcos::future<void> > lazyvals_type;
        lazyvals_type lazyvals;

        for (/**/; function != functions.second; ++function)
        {
            lazyvals.push_back(components::adaptive1d::stubs::functional_component::
                init_async(*function, numsteps, log));
        }

        lcos::wait(lazyvals);   // now wait for the initialization to happen
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Create functional components, one for each data point, use those to
    // initialize the stencil value instances
    void dataflow_stencil::init_stencils(distributed_iterator_range_type const& stencils,
        distributed_iterator_range_type const& functions, int static_step,
        array3d &dst_port,array3d &dst_src,array3d &dst_step,
        array3d &dst_size,array3d &src_size, double cycle_time, parameter const& par)
    {
        components::distributing_factory::iterator_type stencil = stencils.first;
        components::distributing_factory::iterator_type function = functions.first;

        // start an asynchronous operation for each of the stencil value instances
        typedef std::vector<lcos::future<void> > lazyvals_type;
        lazyvals_type lazyvals;

        for (int column = 0; stencil != stencils.second; ++stencil, ++function, ++column)
        {
            namespace stubs = components::adaptive1d::stubs;
            BOOST_ASSERT(function != functions.second);

#if 0       // DEBUG
            std::cout << " row " << static_step << " column " << column << " in " << dst_size(static_step,column,0) << " out " << src_size(static_step,column,0) << std::endl;
#endif
#if 0
            if ( dst_size(static_step,column,0) > 0 ) {
              std::cout << "                      in row:  " << dst_step(static_step,column,0) << " in column " << dst_src(static_step,column,0) << std::endl;
            }
            if ( dst_size(static_step,column,0) > 1 ) {
              std::cout << "                      in row:  " << dst_step(static_step,column,1) << " in column " << dst_src(static_step,column,1) << std::endl;
            }
            if ( dst_size(static_step,column,0) > 2 ) {
              std::cout << "                      in row:  " << dst_step(static_step,column,2) << " in column " << dst_src(static_step,column,2) << std::endl;
            }
            if ( dst_size(static_step,column,0) > 3 ) {
              std::cout << "                      in row:  " << dst_step(static_step,column,3) << " in column " << dst_src(static_step,column,3) << std::endl;
            }
#endif
            lazyvals.push_back(
                stubs::dynamic_stencil_value::set_functional_component_async(
                    *stencil, *function, static_step, column,
                    dst_size(static_step, column, 0),
                    src_size(static_step, column, 0), cycle_time,par));
        }
        //BOOST_ASSERT(function == functions.second);

        lcos::wait(lazyvals);   // now wait for the results
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Get gids of output ports of all functions
    void dataflow_stencil::get_output_ports(
        distributed_iterator_range_type const& stencils,
        std::vector<std::vector<naming::id_type> >& outputs)
    {
        typedef components::distributing_factory::result_type result_type;
        typedef
            std::vector<lcos::future<std::vector<naming::id_type> > >
        lazyvals_type;

        // start an asynchronous operation for each of the stencil value instances
        lazyvals_type lazyvals;
        components::distributing_factory::iterator_type stencil = stencils.first;
        for (/**/; stencil != stencils.second; ++stencil)
        {
            lazyvals.push_back(components::adaptive1d::stubs::dynamic_stencil_value::
                get_output_ports_async(*stencil));
        }

        lcos::wait(lazyvals, outputs);      // now wait for the results
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Connect the given output ports with the correct input ports, creating the
    // required static data-flow structure.
    //
    // Currently we have exactly one stencil_value instance per data point, where
    // the output ports of a stencil_value are connected to the input ports of the
    // direct neighbors of itself.
    void dataflow_stencil::connect_input_ports(
        components::distributing_factory::result_type const* stencils,
        std::vector<std::vector<std::vector<naming::id_type> > > const& outputs,
        array3d &dst_size,array3d &dst_step,array3d &dst_src,array3d &dst_port,
        parameter const& par)
    {
        typedef components::distributing_factory::result_type result_type;
        typedef std::vector<lcos::future<void> > lazyvals_type;

        lazyvals_type lazyvals;

        int steps = (int)outputs.size();
        for (int step = 0; step < steps; ++step)
        {
            components::distributing_factory::iterator_range_type r =
                util::locality_results(stencils[step]);

            components::distributing_factory::iterator_type stencil = r.first;
            for (int i = 0; stencil != r.second; ++stencil, ++i)
            {
                std::vector<naming::id_type> output_ports;

                for (int j = 0; j < dst_size(step, i, 0); ++j) {
                    output_ports.push_back(
                        outputs[dst_step(step,i,j)][dst_src(step,i,j)][dst_port(step,i,j)]);
                }

                lazyvals.push_back(components::adaptive1d::stubs::dynamic_stencil_value::
                    connect_input_ports_async(*stencil, output_ports));
            }
        }

        lcos::wait (lazyvals);      // now wait for the results
    }

    ///////////////////////////////////////////////////////////////////////////////
    void dataflow_stencil::prepare_initial_data(
        distributed_iterator_range_type const& functions,
        std::vector<naming::id_type> const& interp_src_data,
        std::vector<naming::id_type>& initial_data,
        double time,
        std::size_t numvalues,
        parameter const& par)
    {
        typedef std::vector<lcos::future<naming::id_type> > lazyvals_type;

        // create a data item value type for each of the functions
        lazyvals_type lazyvals;
        components::distributing_factory::iterator_type function = functions.first;

        for (std::size_t i = 0; function != functions.second; ++function, ++i)
        {
            lazyvals.push_back(components::adaptive1d::stubs::functional_component::
                alloc_data_async(*function, i, numvalues, 0,interp_src_data,time,par));
        }

        lcos::wait (lazyvals, initial_data);      // now wait for the results
    }

    ///////////////////////////////////////////////////////////////////////////////
    // do actual work
    void dataflow_stencil::execute(
        components::distributing_factory::iterator_range_type const& stencils,
        std::vector<naming::id_type> const& initial_data,
        std::vector<naming::id_type>& result_data)
    {
        // start the execution of all stencil stencils (data items)
        typedef std::vector<lcos::future<naming::id_type> > lazyvals_type;

        lazyvals_type lazyvals;
        components::distributing_factory::iterator_type stencil = stencils.first;
        for (std::size_t i = 0; stencil != stencils.second; ++stencil, ++i)
        {
            BOOST_ASSERT(i < initial_data.size());
            lazyvals.push_back(components::adaptive1d::stubs::dynamic_stencil_value::
                call_async(*stencil, initial_data[i]));
          //  lazyvals.push_back(components::adaptive1d::stubs::stencil_value<3>::
          //      call_async(*stencil, initial_data[i]));
        }

        lcos::wait (lazyvals, result_data);      // now wait for the results
    }

    ///////////////////////////////////////////////////////////////////////////////
    //
    void dataflow_stencil::start_row(
        components::distributing_factory::iterator_range_type const& stencils)
    {
        // start the execution of all stencil stencils (data items)
        typedef std::vector<lcos::future<void> > lazyvals_type;

        lazyvals_type lazyvals;
        components::distributing_factory::iterator_type stencil = stencils.first;
        for (std::size_t i = 0; stencil != stencils.second; ++stencil, ++i)
        {
            lazyvals.push_back(components::adaptive1d::stubs::dynamic_stencil_value::
                start_async(*stencil));
        }

        lcos::wait (lazyvals);      // now wait for the results
    }

    ///////////////////////////////////////////////////////////////////////////
    /// This is the main entry point of this component.
    boost::shared_ptr<std::vector<naming::id_type> > dataflow_stencil::init_execute(
        std::vector<naming::id_type> const& interp_src_data,
        double time,
        components::component_type function_type, std::size_t numvalues,
        std::size_t numsteps,
        components::component_type logging_type,
        parameter const& par)
    {
        //hpx::util::high_resolution_timer t;
        boost::shared_ptr<std::vector<naming::id_type> > result_data
            (new std::vector<naming::id_type>());

        components::component_type stencil_type =
            components::get_component_type<components::adaptive1d::server::dynamic_stencil_value>();

        typedef components::distributing_factory::result_type result_type;

        // create a distributing factory locally
        components::distributing_factory factory;
        factory.create(applier::get_applier().get_runtime_support_gid());

        // create a couple of stencil (functional) components and twice the
        // amount of stencil_value components
        result_type functions = factory.create_components(function_type, numvalues);

        int num_rows = par->num_rows;
        std::vector<std::size_t> each_row;
        each_row.resize(num_rows);
        for (int i=0;i<num_rows;i++) {
          each_row[i] = par->rowsize[par->level_row[i]];
        }

        std::vector<result_type> stencils;
        for (int i=0;i<num_rows;i++) {
          stencils.push_back(factory.create_components(stencil_type, each_row[i]));
        }

        // initialize logging functionality in functions
        result_type logging;
        if (logging_type != components::component_invalid)
            logging = factory.create_components(logging_type);

        init(util::locality_results(functions), util::locality_results(logging), numsteps);

        // prep the connections
        std::size_t memsize = 3;
        array3d dst_port(num_rows,each_row[0],memsize);
        array3d dst_src(num_rows,each_row[0],memsize);
        array3d dst_step(num_rows,each_row[0],memsize);
        array3d dst_size(num_rows,each_row[0],1);
        array3d src_size(num_rows,each_row[0],1);
        prep_ports(dst_port,dst_src,dst_step,dst_size,src_size,
                   num_rows,each_row,par);

        // initialize stencil_values using the stencil (functional) components
        for (int i = 0; i < num_rows; ++i)
            init_stencils(util::locality_results(stencils[i]), util::locality_results(functions), i,
                          dst_port,dst_src,dst_step,dst_size,src_size,time,par);

        // ask stencil instances for their output gids
        std::vector<std::vector<std::vector<naming::id_type> > > outputs(num_rows);
        for (int i = 0; i < num_rows; ++i)
            get_output_ports(util::locality_results(stencils[i]), outputs[i]);

        // connect output gids with corresponding stencil inputs
        connect_input_ports(&*stencils.begin(), outputs,dst_size,dst_step,dst_src,dst_port,par);

        // for loop over second row ; call start for each
        for (int i = 1; i < num_rows; ++i)
            start_row(util::locality_results(stencils[i]));

        // prepare initial data
        std::vector<naming::id_type> initial_data;
        prepare_initial_data(util::locality_results(functions), interp_src_data,
                             initial_data,time,
                             each_row[0],par);

        //std::cout << " Startup grid cost " << t.elapsed() << std::endl;
        // do actual work
        execute(util::locality_results(stencils[0]), initial_data, *result_data);

        // free all allocated components (we can do that synchronously)
        if (!logging.empty())
            factory.free_components_sync(logging);

        for (int i = 0; i < num_rows; ++i)
            factory.free_components_sync(stencils[i]);
        factory.free_components_sync(functions);

        return result_data;
    }

    ///////////////////////////////////////////////////////////////////////////
    /// This the other entry point of this component.
    std::vector<naming::id_type> dataflow_stencil::execute(
        std::vector<naming::id_type> const& initial_data,
        components::component_type function_type, std::size_t numvalues,
        std::size_t numsteps,
        components::component_type logging_type, parameter const& par)
    {
        BOOST_ASSERT(false);
        // This routine is deprecated currently

        std::vector<naming::id_type> result_data;

        return result_data;

    }

    void dataflow_stencil::prep_ports(array3d &dst_port,array3d &dst_src,
                                  array3d &dst_step,array3d &dst_size,
                                  array3d &src_size,std::size_t num_rows,
                                  std::vector<std::size_t> &each_row,
                                  parameter const& par)
    {
      std::size_t j;
      std::size_t /*counter,*/dst;

      // NOTE: this was used uninitialized in the following code
      bool prolongation = false;

      // vcolumn is the destination column number
      // vstep is the destination step (or row) number
      // vsrc_column is the source column number
      // vsrc_step is the source step number
      // vport is the output port number; increases consecutively
      std::vector<int> vcolumn,vstep,vsrc_column,vsrc_step,vport;

      //using namespace boost::assign;
      for (std::size_t step=0;step<num_rows;step = step + 1) {
        for (std::size_t i=0;i<each_row[step];i++) {
          //counter = 0;

          // find the level of this point
          int level = par->level_row[step];
          bool found = false;
          for (int ii=par->allowedl;ii>=0;ii--) {
            if ( i < par->rowsize[ii]) {
              level = ii;
              found = true;
              break;
            }
          }
          if ( !found ) {
            HPX_THROW_IN_CURRENT_FUNC(bad_parameter, "adaptive1d: Problem in prep_ports");
          }
          dst = step;

          // Special case for allowedl==0
          if ( par->allowedl == 0 ) {
            dst += 1;
            if ( dst >= num_rows ) dst = 0;

            if ( i == 0 ) {
              vsrc_step.push_back(step);vsrc_column.push_back(each_row[step]-1);vstep.push_back(dst);vcolumn.push_back(i);vport.push_back(0);
              vsrc_step.push_back(step);vsrc_column.push_back(i);vstep.push_back(dst);vcolumn.push_back(i);vport.push_back(1);
              vsrc_step.push_back(step);vsrc_column.push_back(i+1);vstep.push_back(dst);vcolumn.push_back(i);vport.push_back(2);
            } else if ( i == each_row[step]-1 ) {
              vsrc_step.push_back(step);vsrc_column.push_back(i-1);vstep.push_back(dst);vcolumn.push_back(i);vport.push_back(0);
              vsrc_step.push_back(step);vsrc_column.push_back(i);vstep.push_back(dst);vcolumn.push_back(i);vport.push_back(1);
              vsrc_step.push_back(step);vsrc_column.push_back(0);vstep.push_back(dst);vcolumn.push_back(i);vport.push_back(2);
            } else {
              vsrc_step.push_back(step);vsrc_column.push_back(i-1);vstep.push_back(dst);vcolumn.push_back(i);vport.push_back(0);
              vsrc_step.push_back(step);vsrc_column.push_back(i);vstep.push_back(dst);vcolumn.push_back(i);vport.push_back(1);
              vsrc_step.push_back(step);vsrc_column.push_back(i+1);vstep.push_back(dst);vcolumn.push_back(i);vport.push_back(2);
            }
          } else {
            if ( (step+3)%3 != 0 ) {
               // anytime there is a difference of more than one level between src and dst rows,
              // you need to account for the prolongation/restriction rows going on inbetween them.
              // That is given by 2^{L-l-1}-1
              int intermediate = (int) pow(2.,int(par->allowedl-level)) ;
              if ( par->allowedl-level > 1 ) {
                dst += intermediate + intermediate/2 - 1;
              } else {
                dst += intermediate;
              }
              prolongation = false;
            } else {
              dst += 1; // this is a prolongation/restriction step
              prolongation = true;
            }
          }
          if ( dst >= num_rows ) dst = 0;

          if ( par->allowedl > 0 ) {
            // AMR dataflow goes here
            if ( prolongation == false ) {
              // level zero -- periodic BC
              if ( level == 0 ) {
                int gi = par->levelp[level];
                if ( i == par->gi2item[ gi ] ) {
                  vsrc_step.push_back(step);vsrc_column.push_back(each_row[step]-1);vstep.push_back(dst);vcolumn.push_back(i);vport.push_back(0);
                  vsrc_step.push_back(step);vsrc_column.push_back(i);vstep.push_back(dst);vcolumn.push_back(i);vport.push_back(1);
                  vsrc_step.push_back(step);vsrc_column.push_back(i+1);vstep.push_back(dst);vcolumn.push_back(i);vport.push_back(2);
                } else if ( i == each_row[step]-1 ) {
                  vsrc_step.push_back(step);vsrc_column.push_back(i-1);vstep.push_back(dst);vcolumn.push_back(i);vport.push_back(0);
                  vsrc_step.push_back(step);vsrc_column.push_back(i);vstep.push_back(dst);vcolumn.push_back(i);vport.push_back(1);
                  vsrc_step.push_back(step);vsrc_column.push_back( par->gi2item[ gi ] );vstep.push_back(dst);vcolumn.push_back(i);vport.push_back(2);
                } else {
                  vsrc_step.push_back(step);vsrc_column.push_back(i-1);vstep.push_back(dst);vcolumn.push_back(i);vport.push_back(0);
                  vsrc_step.push_back(step);vsrc_column.push_back(i);vstep.push_back(dst);vcolumn.push_back(i);vport.push_back(1);
                  vsrc_step.push_back(step);vsrc_column.push_back(i+1);vstep.push_back(dst);vcolumn.push_back(i);vport.push_back(2);
                }
              } else {
                if ( par->gr_lbox[ par->item2gi[i] ] && par->gr_rbox[ par->item2gi[i] ] ) {
                  vsrc_step.push_back(step);vsrc_column.push_back(i);vstep.push_back(dst);vcolumn.push_back(i);vport.push_back(0);
                } else if ( par->gr_lbox[ par->item2gi[i] ] && !par->gr_rbox[ par->item2gi[i] ] ) {
                  vsrc_step.push_back(step);vsrc_column.push_back(i);vstep.push_back(dst);vcolumn.push_back(i);vport.push_back(0);
                  vsrc_step.push_back(step);vsrc_column.push_back(i);vstep.push_back(dst);vcolumn.push_back( par->gr_right_neighbor[ par->item2gi[i] ]);vport.push_back(1);
                } else if ( !par->gr_lbox[ par->item2gi[i] ] && par->gr_rbox[ par->item2gi[i] ] ) {
                  vsrc_step.push_back(step);vsrc_column.push_back(i);vstep.push_back(dst);vcolumn.push_back( par->gr_left_neighbor[ par->item2gi[i] ]);vport.push_back(0);
                  vsrc_step.push_back(step);vsrc_column.push_back(i);vstep.push_back(dst);vcolumn.push_back(i);vport.push_back(1);
                } else if ( !par->gr_lbox[ par->item2gi[i] ] && !par->gr_rbox[ par->item2gi[i] ] ) {
                  vsrc_step.push_back(step);vsrc_column.push_back(i);vstep.push_back(dst);vcolumn.push_back( par->gr_left_neighbor[ par->item2gi[i] ]);vport.push_back(0);
                  vsrc_step.push_back(step);vsrc_column.push_back(i);vstep.push_back(dst);vcolumn.push_back(i);vport.push_back(1);
                  vsrc_step.push_back(step);vsrc_column.push_back(i);vstep.push_back(dst);vcolumn.push_back( par->gr_right_neighbor[ par->item2gi[i] ]);vport.push_back(2);
                }
              }
            } else {
              // prolongation true case
              j = i;
              vsrc_step.push_back(step);vsrc_column.push_back(i);vstep.push_back(dst);vcolumn.push_back(j);vport.push_back(0);
#if 0
              if ( par->gr_lbox[ par->item2gi[i] ] && par->gr_rbox[ par->item2gi[i] ] ) {
                vsrc_step.push_back(step);vsrc_column.push_back(i);vstep.push_back(dst);vcolumn.push_back( par->gr_left_neighbor[ par->item2gi[i] ]  );vport.push_back(1);
                vsrc_step.push_back(step);vsrc_column.push_back(i);vstep.push_back(dst);vcolumn.push_back( par->gr_right_neighbor[ par->item2gi[i] ]  );vport.push_back(2);
              } else if ( par->gr_lbox[ par->item2gi[i] ] && !par->gr_rbox[ par->item2gi[i] ] ) {
                vsrc_step.push_back(step);vsrc_column.push_back(i);vstep.push_back(dst);vcolumn.push_back( par->gr_left_neighbor[ par->item2gi[i] ]  );vport.push_back(1);
              } else if ( !par->gr_lbox[ par->item2gi[i] ] && par->gr_rbox[ par->item2gi[i] ] ) {
                vsrc_step.push_back(step);vsrc_column.push_back(i);vstep.push_back(dst);vcolumn.push_back( par->gr_right_neighbor[ par->item2gi[i] ]  );vport.push_back(1);
              }
#endif

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
    }

    bool dataflow_stencil::intersection(double_type xmin,double_type xmax,
                                        double_type xmin2,double_type xmax2)
    {
      double_type pa[3],ea[3];
      static double_type const half = 0.5;
      pa[0] = half*(xmax + xmin);

      ea[0] = xmax - pa[0];

      double_type pb[3],eb[3];
      pb[0] = half*(xmax2 + xmin2);

      eb[0] = xmax2 - pb[0];

      double_type T[3];
      T[0] = pb[0] - pa[0];

      if ( floatcmp_le(fabs(T[0]),ea[0] + eb[0]) ) {
        return true;
      } else {
        return false;
      }

    }

}}}}
