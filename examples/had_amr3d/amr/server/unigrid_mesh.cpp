//  Copyright (c) 2007-2011 Hartmut Kaiser
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
HPX_REGISTER_ACTION_EX(had_unigrid_mesh_type::init_execute_action, 
    had_unigrid_mesh_init_execute_action);
HPX_REGISTER_ACTION_EX(had_unigrid_mesh_type::execute_action, 
    had_unigrid_mesh_execute_action);

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::simple_component<had_unigrid_mesh_type>, had_unigrid_mesh3d);
HPX_DEFINE_GET_COMPONENT_TYPE(had_unigrid_mesh_type);

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr { namespace server 
{
    unigrid_mesh::unigrid_mesh()
    {}

    inline bool unigrid_mesh::floatcmp(had_double_type const& x1, had_double_type const& x2) {
      // compare two floating point numbers
      static had_double_type const epsilon = 1.e-8;
      if ( x1 + epsilon >= x2 && x1 - epsilon <= x2 ) {
        // the numbers are close enough for coordinate comparison
        return true;
      } else {
        return false;
      }
    }

    inline bool unigrid_mesh::floatcmp_ge(had_double_type const& x1, had_double_type const& x2) {
      // compare two floating point numbers
      static had_double_type const epsilon = 1.e-8;

      if ( x1 > x2 ) return true;

      if ( x1 + epsilon >= x2 && x1 - epsilon <= x2 ) {
        // the numbers are close enough for coordinate comparison
        return true;
      } else {
        return false;
      }
    }

    inline bool unigrid_mesh::floatcmp_le(had_double_type const& x1, had_double_type const& x2) {
      // compare two floating point numbers
      static had_double_type const epsilon = 1.e-8;

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
    void unigrid_mesh::init(distributed_iterator_range_type const& functions,
        distributed_iterator_range_type const& logging, std::size_t numsteps)
    {
        components::distributing_factory::iterator_type function = functions.first;
        naming::id_type log = naming::invalid_id;

        if (logging.first != logging.second)
            log = *logging.first;

        typedef std::vector<lcos::future_value<void> > lazyvals_type;
        lazyvals_type lazyvals;

        for (/**/; function != functions.second; ++function)
        {
            lazyvals.push_back(components::amr::stubs::functional_component::
                init_async(*function, numsteps, log));
        }

        wait(lazyvals);   // now wait for the initialization to happen
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Create functional components, one for each data point, use those to 
    // initialize the stencil value instances
    void unigrid_mesh::init_stencils(distributed_iterator_range_type const& stencils,
        distributed_iterator_range_type const& functions, int static_step, 
        Array3D &dst_port,Array3D &dst_src,Array3D &dst_step,
        Array3D &dst_size,Array3D &src_size, Parameter const& par)
    {
        components::distributing_factory::iterator_type stencil = stencils.first;
        components::distributing_factory::iterator_type function = functions.first;

        // start an asynchronous operation for each of the stencil value instances
        typedef std::vector<lcos::future_value<void> > lazyvals_type;
        lazyvals_type lazyvals;

        for (int column = 0; stencil != stencils.second; ++stencil, ++function, ++column)
        {
            namespace stubs = components::amr::stubs;
            BOOST_ASSERT(function != functions.second);

#if 0       // DEBUG
            std::cout << " row " << static_step << " column " << column << " in " << dst_size(static_step,column,0) << " out " << src_size(static_step,column,0) << std::endl;

            // Follow up TEST
            if ( dst_size(static_step,column,0) == 0 ) {
              // figure out the index and level of this point
              std::size_t a,b,c;
              int level = findlevel3D(static_step,column,a,b,c,par);
              std::cout << "                 PROBLEM: level " << level << " index " << a << " " << b << " " << c << std::endl;
            } 
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
                    src_size(static_step, column, 0), par));
        }
        //BOOST_ASSERT(function == functions.second);

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
    void unigrid_mesh::connect_input_ports(
        components::distributing_factory::result_type const* stencils,
        std::vector<std::vector<std::vector<naming::id_type> > > const& outputs,
        Array3D &dst_size,Array3D &dst_step,Array3D &dst_src,Array3D &dst_port,
        Parameter const& par)
    {
        typedef components::distributing_factory::result_type result_type;
        typedef std::vector<lcos::future_value<void> > lazyvals_type;

        lazyvals_type lazyvals;

        int steps = (int)outputs.size();
        for (int step = 0; step < steps; ++step) 
        {
            components::distributing_factory::iterator_range_type r = 
                locality_results(stencils[step]);

            components::distributing_factory::iterator_type stencil = r.first;
            for (int i = 0; stencil != r.second; ++stencil, ++i)
            {
                std::vector<naming::id_type> output_ports;

                for (int j = 0; j < dst_size(step, i, 0); ++j) {
                    output_ports.push_back(
                        outputs[dst_step(step,i,j)][dst_src(step,i,j)][dst_port(step,i,j)]);
                }

                lazyvals.push_back(components::amr::stubs::dynamic_stencil_value::
                    connect_input_ports_async(*stencil, output_ports));
            }
        }

        wait (lazyvals);      // now wait for the results
    }

    ///////////////////////////////////////////////////////////////////////////////
    void unigrid_mesh::prepare_initial_data(
        distributed_iterator_range_type const& functions, 
        std::vector<naming::id_type>& initial_data,
        std::size_t numvalues,
        Parameter const& par)
    {
        typedef std::vector<lcos::future_value<naming::id_type> > lazyvals_type;

        // create a data item value type for each of the functions
        lazyvals_type lazyvals;
        components::distributing_factory::iterator_type function = functions.first;

        for (std::size_t i = 0; function != functions.second; ++function, ++i)
        {
            lazyvals.push_back(components::amr::stubs::functional_component::
                alloc_data_async(*function, i, numvalues, 0,par));
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
        Parameter const& par)
    {
        //hpx::util::high_resolution_timer t;
        std::vector<naming::id_type> result_data;

        components::component_type stencil_type = 
            components::get_component_type<components::amr::server::dynamic_stencil_value>();

        typedef components::distributing_factory::result_type result_type;

        // create a distributing factory locally
        components::distributing_factory factory;
        factory.create(applier::get_applier().get_runtime_support_gid());

        // create a couple of stencil (functional) components and twice the 
        // amount of stencil_value components
        result_type functions = factory.create_components(function_type, numvalues);

        int num_rows = par->level_row.size();

        // Each row potentially has a different number of points depending on the
        // number of levels of refinement.  There are 2^(nlevel) rows each timestep;
        // this mesh is set up to take two timesteps at a time (i.e. nt0 must be even).
        // The rowsize vector computes how many points are in a row which includes
        // a specified number of levels.  For example, rowsize[0] indicates the total
        // number of points in a row that includes all levels down to the coarsest, level=0; 
        // similarly, rowsize[1] indicates the total number of points in a row that includes
        // all finer levels down to level=1.

        // vectors level_begin and level_end indicate the level to which a particular index belongs

        // The vector each_row connects the rowsize vector with each of the 2^nlevel rows.
        // The pattern is as follows: 
        //     For 0 levels of refinement--
        //       there are 2 rows:    
        //                            1  rowsize[0]
        //                            0  rowsize[0]
        //  
        //     For 1 level of refinement--
        //       there are 4 rows:  
        //                            3  rowsize[1]
        //                            2  rowsize[0]
        //                            1  rowsize[1]
        //                            0  rowsize[0]
        // 
        //     For 2 levels of refinement--
        //       there are 8 rows: 
        //                             7 rowsize[2]
        //                             6 rowsize[1]
        //                             5 rowsize[2]
        //                             4 rowsize[0]
        //                             3 rowsize[2]
        //                             2 rowsize[1]
        //                             1 rowsize[2]
        //                             0 rowsize[0]
        //
        //  This structure is designed so that every 2 rows the timestep is the same for every point in that row

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

        init(locality_results(functions), locality_results(logging), numsteps);

        // prep the connections
        std::size_t memsize = 35;
        Array3D dst_port(num_rows,each_row[0],memsize);
        Array3D dst_src(num_rows,each_row[0],memsize);
        Array3D dst_step(num_rows,each_row[0],memsize);
        Array3D dst_size(num_rows,each_row[0],1);
        Array3D src_size(num_rows,each_row[0],1);
        prep_ports(dst_port,dst_src,dst_step,dst_size,src_size,
                   num_rows,each_row,par);

        // initialize stencil_values using the stencil (functional) components
        for (int i = 0; i < num_rows; ++i) 
            init_stencils(locality_results(stencils[i]), locality_results(functions), i,
                          dst_port,dst_src,dst_step,dst_size,src_size, par);

        // ask stencil instances for their output gids
        std::vector<std::vector<std::vector<naming::id_type> > > outputs(num_rows);
        for (int i = 0; i < num_rows; ++i) 
            get_output_ports(locality_results(stencils[i]), outputs[i]);

        // connect output gids with corresponding stencil inputs
        connect_input_ports(&*stencils.begin(), outputs,dst_size,dst_step,dst_src,dst_port,par);

        // for loop over second row ; call start for each
        for (int i = 1; i < num_rows; ++i) 
            start_row(locality_results(stencils[i]));

        // prepare initial data
        std::vector<naming::id_type> initial_data;
        prepare_initial_data(locality_results(functions), initial_data, 
                             each_row[0],par);

        //std::cout << " Startup grid cost " << t.elapsed() << std::endl;
        // do actual work
        execute(locality_results(stencils[0]), initial_data, result_data);

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
    std::vector<naming::id_type> unigrid_mesh::execute(
        std::vector<naming::id_type> const& initial_data,
        components::component_type function_type, std::size_t numvalues, 
        std::size_t numsteps,
        components::component_type logging_type, Parameter const& par)
    {
        BOOST_ASSERT(false);
        // This routine is deprecated currently

        std::vector<naming::id_type> result_data;
#if 0
        components::component_type stencil_type = 
            components::get_component_type<components::amr::server::dynamic_stencil_value >();

        typedef components::distributing_factory::result_type result_type;

        // create a distributing factory locally
        components::distributing_factory factory;
        factory.create(applier::get_applier().get_runtime_support_gid());

        // create a couple of stencil (functional) components and twice the 
        // amount of stencil_value components
        result_type functions = factory.create_components(function_type, numvalues);

        std::vector<result_type> stencils;

        had_double_type tmp = 3*pow(2,par->allowedl);
        int num_rows = (int) tmp; 

        std::vector<std::size_t> rowsize;
        for (int j=0;j<=par->allowedl;j++) {
          rowsize.push_back(par->nx[par->allowedl]);
          for (int i=par->allowedl-1;i>=j;i--) {
            // remove duplicates
            rowsize[j] += par->nx[i] - (par->nx[i+1]+1)/2;
          }
        }

        for (int i=0;i<num_rows;i++) stencils.push_back(factory.create_components(stencil_type, numvalues));

        // initialize logging functionality in functions
        result_type logging;
        if (logging_type != components::component_invalid)
            logging = factory.create_components(logging_type);

        init(locality_results(functions), locality_results(logging), numsteps);

        // initialize stencil_values using the stencil (functional) components
        for (int i = 0; i < num_rows; ++i) 
            init_stencils(locality_results(stencils[i]), locality_results(functions), i, numvalues, par);

        // ask stencil instances for their output gids
        std::vector<std::vector<std::vector<naming::id_type> > > outputs(num_rows);
        for (int i = 0; i < num_rows; ++i) 
            get_output_ports(locality_results(stencils[i]), outputs[i]);

        // connect output gids with corresponding stencil inputs
        connect_input_ports(&*stencils.begin(), outputs,numvalues,par);

        // for loop over second row ; call start for each
        for (int i = 1; i < num_rows; ++i) 
            start_row(locality_results(stencils[i]));

        // do actual work
        execute(locality_results(stencils[0]), initial_data, result_data);

        // free all allocated components (we can do that synchronously)
        if (!logging.empty())
            factory.free_components_sync(logging);

        for (int i = 0; i < num_rows; ++i) 
            factory.free_components_sync(stencils[i]);
        factory.free_components_sync(functions);
#endif
        return result_data;

    }

    void unigrid_mesh::prep_ports(Array3D &dst_port,Array3D &dst_src,
                                  Array3D &dst_step,Array3D &dst_size,Array3D &src_size,std::size_t num_rows,
                                  std::vector<std::size_t> &each_row,
                                  Parameter const& par)
    {
      int i,j,k;
      
      // vcolumn is the destination column number
      // vstep is the destination step (or row) number
      // vsrc_column is the source column number
      // vsrc_step is the source step number
      // vport is the output port number; increases consecutively
      std::vector<int> vcolumn,vstep,vsrc_column,vsrc_step,vport;

      //using namespace boost::assign;

      int counter;
      int step,dst;
//      int found;

      std::size_t a,b,c;
      for (step=0;step<num_rows;step = step + 1) {
        for (i=0;i<each_row[step];i++) {
          counter = 0;

          // get 3D coordinates from 'i' value
          // i.e. i = a + nx*(b+c*nx);
          int level = findlevel3D(step,i,a,b,c,par);

          // communicate 27
          int kstart = c-1;
          if ( kstart < 0 ) kstart = 0;
          int kend = c+2;
          if ( kend > par->nx[level] ) kend = par->nx[level];

          int jstart = b-1;
          if ( jstart < 0 ) jstart = 0;
          int jend = b+2;
          if ( jend > par->nx[level] ) jend = par->nx[level];

          int istart = a-1;
          if ( istart < 0 ) istart = 0;
          int iend = a+2;
          if ( iend > par->nx[level] ) iend = par->nx[level];

          dst = par->ndst[level+(par->allowedl+1)*step];
          bool prolongation;
          if ( par->prores[step] == 100 ) {
            prolongation = false;
          } else {
            prolongation = true;
          }

          if ( prolongation == false ) {
            for (int kk=kstart;kk<kend;kk++) {
            for (int jj=jstart;jj<jend;jj++) {
            for (int ii=istart;ii<iend;ii++) {
              j = ii + par->nx[level]*(jj+kk*par->nx[level]);

              if ( level != par->allowedl ) {
                j += par->rowsize[level+1];
              }
              vsrc_step.push_back(step);vsrc_column.push_back(i);vstep.push_back(dst);vcolumn.push_back(j);vport.push_back(counter);
              counter++;
            }}}
          } else {
            j = i;
            vsrc_step.push_back(step);vsrc_column.push_back(i);vstep.push_back(dst);vcolumn.push_back(j);vport.push_back(counter);
            counter++;
#if 0
            if ( level != par->allowedl ) { 
              // prolongation {{{
              // send data to higher level boundary points
              // determine which, if any, finer mesh cube share a common border with this coarser mesh cube

              // here is the bounding box of the coarser mesh cube
              had_double_type dx = par->dx0/pow(2.0,level);
              had_double_type fdx = par->dx0/pow(2.0,level+1);
              had_double_type xmin = par->min[level] + a*par->granularity*dx;
              had_double_type xmax = par->min[level] + (a*par->granularity+par->granularity-1)*dx; 
              had_double_type ymin = par->min[level] + b*par->granularity*dx;  
              had_double_type ymax = par->min[level] + (b*par->granularity+par->granularity-1)*dx;
              had_double_type zmin = par->min[level] + c*par->granularity*dx;  
              had_double_type zmax = par->min[level] + (c*par->granularity+par->granularity-1)*dx;
#if 0
//              std::cout << " PROLONGATION : " << par->min[level+1]-fdx*par->granularity << " " << par->max[level+1]+fdx*par->granularity << std::endl;
              std::cout << "              : " << xmin << " " << xmax << std::endl;
              std::cout << "              : " << ymin << " " << ymax << std::endl;
              std::cout << "              : " << zmin << " " << zmax << std::endl;
              std::cout << "     " << std::endl;
#endif

              // see if this overlaps a gw region in the next finest level
              int extension0 = 2;
              int extension = par->granularity;
              if ( ( 
                    (  (floatcmp_le(par->min[level+1]-extension0*dx,xmin) 
                     && floatcmp_ge(par->max[level+1]+extension0*dx,xmin)) || 
                       (floatcmp_le(par->min[level+1]-extension0*dx,xmax) 
                     && floatcmp_ge(par->max[level+1]+extension0*dx,xmax)) ) &&

                    (  (floatcmp_le(par->min[level+1]-extension0*dx,ymin) 
                     && floatcmp_ge(par->max[level+1]+extension0*dx,ymin)) || 
                       (floatcmp_le(par->min[level+1]-extension0*dx,ymax) 
                     && floatcmp_ge(par->max[level+1]+extension0*dx,ymax)) ) &&

                    (  (floatcmp_le(par->min[level+1]-extension0*dx,zmin) 
                     && floatcmp_ge(par->max[level+1]+extension0*dx,zmin)) || 
                       (floatcmp_le(par->min[level+1]-extension0*dx,zmax) 
                     && floatcmp_ge(par->max[level+1]+extension0*dx,zmax)) ) ) &&
                  !( par->min[level+1]+par->gw*dx < xmin && par->max[level+1]-par->gw*dx > xmax &&
                     par->min[level+1]+par->gw*dx < ymin && par->max[level+1]-par->gw*dx > ymax &&
                     par->min[level+1]+par->gw*dx < zmin && par->max[level+1]-par->gw*dx > zmax 
                   ) 
                       ) 
              {
                // there is overlap with the prolongation region; 
                // find out which PX thread has the border and send the data there
                // there could be multiple borders to send to
#if 0
              std::cout << " OVERLAP      : " << std::endl;
              std::cout << "              : " << xmin << " " << xmax << std::endl;
              std::cout << "              : " << ymin << " " << ymax << std::endl;
              std::cout << "              : " << zmin << " " << zmax << std::endl;
              std::cout << "     " << std::endl;
#endif

                bool pfound = false;

                // find out which of the level+1 px threads overlap with this px thread
                for (int sk=0;sk<par->nx[level+1];sk++) {
                for (int sj=0;sj<par->nx[level+1];sj++) {
                for (int si=0;si<par->nx[level+1];si++) {
                  if ( si == 0 || si == 1 || si == par->nx[level+1]-2 || si == par->nx[level+1]-1 ||
                       sj == 0 || sj == 1 || sj == par->nx[level+1]-2 || sj == par->nx[level+1]-1 ||
                       sk == 0 || sk == 1 || sk == par->nx[level+1]-2 || sk == par->nx[level+1]-1 ) {
#if 0
                      std::cout << "        " << std::endl;
                      std::cout << "        " << std::endl;
                      std::cout << " DEBUG  : " << xmin << " " << xmax << std::endl;
                      std::cout << "        " << ymin << " " << ymax << std::endl;
                      std::cout << "        " << zmin << " " << zmax << std::endl;
                      std::cout << "        " << std::endl;
                      std::cout << "    fdx  " << fdx << std::endl;
                      std::cout << "        " << std::endl;
                      std::cout << " FINER MESH    : " << par->min[level+1] + si*par->granularity*fdx << " " << par->min[level+1] + (si*par->granularity+par->granularity-1)*fdx << std::endl;
                      std::cout << "               : " << par->min[level+1] + sj*par->granularity*fdx << " " << par->min[level+1] + (sj*par->granularity+par->granularity-1)*fdx << std::endl;
                      std::cout << "               : " << par->min[level+1] + sk*par->granularity*fdx << " " << par->min[level+1] + (sk*par->granularity+par->granularity-1)*fdx << std::endl;
#endif

                    // Check for overlap 
                    if ( (  (floatcmp_le(par->min[level+1] +  si*par->granularity*fdx - extension*dx,xmin) && 
                             floatcmp_ge(par->min[level+1] + (si*par->granularity+par->granularity-1)*fdx + extension*dx,xmin)) ||
                            (floatcmp_le(par->min[level+1] +  si*par->granularity*fdx,xmax - extension*dx) && 
                             floatcmp_ge(par->min[level+1] + (si*par->granularity+par->granularity-1)*fdx + extension*dx,xmax)) 
                         ) &&
                         (  (floatcmp_le(par->min[level+1] +  sj*par->granularity*fdx - extension*dx,ymin) && 
                             floatcmp_ge(par->min[level+1] + (sj*par->granularity+par->granularity-1)*fdx + extension*dx,ymin)) ||
                            (floatcmp_le(par->min[level+1] +  sj*par->granularity*fdx - extension*dx,ymax) && 
                             floatcmp_ge(par->min[level+1] + (sj*par->granularity+par->granularity-1)*fdx + extension*dx,ymax)) 
                         ) &&
                         (  (floatcmp_le(par->min[level+1] +  sk*par->granularity*fdx - extension*dx,zmin) && 
                             floatcmp_ge(par->min[level+1] + (sk*par->granularity+par->granularity-1)*fdx + extension*dx,zmin)) ||
                            (floatcmp_le(par->min[level+1] +  sk*par->granularity*fdx - extension*dx,zmax) && 
                             floatcmp_ge(par->min[level+1] + (sk*par->granularity+par->granularity-1)*fdx + extension*dx,zmax)) 
                         ) 
                       ) 
                    {
                      pfound = true;
                      // This finer mesh point overlaps the coarse mesh point which contains the necessary prolongation info
#if 0
                      std::cout << "        " << std::endl;
                      std::cout << "        " << std::endl;
                      std::cout << " DEBUG A: " << xmin << " " << xmax << std::endl;
                      std::cout << "        " << ymin << " " << ymax << std::endl;
                      std::cout << "        " << zmin << " " << zmax << std::endl;
                      std::cout << "        " << std::endl;
                      std::cout << " FINER MESH A  : " << par->min[level+1] + si*par->granularity*fdx << " " << par->min[level+1] + (si*par->granularity+par->granularity-1)*fdx << std::endl;
                      std::cout << "               : " << par->min[level+1] + sj*par->granularity*fdx << " " << par->min[level+1] + (sj*par->granularity+par->granularity-1)*fdx << std::endl;
                      std::cout << "               : " << par->min[level+1] + sk*par->granularity*fdx << " " << par->min[level+1] + (sk*par->granularity+par->granularity-1)*fdx << std::endl;
                      std::cout << " si sj sk        " << si << " " << sj << " " << sk << std::endl;
#endif
                      // Figure out which px thread this should be sent to
                      j = si + par->nx[level+1]*(sj+sk*par->nx[level+1]);
                      if ( level+1 != par->allowedl ) {
                        j += par->rowsize[level+1];
                      }
                      vsrc_step.push_back(step);vsrc_column.push_back(i);vstep.push_back(dst);vcolumn.push_back(j);vport.push_back(counter);
                      counter++;
                    }
                  }
                } } }

                if ( pfound == false ) {
                  std::cout << " PROBLEM: overlap point not found " << xmin << " " << xmax << " " << ymin << " " << ymax << " " << zmin << " " << zmax << std::endl;
                  std::cout << " GW BBOX : " << par->min[level+1]+par->gw*dx << " " << par->max[level+1]-par->gw*dx << std::endl;
                  std::cout << " BBOX : " << par->min[level+1] << " " << par->max[level+1] << std::endl;
                  std::cout << "  " << std::endl;
                  BOOST_ASSERT(false);
                }
              } 
              // }}}
            }
#endif
#if 0
            if ( level > par->level_row[step] && level == par->prores[step] ) { 
              // restriction {{{
              // send data to all levels less than this one (for restriction)
              had_double_type dx = par->dx0/pow(2.0,level);
              had_double_type xmin = par->min[level] + a*par->granularity*dx;
              had_double_type xmax = par->min[level] + (a*par->granularity+par->granularity-1)*dx; 
              had_double_type ymin = par->min[level] + b*par->granularity*dx;  
              had_double_type ymax = par->min[level] + (b*par->granularity+par->granularity-1)*dx;
              had_double_type zmin = par->min[level] + c*par->granularity*dx;  
              had_double_type zmax = par->min[level] + (c*par->granularity+par->granularity-1)*dx;
              had_double_type /*cxmin,cxmax,cymin,cymax,czmin,czmax,*/cdx; 
              had_double_type ca,cb,cc;
          
              int ii = level-1;
              // determine which coarser mesh cube overlaps this finer mesh cube
              cdx = par->dx0/pow(2.0,ii);
              ca = (xmin - par->min[ii])/(par->granularity*cdx);
              cb = (ymin - par->min[ii])/(par->granularity*cdx);
              cc = (zmin - par->min[ii])/(par->granularity*cdx);

              int starta = (int) floor(ca);
              int startb = (int) floor(cb);
              int startc = (int) floor(cc);
              for (int sk=startc;sk<=startc+1;sk++) {
              for (int sj=startb;sj<=startb+1;sj++) {
              for (int si=starta;si<=starta+1;si++) {
                // check if there is any overlap
                if ( (  (floatcmp_le(par->min[ii] + si*par->granularity*cdx,xmin) && 
                         floatcmp_ge(par->min[ii] + (si*par->granularity+par->granularity-1)*cdx,xmin)) ||
                        (floatcmp_le(par->min[ii] + si*par->granularity*cdx,xmax) && 
                         floatcmp_ge(par->min[ii] + (si*par->granularity+par->granularity-1)*cdx,xmax)) 
                     ) &&
                     (  (floatcmp_le(par->min[ii] + sj*par->granularity*cdx,ymin) && 
                         floatcmp_ge(par->min[ii] + (sj*par->granularity+par->granularity-1)*cdx,ymin)) ||
                        (floatcmp_le(par->min[ii] + sj*par->granularity*cdx,ymax) && 
                         floatcmp_ge(par->min[ii] + (sj*par->granularity+par->granularity-1)*cdx,ymax)) 
                     ) &&
                     (  (floatcmp_le(par->min[ii] + sk*par->granularity*cdx,zmin) && 
                         floatcmp_ge(par->min[ii] + (sk*par->granularity+par->granularity-1)*cdx,zmin)) ||
                        (floatcmp_le(par->min[ii] + sk*par->granularity*cdx,zmax) && 
                         floatcmp_ge(par->min[ii] + (sk*par->granularity+par->granularity-1)*cdx,zmax)) 
                     ) 
                   ) 
                {

#if 0
                  // DEBUG
                  std::cout << " " << std::endl;
                  std::cout << " DEBUG coarse: x " << par->min[ii] + si*par->granularity*cdx << " " << par->min[ii] + (si*par->granularity+par->granularity-1)*cdx << std::endl;
                  std::cout << " DEBUG coarse: y " << par->min[ii] + sj*par->granularity*cdx << " " << par->min[ii] + (sj*par->granularity+par->granularity-1)*cdx << std::endl;
                  std::cout << " DEBUG coarse: z " << par->min[ii] + sk*par->granularity*cdx << " " << par->min[ii] + (sk*par->granularity+par->granularity-1)*cdx << std::endl;
                  std::cout << " DEBUG fine: x " << xmin << " " << xmax << std::endl;
                  std::cout << " DEBUG fine: y " << ymin << " " << ymax << std::endl;
                  std::cout << " DEBUG fine: z " << zmin << " " << zmax << std::endl;
#endif

                  // there is overlap -- some points from the specified finer mesh
                  // need to be sent to this particular coarse mesh
                  j = si + par->nx[ii]*(sj+sk*par->nx[ii]);

                  if ( ii != par->allowedl ) {
                    j += par->rowsize[ii+1];
                  }

                  vsrc_step.push_back(step);vsrc_column.push_back(i);vstep.push_back(dst);vcolumn.push_back(j);vport.push_back(counter);
                  counter++;
                } 
              } } }
            // }}}
            }
#endif

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
      int t1,kk;
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

    // This routine finds the level of a specified point given its row (step) and column (point)
    std::size_t unigrid_mesh::findlevel3D(std::size_t step, std::size_t item, std::size_t &a, std::size_t &b, std::size_t &c, Parameter const& par)
    {
      int ll = par->level_row[step];
      // discover what level to which this point belongs
      int level = -1;
      if ( ll == par->allowedl ) {
        level = ll;
        // get 3D coordinates from 'i' value
        // i.e. i = a + nx*(b+c*nx);
        int tmp_index = item/par->nx[ll];
        c = tmp_index/par->nx[ll];
        b = tmp_index%par->nx[ll];
        a = item - par->nx[ll]*(b+c*par->nx[ll]);
        BOOST_ASSERT(item == a + par->nx[ll]*(b+c*par->nx[ll]));
      } else {
        if ( item < par->rowsize[par->allowedl] ) {
          level = par->allowedl;
        } else {
          for (int j=par->allowedl-1;j>=ll;j--) {
            if ( item < par->rowsize[j] && item >= par->rowsize[j+1] ) { 
              level = j;
              break;
            }
          }
        }
 
        if ( level < par->allowedl ) {
          int tmp_index = (item - par->rowsize[level+1])/par->nx[level];
          c = tmp_index/par->nx[level];
          b = tmp_index%par->nx[level];
          a = (item-par->rowsize[level+1]) - par->nx[level]*(b+c*par->nx[level]);
          BOOST_ASSERT(item-par->rowsize[level+1] == a + par->nx[level]*(b+c*par->nx[level]));
        } else {
          int tmp_index = item/par->nx[level];
          c = tmp_index/par->nx[level];
          b = tmp_index%par->nx[level];
          a = item - par->nx[level]*(b+c*par->nx[level]);
        }
      }
      BOOST_ASSERT(level >= 0);
      return level;
    }

}}}}
