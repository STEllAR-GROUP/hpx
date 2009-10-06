//  Copyright (c) 2007-2009 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/lcos/future_wait.hpp>

#include <boost/foreach.hpp>

#include "../amr/amr_mesh.hpp"

#include "stencil.hpp"
#include "logging.hpp"
#include "stencil_data.hpp"
#include "stencil_functions.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr 
{
    stencil::stencil()
      : numsteps_(0)
    {}

    ///////////////////////////////////////////////////////////////////////////
    // Implement actual functionality of this stencil
    // Compute the result value for the current time step
    int stencil::eval(naming::id_type const& result, 
        std::vector<naming::id_type> const& gids, int row, int column)
    {
        BOOST_ASSERT(gids.size() <= 3);

        // make sure all the gids are looking valid
        if (result == naming::invalid_id)
        {
            HPX_THROW_EXCEPTION(bad_parameter,
                "stencil::eval", "result gid is invalid");
            return -1;
        }

        // this should occur only after result has been delivered already
        BOOST_FOREACH(naming::id_type gid, gids)
        {
            if (gid == naming::invalid_id)
                return -1;
        }

        // start asynchronous get operations

        // get all input memory_block_data instances
        access_memory_block<stencil_data> val1, val2, val3, resultval;
        if (gids.size() == 3) { 
          boost::tie(val1, val2, val3, resultval) = 
            wait(components::stubs::memory_block::get_async(gids[0]), 
                 components::stubs::memory_block::get_async(gids[1]), 
                 components::stubs::memory_block::get_async(gids[2]),
                 components::stubs::memory_block::get_async(result));
        } else if ( gids.size() == 2) {
          boost::tie(val1, val2, resultval) = 
            wait(components::stubs::memory_block::get_async(gids[0]), 
                 components::stubs::memory_block::get_async(gids[1]), 
                 components::stubs::memory_block::get_async(result));
        } else {
          BOOST_ASSERT(0 == 1);
        }

        // make sure all input data items agree on the time step number
       // BOOST_ASSERT(val1->timestep_ == val2->timestep_);
       // if ( gids.size() == 3 ) {
       //   BOOST_ASSERT(val1->timestep_ == val3->timestep_);
       // }

        // the predecessor
        std::size_t middle_timestep;
        if ( gids.size() == 3 ) middle_timestep = val2->timestep_;
        else if ( gids.size() == 2 && column == 0 ) middle_timestep = val2->timestep_;
        else middle_timestep = val1->timestep_;
        if (middle_timestep < numsteps_) {

            if ( gids.size() == 3 ) {
              // this is the actual calculation, call provided (external) function
              evaluate_timestep(val1.get_ptr(), val2.get_ptr(), val3.get_ptr(), 
                  resultval.get_ptr(), numsteps_);
              printf(" TEST left %d mid %d right %d\n",val1->refine_,val2->refine_,val3->refine_);
            } else if ( gids.size() == 2 ) {
              // bdry computation
              if ( column == 0 ) {
                evaluate_left_bdry_timestep(val1.get_ptr(), val2.get_ptr(),
                  resultval.get_ptr(), numsteps_);
              } else {
                evaluate_right_bdry_timestep(val1.get_ptr(), val2.get_ptr(),
                  resultval.get_ptr(), numsteps_);
              }
            }

            // Interpolation 

            // Instatiate new mesh
            // Check if a finer gid has been created
            if ( gids.size() == 3 ) {
              if ( val1->right_neighbor_.id_lsb_ > 5000 && 
                   val2->right_neighbor_.id_lsb_ > 5000 ) {
                // the initial data for the child mesh comes from the parent mesh
                naming::id_type here = applier::get_applier().get_runtime_support_gid();
                components::component_type logging_type =
                          components::get_component_type<components::amr::server::logging>();
                components::component_type function_type =
                          components::get_component_type<components::amr::stencil>();
                components::amr::amr_mesh child_mesh (
                          components::amr::amr_mesh::create(here, 1, true));
                std::vector<naming::id_type> initial_data;
                initial_data.push_back(val1->right_neighbor_);
                initial_data.push_back(gids[1]);
                initial_data.push_back(val2->right_neighbor_);
  
                std::vector<naming::id_type> result_data(
                            child_mesh.execute(initial_data,function_type,3,2,3,
                            logging_type));

                // evaluate result data
                // release initial data
                // release result data
              }
            }

            // clone the gids and communicate
            if ( gids.size() == 3 ) {
              memset(&resultval->right_neighbor_,0,sizeof(gid));
              memset(&resultval->left_neighbor_,0,sizeof(gid));
              if ( val1->refine_  && val2->refine_  && val3->refine_) {
                // center refinement -- clone a right neighbor
                resultval->right_neighbor_ = 
                  components::stubs::memory_block::clone(gids[1]);
              } else if ( val1->refine_ && val2->refine_ && !val3->refine_ ) {
                // 1 point away from right boundary -- clone a right neighbor
                resultval->right_neighbor_ = 
                  components::stubs::memory_block::clone(gids[1]);
              } else if ( val1->refine_ && !val2->refine_ && val3->refine_ ) {
                // point inbetween refinement boundaries -- clone a right neighbor
                resultval->right_neighbor_ = 
                  components::stubs::memory_block::clone(gids[1]);
              } else if ( val1->refine_ && !val2->refine_ && !val3->refine_ ) {
                // right boundary -- clone a right neighbor
                resultval->right_neighbor_ = 
                  components::stubs::memory_block::clone(gids[1]);
              } else if ( !val1->refine_ && val2->refine_ && val3->refine_ ) {
                // 1 point away from left boundary -- clone a right neigbhor
                resultval->right_neighbor_ = 
                  components::stubs::memory_block::clone(gids[1]);
              } else if ( !val1->refine_ && val2->refine_ && !val3->refine_ ) {
                // unique refinement node -- no cloning
              } else if ( !val1->refine_ && !val2->refine_ && val3->refine_ ) {
                // left boundary -- clone left and right neighbor
                boost::tie(resultval->right_neighbor_,resultval->left_neighbor_) = 
                  components::wait(components::stubs::memory_block::clone_async(gids[1]),
                                   components::stubs::memory_block::clone_async(gids[1]));
              } else if ( !val1->refine_ && !val2->refine_ && !val3->refine_ ) {
                // no refinement -- no cloning
              }

            }

#if 0
            // this will be a parameter someday
            std::size_t allowedl = 1;
            if ( val2->refine_ && val2->level_ <= allowedl && gids.size() == 3 ) {
              boost::tie(val2->left_neighbor_, val2->middle_neighbor_,val2->right_neighbor_) = 
                components::wait(components::stubs::memory_block::clone_async(gids[0]), 
                      components::stubs::memory_block::clone_async(gids[1]),
                      components::stubs::memory_block::clone_async(gids[2]));

              // the initial data for the child mesh comes from the parent mesh
              naming::id_type here = applier::get_applier().get_runtime_support_gid();
              components::component_type logging_type =
                        components::get_component_type<components::amr::server::logging>();
              components::component_type function_type =
                        components::get_component_type<components::amr::stencil>();
              components::amr::amr_mesh child_mesh (
                        components::amr::amr_mesh::create(here, 1, true));
              std::vector<naming::id_type> initial_data;

              std::vector<naming::id_type> result_data(
                          child_mesh.execute(initial_data,function_type,3,2,3,
                          logging_type));

              // evaluate result data
              // release initial data
              // release result data
            }
#endif


            if (log_)     // send result to logging instance
                stubs::logging::logentry(log_, resultval.get(), row);
        }
        else {
            // the last time step has been reached, just copy over the data
            resultval.get() = val2.get();
            ++resultval->timestep_;
        }
        // set return value difference between actual and required number of
        // timesteps (>0: still to go, 0: last step, <0: overdone)
        return numsteps_ - resultval->timestep_;
    }

    naming::id_type stencil::alloc_data(int item, int maxitems, int row)
    {
        naming::id_type result = components::stubs::memory_block::create(
            applier::get_applier().get_runtime_support_gid(), sizeof(stencil_data));

        if (-1 != item) {
            // provide initial data for the given data value 
            access_memory_block<stencil_data> val(
                components::stubs::memory_block::checkout(result));

            // call provided (external) function
            generate_initial_data(val.get_ptr(), item, maxitems, row);

            if (log_)         // send initial value to logging instance
                stubs::logging::logentry(log_, val.get(), row);
        }
        return result;
    }

    void stencil::init(std::size_t numsteps, naming::id_type const& logging)
    {
        numsteps_ = numsteps;
        log_ = logging;
    }

}}}

