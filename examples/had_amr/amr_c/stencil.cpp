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
              //printf(" TEST left %d mid %d right %d\n",val1->refine_,val2->refine_,val3->refine_);
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

            // this will be a parameter someday
            std::size_t allowedl = 1;
            if ( val2->refine_ && val2->level_ < allowedl && gids.size() == 3 ) {
              naming::id_type gval1, gval2, gval3, gval4, gval5;
              boost::tie(gval1, gval2, gval3, gval4, gval5) = 
                                components::wait(components::stubs::memory_block::clone_async(gids[1]), 
                                     components::stubs::memory_block::clone_async(gids[1]),
                                     components::stubs::memory_block::clone_async(gids[1]),
                                     components::stubs::memory_block::clone_async(gids[1]),
                                     components::stubs::memory_block::clone_async(gids[1]));

              access_memory_block<stencil_data> mval1, mval2,mval3,mval4,mval5;
              boost::tie(mval1,mval2,mval3,mval4,mval5) = 
                              components::wait(components::stubs::memory_block::get_async(gval1), 
                                   components::stubs::memory_block::get_async(gval2),
                                   components::stubs::memory_block::get_async(gval3),
                                   components::stubs::memory_block::get_async(gval4),
                                   components::stubs::memory_block::get_async(gval5));

              // increase the level by one
              ++mval1->level_;
              ++mval2->level_;
              ++mval3->level_;
              ++mval4->level_;
              ++mval5->level_;

              // initialize timestep for the fine mesh
              mval1->timestep_ = 0;
              mval2->timestep_ = 0;
              mval3->timestep_ = 0;
              mval4->timestep_ = 0;
              mval5->timestep_ = 0;

              // initialize the index
              mval1->index_ = 0;
              mval2->index_ = 1;
              mval3->index_ = 2;
              mval4->index_ = 3;
              mval5->index_ = 4;

              // the initial data for the child mesh comes from the parent mesh
              naming::id_type here = applier::get_applier().get_runtime_support_gid();
              components::component_type logging_type =
                        components::get_component_type<components::amr::server::logging>();
              components::component_type function_type =
                        components::get_component_type<components::amr::stencil>();
              components::amr::amr_mesh child_mesh (
                        components::amr::amr_mesh::create(here, 1, true));
              std::vector<naming::id_type> initial_data;
              initial_data.push_back(gval1);
              initial_data.push_back(gval2);
              initial_data.push_back(gval3);
              initial_data.push_back(gval4);
              initial_data.push_back(gval5);

              std::size_t numvalues = 5;
              std::size_t numsteps = 2;
              std::size_t stencilsize = 3;
              std::vector<naming::id_type> result_data(
                          child_mesh.execute(initial_data,function_type,numvalues,numsteps,stencilsize,
                          logging_type));

              access_memory_block<stencil_data> r_val1, r_val2;
              boost::tie(r_val1, r_val2) = 
               wait(components::stubs::memory_block::get_async(result_data[2]), 
                    components::stubs::memory_block::get_async(result_data[3]));

              // overwrite the coarse point computation
              resultval->value_ = r_val1->value_;
      
              // evaluate result data
              resultval->right_alloc_ = 1;
              resultval->right_value_ = r_val2->value_;

              // release result data
              for (std::size_t i = 0; i < result_data.size(); ++i) 
                components::stubs::memory_block::free(result_data[i]);

            }


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

