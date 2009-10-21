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
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        // helper functions to get several memory pointers asynchronously
        inline boost::tuple<
            access_memory_block<stencil_data>, access_memory_block<stencil_data>
          , access_memory_block<stencil_data> >
        get_async(naming::id_type const& g1, naming::id_type const& g2
          , naming::id_type const& g3)
        {
            return wait(components::stubs::memory_block::get_async(g1)
              , components::stubs::memory_block::get_async(g2)
              , components::stubs::memory_block::get_async(g3));
        }

        inline boost::tuple<
            access_memory_block<stencil_data>, access_memory_block<stencil_data>
          , access_memory_block<stencil_data>, access_memory_block<stencil_data> >
        get_async(naming::id_type const& g1, naming::id_type const& g2
          , naming::id_type const& g3, naming::id_type const& g4)
        {
            return wait(components::stubs::memory_block::get_async(g1)
              , components::stubs::memory_block::get_async(g2)
              , components::stubs::memory_block::get_async(g3)
              , components::stubs::memory_block::get_async(g4));
        }

        inline boost::tuple<
            access_memory_block<stencil_data>, access_memory_block<stencil_data>
          , access_memory_block<stencil_data>, access_memory_block<stencil_data>
          , access_memory_block<stencil_data> >
        get_async(naming::id_type const& g1, naming::id_type const& g2
          , naming::id_type const& g3, naming::id_type const& g4
          , naming::id_type const& g5)
        {
            return wait(components::stubs::memory_block::get_async(g1)
              , components::stubs::memory_block::get_async(g2)
              , components::stubs::memory_block::get_async(g3)
              , components::stubs::memory_block::get_async(g4)
              , components::stubs::memory_block::get_async(g5));
        }
    }

    stencil::stencil()
      : numsteps_(0)
    {}

    ///////////////////////////////////////////////////////////////////////////
    // Implement actual functionality of this stencil
    // Compute the result value for the current time step
    int stencil::eval(naming::id_type const& result, 
        std::vector<naming::id_type> const& gids, int row, int column,
        server::Parameter const& par)
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
                detail::get_async(gids[0], gids[1], gids[2], result);
        } 
        else if (gids.size() == 2) {
            boost::tie(val1, val2, resultval) = 
                detail::get_async(gids[0], gids[1], result);
        } 
        else {
            BOOST_ASSERT(false);    // should not happen
        }

        // make sure all input data items agree on the time step number
       // BOOST_ASSERT(val1->timestep_ == val2->timestep_);
       // if ( gids.size() == 3 ) {
       //   BOOST_ASSERT(val1->timestep_ == val3->timestep_);
       // }

        // the predecessor
        std::size_t middle_timestep;
        if (gids.size() == 3) 
            middle_timestep = val2->timestep_;
        else if (gids.size() == 2 && column == 0) 
            middle_timestep = val1->timestep_;      // left boundary point
        else {
            BOOST_ASSERT(gids.size() == 2);
            middle_timestep = val2->timestep_;      // right boundary point
        }

        if (middle_timestep < numsteps_) {

            if (gids.size() == 3) {
              // this is the actual calculation, call provided (external) function
              evaluate_timestep(val1.get_ptr(), val2.get_ptr(), val3.get_ptr(), 
                  resultval.get_ptr(), numsteps_);

              // copy over the coordinate value to the result
              resultval->x_ = val2->x_;
            } 
            else if (gids.size() == 2) {
              // bdry computation
              if ( column == 0 ) {
                evaluate_left_bdry_timestep(val1.get_ptr(), val2.get_ptr(),
                  resultval.get_ptr(), numsteps_);

                // copy over the coordinate value to the result
                resultval->x_ = val1->x_;
              } else {
                evaluate_right_bdry_timestep(val1.get_ptr(), val2.get_ptr(),
                  resultval.get_ptr(), numsteps_);

                // copy over the coordinate value to the result
                resultval->x_ = val2->x_;
              }
            }

            // copy over the coordinate value to the result

            std::size_t allowedl = par.allowedl;
            if ( val2->refine_ && gids.size() == 3 && val2->level_ < allowedl ) {
              finer_mesh(result, gids,par);
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

    ///////////////////////////////////////////////////////////////////////////
    // Implement a finer mesh via interpolation of inter-mesh points
    // Compute the result value for the current time step
    int stencil::finer_mesh(naming::id_type const& result, 
        std::vector<naming::id_type> const& gids,
        server::Parameter const& par) 
    {

      naming::id_type gval1, gval2, gval3, gval4, gval5;
      boost::tie(gval1, gval2, gval3, gval4, gval5) = 
                        components::wait(components::stubs::memory_block::clone_async(gids[0]), 
                             components::stubs::memory_block::clone_async(gids[1]),
                             components::stubs::memory_block::clone_async(gids[1]),
                             components::stubs::memory_block::clone_async(gids[1]),
                             components::stubs::memory_block::clone_async(gids[2]));

      access_memory_block<stencil_data> mval1, mval2, mval3, mval4, mval5;
      boost::tie(mval1, mval2, mval3, mval4, mval5) = 
          detail::get_async(gval1, gval2, gval3, gval4, gval5);

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

      // if the refined point already exists, no need to interpolate
      if (mval1->right_alloc_ == 1 && mval1->right_level_ == mval1->level_) 
          mval2->value_ = mval1->right_value_; 
      if (mval3->right_alloc_ == 1 && mval3->right_level_ == mval3->level_) 
          mval4->value_ = mval3->right_value_; 

      // this updates the coordinate position
      mval2->x_ = 0.5*(mval1->x_ + mval3->x_);
      mval4->x_ = 0.5*(mval3->x_ + mval5->x_);

      // call to user defined interpolation
      interpolation();

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

      bool do_logging = false;
      if ( par.loglevel > 0 ) {
        do_logging = true;
      }
      std::vector<naming::id_type> result_data(
          child_mesh.execute(initial_data, function_type, numvalues, numsteps, 
            do_logging ? logging_type : components::component_invalid,par));

      access_memory_block<stencil_data> r_val1, r_val2, resultval;
      boost::tie(r_val1, r_val2, resultval) = 
          detail::get_async(result_data[2], result_data[3], result);

      // overwrite the coarse point computation
      resultval->value_ = r_val1->value_;

      // remember right neighbor value
      resultval->right_alloc_ = 1;
      resultval->right_value_ = r_val2->value_;
      resultval->right_level_ = r_val2->level_;

      // release result data
      for (std::size_t i = 0; i < result_data.size(); ++i) 
          components::stubs::memory_block::free(result_data[i]);

      for (std::size_t i = 0; i < initial_data.size(); ++i) 
          components::stubs::memory_block::free(initial_data[i]);

      return 0;
    }

    ///////////////////////////////////////////////////////////////////////////
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

