//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Matthew Anderson
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/lcos/future_wait.hpp>
#include <hpx/runtime/components/memory_block.hpp>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/export.hpp>

#include <boost/foreach.hpp>

#include <math.h>

#include "stencil.hpp"
#include "logging.hpp"
#include "stencil_data.hpp"
#include "stencil_functions.hpp"
#include "stencil_data_locking.hpp"
#include "../dataflow/dataflow_stencil.hpp"

// floatcmp_le {{{
HPX_COMPONENT_EXPORT bool floatcmp_le(double const& x1, double const& x2) {
  // compare two floating point numbers
  static double const epsilon = 1.e-8;

  if ( x1 < x2 ) return true;

  if ( x1 + epsilon >= x2 && x1 - epsilon <= x2 ) {
    // the numbers are close enough for coordinate comparison
    return true;
  } else {
    return false;
  }
}
// }}}

// floatcmp_ge {{{
HPX_COMPONENT_EXPORT bool floatcmp_ge(double const& x1, double const& x2) {
  // compare two floating point numbers
  static double const epsilon = 1.e-8;

  if ( x1 > x2 ) return true;

  if ( x1 + epsilon >= x2 && x1 - epsilon <= x2 ) {
    // the numbers are close enough for coordinate comparison
    return true;
  } else {
    return false;
  }
}
// }}}

// floatcmp {{{
HPX_COMPONENT_EXPORT int floatcmp(double const& x1, double const& x2) {
  // compare two floating point numbers
  static double const epsilon = 1.e-8;
  if ( x1 + epsilon >= x2 && x1 - epsilon <= x2 ) {
    // the numbers are close enough for coordinate comparison
    return true;
  } else {
    return false;
  }
}
// }}}


///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace adaptive1d
{
    ///////////////////////////////////////////////////////////////////////////
    // memory block support for config data needed for serialization of
    // stencil_data
    typedef hpx::actions::manage_object_action<server::stencil_config_data>
        manage_object_config_data_action;
    manage_object_config_data_action const manage_stencil_config_data =
        manage_object_config_data_action();

    // memory block support for stencil data (user defined data)
    typedef hpx::actions::manage_object_action<
        stencil_data, server::stencil_config_data> manage_object_data_action;
    manage_object_data_action const manage_stencil_data =
        manage_object_data_action();

//     // memory block support for stencil data (user defined data)
//     typedef hpx::actions::manage_object_action<stencil_data>
//         manage_object_data_action_simple;
//     manage_object_data_action_simple const manage_stencil_data_simple =
//         manage_object_data_action_simple();

    ///////////////////////////////////////////////////////////////////////////
    memory_block_data stencil_config_data::create_and_resolve_target()
    {
        mem_block.create(hpx::find_here(), sizeof(server::stencil_config_data),
            manage_stencil_config_data);
        return mem_block.get();
    }

    stencil_config_data::stencil_config_data(int face, int count)
    {
        // create new instance
        static_cast<base_type&>(*this) = create_and_resolve_target();

        // initialize from arguments
        (*this)->face_ = face;
        (*this)->count_ = count;
    }
}}}

///////////////////////////////////////////////////////////////////////////////
HPX_REGISTER_MANAGE_OBJECT_ACTION(
    hpx::components::adaptive1d::manage_object_config_data_action,
    dataflow_manage_object_action_stencil_config_data)

HPX_REGISTER_MANAGE_OBJECT_ACTION(
    hpx::components::adaptive1d::manage_object_data_action,
    dataflow_manage_object_action_stencil_data)

// HPX_REGISTER_MANAGE_OBJECT_ACTION(
//     hpx::components::adaptive1d::manage_object_data_action_simple,
//     dataflow_manage_object_action_stencil_data_simple)

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace adaptive1d
{
    ///////////////////////////////////////////////////////////////////////////
    stencil::stencil()
      : numsteps_(0)
    {
    }

    inline double
    stencil::interp_linear(double y1, double y2,
                           double x, double x1, double x2) {
      double xx1 = x - x1;
      double xx2 = x - x2;
      double result = xx2*y1/( (x1-x2) ) + xx1*y2/( (x2-x1) );

      return result;
    }

    void stencil::interpolate(double x, double minx,double h,
                              access_memory_block<stencil_data> &val,
                              nodedata &result, parameter const& par) {
      int num_eqns = NUM_EQUATIONS;

      // set up index bounds
      int ii = (int) ( (x-minx)/h );

      bool no_interp_x = false;

      if ( floatcmp(h*ii + minx,x) ) {
        no_interp_x = true;
      }

      if ( no_interp_x ) {
        for (int ll=0;ll<num_eqns;++ll) {
          result.phi[0][ll] = val->value_[ii].phi[0][ll];
        }
        return;
      }

      // Quick sanity check to be sure we have bracketed the point we wish to interpolate
      if ( !no_interp_x ) {
        BOOST_ASSERT(floatcmp_le(h*ii+minx,x) && floatcmp_ge(h*(ii+1)+minx,x) );
      }

      for (int ll=0;ll<num_eqns;++ll) {
        result.phi[0][ll] = interp_linear(val->value_[ii].phi[0][ll],
                                          val->value_[ii+1].phi[0][ll],                                                                                           x, h*ii+minx,h*(ii+1)+minx);
      }
      return;
    }


    ///////////////////////////////////////////////////////////////////////////
    // Implement actual functionality of this stencil
    // Compute the result value for the current time step
    int stencil::eval(naming::id_type const& result,
        std::vector<naming::id_type> const& gids, std::size_t row, std::size_t column,
        double cycle_time, parameter const& par)
    {
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

        bool prolongation;
        if ( par->allowedl == 0 ) {
          prolongation = false;
        } else if ( (row+3)%3 != 0 ) {
          prolongation = false;
        } else {
          prolongation = true;
        }

        if ( prolongation == false ) {
          // {{{
          if ( gids.size() == 1  ) {
            // get all input memory_block_data instances
            typedef std::vector<lcos::future<memory_block_data> >
              lazy_results_type;

            // first invoke all remote operations
            lazy_results_type lazy_results;

            namespace s = hpx::components::stubs;
            lazy_results.push_back(s::memory_block::get_async(gids[0]));

            //  invoke the operation for the result gid as well
            lazy_results.push_back(s::memory_block::get_async(result));

            // then wait for all results to get back to us
            std::vector<access_memory_block<stencil_data> > val;
            BOOST_FOREACH(lcos::future<memory_block_data> const& f, lazy_results)
              val.push_back(f.get());

            // lock all user defined data elements, will be unlocked at function exit
            scoped_values_lock<lcos::local::mutex> l(val);

            val[1]->max_index_ = val[0]->max_index_;
            val[1]->index_ = val[0]->index_;
            val[1]->value_.resize(val[0]->value_.size());
            for (std::size_t i=0;i<val[0]->value_.size();i++) {
              val[1]->value_[i].x = val[0]->value_[i].x;
            }

            double t = val[0]->timestep_*par->h*par->cfl + cycle_time;
            rkupdate1(val,t,*par.p);

            val[1]->timestep_ = val[0]->timestep_ + 1.0;
            if (val[1]->timestep_ >= par->nt0-1) {
              return 0;
            }
            return 1;
          } else if ( gids.size() == 2 ) {
            // get all input memory_block_data instances
            typedef std::vector<lcos::future<memory_block_data> >
              lazy_results_type;

            // first invoke all remote operations
            lazy_results_type lazy_results;
            namespace s = hpx::components::stubs;

            if ( par->gr_lbox[ par->item2gi[column] ] ) {
              stencil_config_data cfg1(1,3*par->num_neighbors);  // serializes the left face coming from the right
              lazy_results.push_back(s::memory_block::get_async(gids[0]));
              lazy_results.push_back(s::memory_block::get_async(gids[1], cfg1.get_memory_block()));
            } else if ( par->gr_rbox[ par->item2gi[column] ] ) {
              stencil_config_data cfg0(0,3*par->num_neighbors);  // serializes the right face coming from the left
              lazy_results.push_back(s::memory_block::get_async(gids[0], cfg0.get_memory_block()));
              lazy_results.push_back(s::memory_block::get_async(gids[1]));
            } else {
              BOOST_ASSERT(false);
            }
            //  invoke the operation for the result gid as well
            lazy_results.push_back(s::memory_block::get_async(result));

            // then wait for all results to get back to us
            std::vector<access_memory_block<stencil_data> > val;
            BOOST_FOREACH(lcos::future<memory_block_data> const& f, lazy_results)
              val.push_back(f.get());

            // lock all user defined data elements, will be unlocked at function exit
            scoped_values_lock<lcos::local::mutex> l(val);

            val[2]->max_index_ = val[0]->max_index_;
            val[2]->index_ = val[0]->index_;
            val[2]->value_.resize(val[0]->value_.size());
            for (std::size_t i=0;i<val[0]->value_.size();i++) {
              val[2]->value_[i].x = val[0]->value_[i].x;
            }

            double t = val[0]->timestep_*par->h*par->cfl + cycle_time;
            if ( par->gr_lbox[ par->item2gi[column] ] ) {
              rkupdate2b(val,t,*par.p);
            } else if ( par->gr_rbox[ par->item2gi[column] ] ) {
              rkupdate2a(val,t,*par.p);
            } else {
              BOOST_ASSERT(false);
            }

            val[2]->timestep_ = val[0]->timestep_ + 1.0;
            if (val[2]->timestep_ >= par->nt0-1) {
              return 0;
            }
            return 1;
          } else if ( gids.size() == 3 ) {
            // Generate new config info
            stencil_config_data cfg0(0,3*par->num_neighbors);  // serializes the right face coming from the left
            stencil_config_data cfg1(1,3*par->num_neighbors);  // serializes the left face coming from the right

            // get all input memory_block_data instances
            typedef std::vector<lcos::future<memory_block_data> >
                lazy_results_type;

            // first invoke all remote operations
            lazy_results_type lazy_results;

            namespace s = hpx::components::stubs;
            lazy_results.push_back(
                s::memory_block::get_async(gids[0], cfg0.get_memory_block()));
            lazy_results.push_back(s::memory_block::get_async(gids[1]));
            lazy_results.push_back(
                s::memory_block::get_async(gids[2], cfg1.get_memory_block()));

            //  invoke the operation for the result gid as well
            lazy_results.push_back(s::memory_block::get_async(result));

            // then wait for all results to get back to us
            std::vector<access_memory_block<stencil_data> > val;
            BOOST_FOREACH(lcos::future<memory_block_data> const& f, lazy_results)
              val.push_back(f.get());

            // lock all user defined data elements, will be unlocked at function exit
            scoped_values_lock<lcos::local::mutex> l(val);

            val[3]->max_index_ = val[1]->max_index_;
            val[3]->index_ = val[1]->index_;
            val[3]->value_.resize(val[1]->value_.size());
            for (std::size_t i=0;i<val[1]->value_.size();i++) {
              val[3]->value_[i].x = val[1]->value_[i].x;
            }

            double t = val[1]->timestep_*par->h*par->cfl + cycle_time;
            rkupdate3(val,t,*par.p);

            val[3]->timestep_ = val[1]->timestep_ + 1.0;

            //std::cout << " row " << row << " column " << column
            //    << " timestep " << val[3]->timestep_
            //    << " left " << val[0]->value_.size() << "(" << val[0]->value_.data_size() << ")"
            //    << " middle " << val[1]->value_.size() << "(" << val[1]->value_.data_size() << ")"
            //    << " right " << val[2]->value_.size() << "(" << val[2]->value_.data_size() << ")"
            //    << std::endl;

            if (val[3]->timestep_ >= par->refine_every-1) {
              return 0;
            }
            return 1;
          } else {
            BOOST_ASSERT(false);
          }
          // }}}
        } else {
          // {{{
          if ( gids.size() == 1  ) {
            // this case just a copy
            // get all input memory_block_data instances
            typedef std::vector<lcos::future<memory_block_data> >
              lazy_results_type;

            // first invoke all remote operations
            lazy_results_type lazy_results;

            namespace s = hpx::components::stubs;
            lazy_results.push_back(s::memory_block::get_async(gids[0]));

            //  invoke the operation for the result gid as well
            lazy_results.push_back(s::memory_block::get_async(result));

            // then wait for all results to get back to us
            std::vector<access_memory_block<stencil_data> > val;
            BOOST_FOREACH(lcos::future<memory_block_data> const& f, lazy_results)
              val.push_back(f.get());

            // lock all user defined data elements, will be unlocked at function exit
            scoped_values_lock<lcos::local::mutex> l(val);

            val[1].get() = val[0].get();

            if (val[1]->timestep_ >= par->nt0-1) {
              return 0;
            }
            return 1;
          } else if ( gids.size() == 2  ) {
            // prolongation/restriction
            // get all input memory_block_data instances
            typedef std::vector<lcos::future<memory_block_data> >
              lazy_results_type;

            // first invoke all remote operations
            lazy_results_type lazy_results;
            namespace s = hpx::components::stubs;

            lazy_results.push_back(s::memory_block::get_async(gids[0]));
            lazy_results.push_back(s::memory_block::get_async(gids[1]));

            //  invoke the operation for the result gid as well
            lazy_results.push_back(s::memory_block::get_async(result));

            // then wait for all results to get back to us
            std::vector<access_memory_block<stencil_data> > val;
            BOOST_FOREACH(lcos::future<memory_block_data> const& f, lazy_results)
              val.push_back(f.get());

            // lock all user defined data elements, will be unlocked at function exit
            scoped_values_lock<lcos::local::mutex> l(val);

            val[2].get() = val[0].get();

            // ghostwidth interpolation
            if ( par->gr_lbox[ par->item2gi[column] ] ) {
              for (std::size_t i=0;i<par->ghostwidth;i++) {
                // the point we are interpolating to
                double x = val[2]->value_[i].x;

                double minx = val[1]->value_[0].x;
                double h = val[1]->value_[1].x-val[1]->value_[0].x;

                interpolate(x,minx,h,
                            val[1],val[2]->value_[i],par);
              }
            } else if ( par->gr_rbox[ par->item2gi[column] ] ) {
              for (std::size_t i=val[2]->value_.size()-par->ghostwidth;i<val[2]->value_.size();i++) {
                double x = val[2]->value_[i].x;

                double minx = val[1]->value_[0].x;
                double h = val[1]->value_[1].x-val[1]->value_[0].x;

                interpolate(x,minx,h,
                            val[1],val[2]->value_[i],par);
              }
            }

            if (val[2]->timestep_ >= par->nt0-1) {
              return 0;
            }
            return 1;
          } else if ( gids.size() == 3  ) {
            BOOST_ASSERT(par->gr_lbox[ par->item2gi[column] ] && par->gr_rbox[ par->item2gi[column] ] );

            // prolongation/restriction
            // get all input memory_block_data instances
            typedef std::vector<lcos::future<memory_block_data> >
              lazy_results_type;

            // first invoke all remote operations
            lazy_results_type lazy_results;
            namespace s = hpx::components::stubs;

            lazy_results.push_back(s::memory_block::get_async(gids[0]));
            lazy_results.push_back(s::memory_block::get_async(gids[1]));
            lazy_results.push_back(s::memory_block::get_async(gids[2]));

            //  invoke the operation for the result gid as well
            lazy_results.push_back(s::memory_block::get_async(result));

            // then wait for all results to get back to us
            std::vector<access_memory_block<stencil_data> > val;
            BOOST_FOREACH(lcos::future<memory_block_data> const& f, lazy_results)
                val.push_back(f.get());

            // lock all user defined data elements, will be unlocked at function exit
            scoped_values_lock<lcos::local::mutex> l(val);

            val[3].get() = val[0].get();

            // ghostwidth interpolation
            // left end
            for (std::size_t i=0;i<par->ghostwidth;i++) {
              double x = val[3]->value_[i].x;

              double minx = val[1]->value_[0].x;
              double h = val[1]->value_[1].x-val[1]->value_[0].x;

              interpolate(x,minx,h,
                          val[1],val[3]->value_[i],par);
            }
            // right end
            for (std::size_t i=val[3]->value_.size()-par->ghostwidth;i<val[3]->value_.size();i++) {
              double x = val[3]->value_[i].x;

              double minx = val[2]->value_[0].x;
              double h = val[2]->value_[1].x-val[2]->value_[0].x;

              interpolate(x,minx,h,
                          val[2],val[3]->value_[i],par);
            }
            if (val[3]->timestep_ >= par->nt0-1) {
              return 0;
            }
            return 1;
          }
          // }}}

        }
        BOOST_ASSERT(false);
        return 0;
    }

    ///////////////////////////////////////////////////////////////////////////
    naming::id_type stencil::alloc_data(int item, int maxitems, int row,
                           std::vector<naming::id_type> const& interp_src_data,
                           double time,
                           parameter const& par)
    {
        naming::id_type here = applier::get_applier().get_runtime_support_gid();
        naming::id_type result = components::stubs::memory_block::create(
            here, sizeof(stencil_data), manage_stencil_data);

        if (-1 != item) {
            // provide initial data for the given data value
            access_memory_block<stencil_data> val(
                components::stubs::memory_block::checkout(result));

            if ( time < 1.e-8 ) {
              // call provided (external) function
              generate_initial_data(val.get_ptr(), item, maxitems, row, *par.p);
            } else {
              // use the previous data (or interpolate if necessary)
              access_memory_block<stencil_data> prev_val(
                      components::stubs::memory_block::checkout(interp_src_data[item]));

              val.get() = prev_val.get();
              val->max_index_ = maxitems;
              val->index_ = item;
              val->timestep_ = 0;
            }

            //if (log_ && par->loglevel > 1)         // send initial value to logging instance
            //    stubs::logging::logentry(log_, val.get(), row,item, par);
        }
        return result;
    }

    void stencil::init(std::size_t numsteps, naming::id_type const& logging)
    {
        numsteps_ = numsteps;
        log_ = logging;
    }
}}}

