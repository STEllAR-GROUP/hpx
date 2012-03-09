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
#include "../mesh/unigrid_mesh.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr
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

    stencil_config_data::stencil_config_data(int start, int count)
    {
        // create new instance
        static_cast<base_type&>(*this) = create_and_resolve_target();

        // initialize from arguments
        (*this)->start_ = start;
        (*this)->count_ = count;
    }
}}}

///////////////////////////////////////////////////////////////////////////////
HPX_REGISTER_MANAGE_OBJECT_ACTION(
    hpx::components::amr::manage_object_config_data_action,
    dataflow_manage_object_action_stencil_config_data)

HPX_REGISTER_MANAGE_OBJECT_ACTION(
    hpx::components::amr::manage_object_data_action,
    dataflow_manage_object_action_stencil_data)

// HPX_REGISTER_MANAGE_OBJECT_ACTION(
//     hpx::components::amr::manage_object_data_action_simple,
//     dataflow_manage_object_action_stencil_data_simple)

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr
{
    ///////////////////////////////////////////////////////////////////////////
    stencil::stencil()
      : numsteps_(0)
    {
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

        // Generate new config info
        stencil_config_data cfg_right(0, 2);
        stencil_config_data cfg_left(par->grain_size-3, 2);

        // get all input memory_block_data instances
        typedef std::vector<lcos::future<memory_block_data> >
            lazy_results_type;

        // first invoke all remote operations
        lazy_results_type lazy_results;

        namespace s = hpx::components::stubs;
        lazy_results.push_back(
            s::memory_block::get_async(gids[0], cfg_left.get_memory_block()));
        lazy_results.push_back(s::memory_block::get_async(gids[1]));
        lazy_results.push_back(
            s::memory_block::get_async(gids[2], cfg_right.get_memory_block()));

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
        val[3]->value_ = val[1]->value_;

        //resultval->value_.resize(val[1]->value_.size());
        //for (std::size_t i=0;i<val[1]->value_.size();i++) {
        //  resultval->value_[i] = val[1]->value_[i];
        //}

        val[3]->timestep_ = val[1]->timestep_ + 1.0;

        std::cout << " row " << row << " column " << column
            << " timestep " << val[3]->timestep_
            << " left " << val[0]->value_.size() << "(" << val[0]->value_.data_size() << ")"
            << " middle " << val[1]->value_.size() << "(" << val[1]->value_.data_size() << ")"
            << " right " << val[2]->value_.size() << "(" << val[2]->value_.data_size() << ")"
            << std::endl;

        //std::cout << " row " << row << " column " << column << " timestep " << resultval->timestep_ << " size " << val.size() << std::endl;
        if (val[3]->timestep_ >= par->nt0-1) {
          return 0;
        }
        return 1;
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

            // call provided (external) function
            generate_initial_data(val.get_ptr(), item, maxitems, row, *par.p);

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

