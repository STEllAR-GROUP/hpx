//  Copyright (c) 2012 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_226CC70A_D749_4ADA_BB55_70F85566B5CC)
#define HPX_226CC70A_D749_4ADA_BB55_70F85566B5CC

#include <vector>
#include <queue>

#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/util/unlock_guard.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace ag { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    class HPX_COMPONENT_EXPORT allgather
      : public hpx::components::managed_component_base<allgather>
    {
    public:
        allgather()
        {}

        ///////////////////////////////////////////////////////////////////////
        // Exposed functionality of this component.

        void init(std::size_t item,std::size_t np);
        void compute(std::vector<hpx::naming::id_type> const& point_components);
        double get_item() const;
        void print();

        // Each of the exposed functions needs to be encapsulated into an
        // action type, generating all required boilerplate code for threads,
        // serialization, etc.

        HPX_DEFINE_COMPONENT_ACTION(allgather, init, init_action);
        HPX_DEFINE_COMPONENT_ACTION(allgather, compute, compute_action);
        HPX_DEFINE_COMPONENT_ACTION(allgather, get_item, get_item_action);
        HPX_DEFINE_COMPONENT_ACTION(allgather, print, print_action);

    private:
        //hpx::lcos::local::mutex mtx_;
        hpx::util::spinlock mtx_;
        double value_;
        std::size_t item_;
        std::vector<double> n_;
    };
}}

// Declaration of serialization support for the actions
HPX_REGISTER_ACTION_DECLARATION(
    ag::server::allgather::init_action,
    allgather_init_action);

HPX_REGISTER_ACTION_DECLARATION(
    ag::server::allgather::compute_action,
    allgather_compute_action);

HPX_REGISTER_ACTION_DECLARATION(
    ag::server::allgather::get_item_action,
    allgather_get_item_action);

HPX_REGISTER_ACTION_DECLARATION(
    ag::server::allgather::print_action,
    allgather_print_action);

#endif

