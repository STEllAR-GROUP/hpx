//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_226CC70A_D749_4ADA_BB55_70F85566B5CC)
#define HPX_226CC70A_D749_4ADA_BB55_70F85566B5CC

#include <vector>
#include <queue>

#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/util/unlock_lock.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace ad { namespace server
{

    ///////////////////////////////////////////////////////////////////////////
    class HPX_COMPONENT_EXPORT point
      : public hpx::components::managed_component_base<point>
    {
    public:
        point()
        {}

        ///////////////////////////////////////////////////////////////////////
        // Exposed functionality of this component.

        void init(std::size_t item,std::size_t np);
        void compute(std::vector<hpx::naming::id_type> const& point_components);
        std::size_t get_item();
        void remove_item(std::size_t replace,std::size_t substitute);

        // Each of the exposed functions needs to be encapsulated into an
        // action type, generating all required boilerplate code for threads,
        // serialization, etc.

        HPX_DEFINE_COMPONENT_ACTION(point, init);
        HPX_DEFINE_COMPONENT_ACTION(point, remove_item);
        HPX_DEFINE_COMPONENT_ACTION(point, compute);
        HPX_DEFINE_COMPONENT_ACTION(point, get_item);

    private:
        //hpx::lcos::local::mutex mtx_;
        hpx::util::spinlock mtx_;
        bool active_;
        std::size_t item_;
        std::vector<std::size_t> neighbors_;
        std::size_t sum_;
    };
}}

// Declaration of serialization support for the actions
HPX_REGISTER_ACTION_DECLARATION(
    ad::server::point::init_action,
    ad_point_init_action);

HPX_REGISTER_ACTION_DECLARATION(
    ad::server::point::compute_action,
    ad_point_compute_action);

HPX_REGISTER_ACTION_DECLARATION(
    ad::server::point::get_item_action,
    ad_point_get_item_action);

HPX_REGISTER_ACTION_DECLARATION(
    ad::server::point::remove_item_action,
    ad_point_remove_item_action);
#endif

