//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_A07C7784_8AD2_4A12_B5BA_174DFBE03222)
#define HPX_A07C7784_8AD2_4A12_B5BA_174DFBE03222

#include <vector>
#include <queue>

#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/util/unlock_lock.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace gtc { namespace server
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

        void setup(std::size_t numberpe,std::size_t mype,
                   std::vector<hpx::naming::id_type> const& point_components);
        void chargei();

        // Each of the exposed functions needs to be encapsulated into an
        // action type, generating all required boilerplate code for threads,
        // serialization, etc.

        /// Action codes.
        enum actions
        {
            point_setup = 0,
            point_chargei = 1
        };

        typedef hpx::actions::action3<
            // Component server type.
            point,
            // Action code.
            point_setup,
            // Arguments of this action.
            std::size_t,
            std::size_t,
            std::vector<hpx::naming::id_type> const&,
            // Method bound to this action.
            &point::setup
        > setup_action;

        typedef hpx::actions::action0<
            // Component server type.
            point,
            // Action code.
            point_chargei,
            // Arguments of this action.
            // Method bound to this action.
            &point::chargei
        > chargei_action;

    private:
        hpx::lcos::local::mutex mtx_;
        std::size_t item_;
    };
}}

// Declaration of serialization support for the actions
HPX_REGISTER_ACTION_DECLARATION_EX(
    gtc::server::point::setup_action,
    gtc_point_setup_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    gtc::server::point::chargei_action,
    gtc_point_chargei_action);
#endif

