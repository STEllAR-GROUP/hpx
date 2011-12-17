//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_236CC70A_D748_4ADA_BB55_70F85566B5CC)
#define HPX_236CC70A_D748_4ADA_BB55_70F85566B5CC

#include <vector>

#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <boost/numeric/ublas/vector_sparse.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace bfs_tm { namespace server
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

        void manager(std::size_t level,std::size_t edge,std::vector<std::size_t> const& neighbors);

        void init(std::size_t objectid,
                 boost::numeric::ublas::mapped_vector<std::size_t> const& index,
                 std::vector<hpx::naming::id_type> const& points_components);

        /// Action codes.
        enum actions
        {
            point_manager = 0,
            point_init = 1
        };

        typedef hpx::actions::action3<
            // Component server type.
            point,
            // Action code.
            point_manager,
            // Arguments of this action.
            std::size_t,
            std::size_t,
            std::vector<std::size_t> const&,
            // Method bound to this action.
            &point::manager
        > manager_action;

        typedef hpx::actions::action3<
            // Component server type.
            point,
            // Action code.
            point_init,
            // Arguments of this action.
            std::size_t,
            boost::numeric::ublas::mapped_vector<std::size_t> const&,
            std::vector<hpx::naming::id_type> const&,
            // Method bound to this action.
            &point::init
        > init_action;

    private:
        std::size_t idx_;
        boost::numeric::ublas::mapped_vector<std::size_t> index_;
        std::vector<hpx::naming::id_type> points_components_;
    };
}}

// Declaration of serialization support for the actions
HPX_REGISTER_ACTION_DECLARATION_EX(
    bfs_tm::server::point::manager_action,
    bfs_tm_point_manager_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    bfs_tm::server::point::init_action,
    bfs_tm_point_init_action);

#endif

