//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_226CC70A_D748_4ADA_BB55_70F85566B5CC)
#define HPX_226CC70A_D748_4ADA_BB55_70F85566B5CC

#include <vector>

#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <boost/numeric/ublas/vector_sparse.hpp>

struct nodedata
{
  std::size_t neighbor;
  std::size_t parent;
  std::size_t level;

  nodedata() {}

  private:
  // serialization support
  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    ar & neighbor & parent & level;
  }
};


///////////////////////////////////////////////////////////////////////////////
namespace bfs { namespace server
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

        // kernel 1
        void init(std::size_t objectid,std::size_t grainsize,
        std::size_t max_num_neighbors,std::vector<std::size_t> const& nodelist,
        std::vector<std::size_t> const& neighborlist,
        boost::numeric::ublas::mapped_vector<std::size_t> const& index,std::size_t max_levels);

        /// Traverse the graph.
        std::vector<std::size_t> traverse(std::size_t level, std::size_t parent,std::size_t edge);
        std::vector<nodedata> depth_traverse(std::size_t level,std::size_t parent,std::size_t edge);
        std::vector<nodedata> unlocked_depth_traverse(std::size_t level,std::size_t parent,std::size_t edge);


        std::size_t get_parent(std::size_t edge);
        std::size_t get_level(std::size_t edge);
        void reset_visited(std::size_t objectid);

        // Each of the exposed functions needs to be encapsulated into an
        // action type, generating all required boilerplate code for threads,
        // serialization, etc.

        /// Action codes.
        enum actions
        {
            point_init = 0,
            point_traverse = 1,
            point_get_parent = 2,
            point_get_level = 3,
            point_reset_visited = 4,
            point_depth_traverse = 5
        };

        typedef hpx::actions::action7<
            // Component server type.
            point,
            // Action code.
            point_init,
            // Arguments of this action.
            std::size_t,
            std::size_t,
            std::size_t,
            std::vector<std::size_t> const&,
            std::vector<std::size_t> const&,
            boost::numeric::ublas::mapped_vector<std::size_t> const&,
            std::size_t,
            // Method bound to this action.
            &point::init
        > init_action;

        typedef hpx::actions::action1<
            // Component server type.
            point,
            // Action code.
            point_reset_visited,
            // Arguments of this action.
            std::size_t,
            // Method bound to this action.
            &point::reset_visited
        > reset_visited_action;

        typedef hpx::actions::result_action3<
            // Component server type.
            point,
            // Return type.
            std::vector<std::size_t>,
            // Action code.
            point_traverse,
            // Arguments of this action.
            std::size_t,
            std::size_t,
            std::size_t,
            // Method bound to this action.
            &point::traverse
        > traverse_action;

        typedef hpx::actions::result_action3<
            // Component server type.
            point,
            // Return type.
            std::vector<nodedata>,
            // Action code.
            point_depth_traverse,
            // Arguments of this action.
            std::size_t,
            std::size_t,
            std::size_t,
            // Method bound to this action.
            &point::depth_traverse
        > depth_traverse_action;

        typedef hpx::actions::result_action1<
            // Component server type.
            point,
            // Return type.
            std::size_t,
            // Action code.
            point_get_parent,
            // Arguments of this action.
            std::size_t,
            // Method bound to this action.
            &point::get_parent
        > get_parent_action;

        typedef hpx::actions::result_action1<
            // Component server type.
            point,
            // Return type.
            std::size_t,
            // Action code.
            point_get_level,
            // Arguments of this action.
            std::size_t,
            // Method bound to this action.
            &point::get_level
        > get_level_action;

    private:
        hpx::lcos::local::mutex mtx_;
        std::size_t idx_;
        std::vector<std::size_t> level_;
        std::vector<bool> visited_;
        std::vector< std::vector<std::size_t> > neighbors_;
        std::vector<std::size_t> parent_;
        std::size_t grainsize_;
        std::size_t max_levels_;
        boost::numeric::ublas::mapped_vector<std::size_t> mapping_;
        boost::numeric::ublas::mapped_vector<std::size_t> index_;
    };
}}

// Declaration of serialization support for the actions
HPX_REGISTER_ACTION_DECLARATION(
    bfs::server::point::init_action,
    bfs_point_init_action);

HPX_REGISTER_ACTION_DECLARATION(
    bfs::server::point::traverse_action,
    bfs_point_traverse_action);

HPX_REGISTER_ACTION_DECLARATION(
    bfs::server::point::depth_traverse_action,
    bfs_point_depth_traverse_action);

HPX_REGISTER_ACTION_DECLARATION(
    bfs::server::point::get_parent_action,
    bfs_point_get_parent_action);

HPX_REGISTER_ACTION_DECLARATION(
    bfs::server::point::get_level_action,
    bfs_point_get_level_action);

HPX_REGISTER_ACTION_DECLARATION(
    bfs::server::point::reset_visited_action,
    bfs_point_reset_visited_action);

HPX_REGISTER_ACTION_DECLARATION(
    hpx::lcos::base_lco_with_value<std::vector<std::size_t> >::get_value_action,
    get_value_action_vector_size_t);

HPX_REGISTER_ACTION_DECLARATION(
    hpx::lcos::base_lco_with_value<std::vector<std::size_t> >::set_value_action,
    set_value_action_vector_size_t);

#endif

