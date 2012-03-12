//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_226CC70A_D748_4ADA_BB55_70F85566B5CC)
#define HPX_226CC70A_D748_4ADA_BB55_70F85566B5CC

#include <vector>
#include <queue>

#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/util/unlock_lock.hpp>
#include "graph_generator.hpp"
#include "splittable_mrg.hpp"
#include "../../array.hpp"

struct leveldata
{
  std::size_t level;
  int64_t parent;

  leveldata() {}

  private:
  // serialization support
  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    ar & level & parent;
  }
};

struct resolvedata
{
  std::vector<std::size_t> level;
  std::vector<int64_t> parent;
  int64_t edge;

  resolvedata() {}

  private:
  // serialization support
  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    ar & level & parent & edge;
  }
};


///////////////////////////////////////////////////////////////////////////////
namespace graph500 { namespace server
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

        void init(std::size_t objectid,std::size_t scale,std::size_t number_partitions,double overlap);

        void root(std::vector<int64_t> const& bfs_roots);

        void bfs();

        void resolve_conflict();

        std::vector<int> distributed_validate(std::size_t scale);

        bool get_numedges_callback(std::size_t i,resolvedata r);

        std::vector<int64_t> get_numedges();
  
        void ppedge(int64_t start, int64_t stop, std::vector<hpx::naming::id_type> const& point_components);

        resolvedata get_parent(int64_t edge);

        bool has_edge(int64_t edge);

        bool resolve_conflict_callback(std::size_t i,resolvedata r);

        void receive_duplicates(int64_t j,                
                  std::vector<hpx::naming::id_type> const& duplicate_components,
                  std::vector<std::size_t> const& duplicateid);

        // Each of the exposed functions needs to be encapsulated into an
        // action type, generating all required boilerplate code for threads,
        // serialization, etc.

        /// Action codes.
        enum actions
        {
            point_init = 0,
            point_bfs = 1,
            point_has_edge = 2,
            point_root = 3,
            point_receive_duplicates = 4,
            point_resolve_conflict = 5,
            point_get_parent = 6,
            point_distributed_validate = 7,
            point_get_numedges = 8,
            point_ppedge = 9
        };

        typedef hpx::actions::action4<
            // Component server type.
            point,
            // Action code.
            point_init,
            // Arguments of this action.
            std::size_t,
            std::size_t,
            std::size_t,
            double,
            // Method bound to this action.
            &point::init
        > init_action;

        typedef hpx::actions::action1<
            // Component server type.
            point,
            // Action code.
            point_root,
            // Arguments of this action.
            std::vector<int64_t> const&,
            // Method bound to this action.
            &point::root
        > root_action;

        typedef hpx::actions::action3<
            // Component server type.
            point,
            // Action code.
            point_receive_duplicates,
            // Arguments of this action.
            int64_t,
            std::vector<hpx::naming::id_type> const&,
            std::vector<std::size_t> const&,
            // Method bound to this action.
            &point::receive_duplicates
        > receive_duplicates_action;

        typedef hpx::actions::action3<
            // Component server type.
            point,
            // Action code.
            point_ppedge,
            // Arguments of this action.
            int64_t,
            int64_t,
            std::vector<hpx::naming::id_type> const&,
            // Method bound to this action.
            &point::ppedge
        > ppedge_action;

        typedef hpx::actions::action0<
            // Component server type.
            point,
            // Action code.
            point_bfs,
            // Arguments of this action.
            // Method bound to this action.
            &point::bfs
        > bfs_action;

        typedef hpx::actions::action0<
            // Component server type.
            point,
            // Action code.
            point_resolve_conflict,
            // Arguments of this action.
            // Method bound to this action.
            &point::resolve_conflict
        > resolve_conflict_action;

        typedef hpx::actions::result_action1<
            // Component server type.
            point,
            // Return type.
            std::vector<int>,
            // Action code.
            point_distributed_validate,
            // Arguments of this action.
            std::size_t,
            // Method bound to this action.
            &point::distributed_validate
        > distributed_validate_action;

        typedef hpx::actions::result_action0<
            // Component server type.
            point,
            // Return type.
            std::vector<int64_t>,
            // Action code.
            point_get_numedges,
            // Arguments of this action.
            // Method bound to this action.
            &point::get_numedges
        > get_numedges_action;

        typedef hpx::actions::result_action1<
            // Component server type.
            point,
            // Return type.
            resolvedata, 
            // Action code.
            point_get_parent,
            // Arguments of this action.
            int64_t,
            // Method bound to this action.
            &point::get_parent
        > get_parent_action;

        typedef hpx::actions::result_action1<
            // Component server type.
            point,
            // Return type.
            bool,
            // Action code.
            point_has_edge,
            // Arguments of this action.
            int64_t,
            // Method bound to this action.
            &point::has_edge
        > has_edge_action;

    private:
        hpx::lcos::local::mutex mtx_;
        std::size_t idx_;
        int64_t N_;
        std::vector< std::vector<int64_t> > neighbors_;
        bfsg::array<leveldata> parent_;
        std::vector< std::vector<hpx::naming::id_type> > duplicates_;
        std::vector< std::vector<std::size_t> > duplicatesid_;
        int64_t minnode_;
        std::vector<packed_edge> local_edges_;
        std::vector<int64_t> nedge_bins_;
        std::vector<int64_t> bfs_roots_;
    };
}}

// Declaration of serialization support for the actions
HPX_REGISTER_ACTION_DECLARATION_EX(
    graph500::server::point::init_action,
    graph500_point_init_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    graph500::server::point::root_action,
    graph500_point_root_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    graph500::server::point::ppedge_action,
    graph500_point_ppedge_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    graph500::server::point::receive_duplicates_action,
    graph500_point_receive_duplicates_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    graph500::server::point::get_parent_action,
    graph500_point_get_parent_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    graph500::server::point::bfs_action,
    graph500_point_bfs_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    graph500::server::point::resolve_conflict_action,
    graph500_point_resolve_conflict_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    graph500::server::point::get_numedges_action,
    graph500_point_get_numedges_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    graph500::server::point::distributed_validate_action,
    graph500_point_distributed_validate_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    graph500::server::point::has_edge_action,
    graph500_point_has_edge_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco_with_value<std::vector<std::size_t> >::get_value_action,
    get_value_action_vector_size_t);

HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco_with_value<std::vector<std::size_t>>::set_result_action,
    set_result_action_vector_size_t);

#endif

