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

        std::vector<bool> findwhohasthisedge(int64_t edge,
           std::vector<hpx::naming::id_type> const& point_components);

        bool findwhohasthisedge_callback(int64_t j,std::vector<bool> const& has_edge,
                     std::vector<hpx::naming::id_type> const& point_components,
                     int64_t start);
  
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

        HPX_DEFINE_COMPONENT_ACTION(point, init);
        HPX_DEFINE_COMPONENT_ACTION(point, root);
        HPX_DEFINE_COMPONENT_ACTION(point, receive_duplicates);
        HPX_DEFINE_COMPONENT_ACTION(point, ppedge);
        HPX_DEFINE_COMPONENT_ACTION(point, bfs);
        HPX_DEFINE_COMPONENT_ACTION(point, resolve_conflict);
        HPX_DEFINE_COMPONENT_ACTION(point, distributed_validate);
        HPX_DEFINE_COMPONENT_ACTION(point, findwhohasthisedge);
        HPX_DEFINE_COMPONENT_ACTION(point, get_numedges);
        HPX_DEFINE_COMPONENT_ACTION(point, get_parent);
        HPX_DEFINE_COMPONENT_ACTION(point, has_edge);

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
HPX_REGISTER_ACTION_DECLARATION(
    graph500::server::point::init_action,
    graph500_point_init_action);

HPX_REGISTER_ACTION_DECLARATION(
    graph500::server::point::root_action,
    graph500_point_root_action);

HPX_REGISTER_ACTION_DECLARATION(
    graph500::server::point::ppedge_action,
    graph500_point_ppedge_action);

HPX_REGISTER_ACTION_DECLARATION(
    graph500::server::point::findwhohasthisedge_action,
    graph500_point_findwhohasthisedge_action);

HPX_REGISTER_ACTION_DECLARATION(
    graph500::server::point::receive_duplicates_action,
    graph500_point_receive_duplicates_action);

HPX_REGISTER_ACTION_DECLARATION(
    graph500::server::point::get_parent_action,
    graph500_point_get_parent_action);

HPX_REGISTER_ACTION_DECLARATION(
    graph500::server::point::bfs_action,
    graph500_point_bfs_action);

HPX_REGISTER_ACTION_DECLARATION(
    graph500::server::point::resolve_conflict_action,
    graph500_point_resolve_conflict_action);

HPX_REGISTER_ACTION_DECLARATION(
    graph500::server::point::get_numedges_action,
    graph500_point_get_numedges_action);

HPX_REGISTER_ACTION_DECLARATION(
    graph500::server::point::distributed_validate_action,
    graph500_point_distributed_validate_action);

HPX_REGISTER_ACTION_DECLARATION(
    graph500::server::point::has_edge_action,
    graph500_point_has_edge_action);

HPX_REGISTER_ACTION_DECLARATION(
    hpx::lcos::base_lco_with_value<std::vector<std::size_t> >::get_value_action,
    get_value_action_vector_size_t);

HPX_REGISTER_ACTION_DECLARATION(
    hpx::lcos::base_lco_with_value<std::vector<std::size_t> >::set_value_action,
    set_value_action_vector_size_t);

#endif

