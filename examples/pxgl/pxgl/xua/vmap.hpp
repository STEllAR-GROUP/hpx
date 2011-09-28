// Copyright (c) 2010-2011 Dylan Stark
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying 
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(PXGL_XUA_VMAP_20110217T0827)
#define PXGL_XUA_VMAP_20110217T0827

#include <algorithm>
#include <boost/unordered_map.hpp>

// Bring in PXGL headers
#include "../../pxgl/pxgl.hpp"

#include "../../pxgl/xua/range.hpp"
#include "../../pxgl/xua/arbitrary_distribution.hpp"
#include "../../pxgl/xua/vector.hpp"

#include "../../pxgl/graphs/edge_tuple.hpp"
#include "../../pxgl/graphs/extension_info.hpp"
#include "../../pxgl/graphs/csr_graph.hpp"
#include "../../pxgl/graphs/dynamic_graph_client.hpp"

#include "../../pxgl/util/component.hpp"

////////////////////////////////////////////////////////////////////////////////
namespace pxgl {
  namespace xua {
    class vmap;
  }
}

////////////////////////////////////////////////////////////////////////////////
namespace pxgl { namespace xua { namespace server {

  class vmap
    : public HPX_MANAGED_BASE_0(vmap)
  {
  public:
    enum actions
    {
      // Construction/initialization
      vmap_construct,
      vmap_replistruct,
      vmap_request_extension,
      vmap_constructed,
      vmap_visit,
      vmap_visits,
      vmap_init,
      vmap_signal_init,
      vmap_finalize_init,
      vmap_ready,
      // Use
      vmap_get_distribution,
      vmap_local_to,
      vmap_size,
    };

    ////////////////////////////////////////////////////////////////////////////
    typedef hpx::naming::id_type id_type;
    typedef std::vector<id_type> ids_type;

    typedef hpx::naming::gid_type gid_type;
    typedef std::vector<hpx::naming::gid_type> gids_type;

    typedef unsigned long size_type;
    typedef std::vector<size_type> sizes_type;

    typedef hpx::lcos::promise<size_type> future_size_type;
    typedef std::vector<future_size_type> future_sizes_type;
     
    typedef pxgl::graphs::server::edge_tuple_type edge_tuple_type;
    typedef std::vector<edge_tuple_type> edge_tuples_type;

    typedef pxgl::graphs::server::extension_info_type extension_info_type;

    typedef pxgl::xua::arbitrary_distribution<id_type, pxgl::xua::range>
            arbitrary_distribution_type;
    typedef arbitrary_distribution_type distribution_type;

    typedef pxgl::xua::vector<
        arbitrary_distribution_type,
        pxgl::graphs::server::edge_tuple_type
    > edge_container_type;
    
    typedef pxgl::graphs::csr_graph<
        edge_container_type,
        arbitrary_distribution_type
    > graph_type;
    
    typedef pxgl::graphs::dynamic_graph subgraph_type;

    typedef pxgl::xua::vmap vmap_type;

    typedef boost::unordered_map<size_type, sizes_type> target_map_type;

    typedef boost::unordered_map<size_type, size_type> color_map_type;

    ////////////////////////////////////////////////////////////////////////////
    // Construction/initialization
    vmap();
    ~vmap();

    void construct(
        id_type const &, 
        distribution_type const &,
        id_type const &,
        id_type const &);
    typedef hpx::actions::action4<
        vmap, 
        vmap_construct,
            id_type const &, 
            distribution_type const &,
            id_type const &, 
            id_type const &, 
        &vmap::construct
    > construct_action;

    void replistruct(
        id_type const &, 
        distribution_type const &, 
        ids_type const &,
        id_type const &,
        id_type const &);
    typedef hpx::actions::action5<
        vmap, 
        vmap_replistruct,
            id_type const &, 
            distribution_type const &, 
            ids_type const &,
            id_type const &, 
            id_type const &, 
        &vmap::replistruct
    > replistruct_action;

    extension_info_type request_extension(size_type);
    typedef hpx::actions::result_action1<
        vmap,
            extension_info_type,
        vmap_request_extension,
            size_type,
        &vmap::request_extension
    > request_extension_action;

    void constructed(void);
    typedef hpx::actions::action0<
        vmap, vmap_constructed, &vmap::constructed
    > constructed_action;

    void not_constructed(void);

    void visit(size_type, size_type);
    typedef hpx::actions::action2<
        vmap,
        vmap_visit,
            size_type,
            size_type,
        &vmap::visit
    > visit_action;

    void visits(sizes_type, size_type);
    typedef hpx::actions::action2<
        vmap,
        vmap_visits,
            sizes_type,
            size_type,
        &vmap::visits
    > visits_action;

    void init(void);
    typedef hpx::actions::action0<
        vmap, 
        vmap_init, 
        &vmap::init
    > init_action;

    size_type signal_init(void);
    typedef hpx::actions::result_action0<
        vmap, 
            size_type,
        vmap_signal_init, 
        &vmap::signal_init
    > signal_init_action;

    void finalize_init(
        distribution_type const &, 
        ids_type const &,
        size_type);
    typedef hpx::actions::action3<
        vmap, 
        vmap_finalize_init,
            distribution_type const &, 
            ids_type const &,
            size_type,
        &vmap::finalize_init
    > finalize_init_action;

    void ready(void);
    typedef hpx::actions::action0<
        vmap, vmap_ready, &vmap::ready
    > ready_action;

    void not_ready(void);

    ////////////////////////////////////////////////////////////////////////////
    // Use interface

    distribution_type get_distribution(void);
    typedef hpx::actions::result_action0<
        vmap,
            distribution_type,
        vmap_get_distribution,
        &vmap::get_distribution
    > get_distribution_action;

    id_type local_to(size_type);
    typedef hpx::actions::result_action1<
        vmap,
            id_type,
        vmap_local_to,
            size_type,
        &vmap::local_to
    > local_to_action;

    size_type size(void);
    typedef hpx::actions::result_action0<
        vmap, 
            size_type, 
        vmap_size, 
        &vmap::size
    > size_action;

    ////////////////////////////////////////////////////////////////////////////
  private:
    size_type size_;

    id_type me_;
    id_type here_;

    distribution_type distribution_;

    std::vector<vmap_type> siblings_;

    target_map_type map_;

    color_map_type seen_;

    graph_type graph_;
    subgraph_type subgraph_;

    ////////////////////////////////////////////////////////////////////////////
    // Synchronization members
    struct tag {};
    typedef hpx::util::spinlock_pool<tag> mutex_type;
    typedef mutex_type::scoped_lock scoped_lock;

    typedef int result_type;
    typedef boost::exception_ptr error_type;
    typedef boost::variant<result_type, error_type> feb_data_type;

    // Used to suspend calling threads until data structure is constructed
    // Note: this is required because we cannot pass arguments to the
    // component constructor
    bool constructed_;
    hpx::util::full_empty<feb_data_type> constructed_feb_;

    // Used to suspend calling threads until data structure is initialized
    bool initialized_;
    hpx::util::full_empty<feb_data_type> initialized_feb_;

    // Use to block threads around critical sections
    hpx::util::full_empty<feb_data_type> use_feb_;
  };
}}}

#endif
