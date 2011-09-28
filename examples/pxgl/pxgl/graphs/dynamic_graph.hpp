// Copyright (c) 2010-2011 Dylan Stark
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying 
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(PXGL_GRAPHS_DYNAMIC_GRAPH_20110217T0827)
#define PXGL_GRAPHS_DYNAMIC_GRAPH_20110217T0827

#include <algorithm>
#include <boost/unordered_map.hpp>

// Bring in PXGL headers
#include "../../pxgl/pxgl.hpp"

#include "../../pxgl/xua/range.hpp"
#include "../../pxgl/xua/arbitrary_distribution.hpp"

#include "../../pxgl/graphs/signal_value.hpp"
#include "../../pxgl/graphs/extension_info.hpp"
#include "../../pxgl/graphs/edge_tuple.hpp"

#include "../../pxgl/util/component.hpp"

////////////////////////////////////////////////////////////////////////////////
namespace pxgl {
  namespace graphs {
    class dynamic_graph;
  }
}

////////////////////////////////////////////////////////////////////////////////
namespace pxgl { namespace graphs { namespace server {

  class dynamic_graph
    : public HPX_MANAGED_BASE_0(dynamic_graph)
  {
  public:
    enum actions
    {
      // Construction/initialization
      dynamic_graph_set_distribution,
      dynamic_graph_replistruct,
      dynamic_graph_request_extension,
      dynamic_graph_constructed,
      dynamic_graph_init,
      dynamic_graph_signal_init,
      dynamic_graph_finalize_init,
      dynamic_graph_ready,
      // Use
      dynamic_graph_add_edge,
      dynamic_graph_add_edges,
      dynamic_graph_add_vertex,
      dynamic_graph_order,
      dynamic_graph_size,
    };

    ////////////////////////////////////////////////////////////////////////////
    typedef hpx::naming::id_type id_type;
    typedef std::vector<id_type> ids_type;

    typedef unsigned long size_type;
    typedef std::vector<size_type> sizes_type;

    typedef hpx::lcos::promise<signal_value_type> future_signal_value_type;
    typedef std::vector<future_signal_value_type> future_signal_values_type;
     
    typedef pxgl::graphs::server::edge_tuple_type edge_tuple_type;
    typedef std::vector<edge_tuple_type> edge_tuples_type;

    typedef pxgl::xua::arbitrary_distribution<id_type, pxgl::xua::range>
            arbitrary_distribution_type;
    typedef arbitrary_distribution_type distribution_type;

    typedef pxgl::graphs::dynamic_graph dynamic_graph_type;

    typedef boost::unordered_map<size_type, sizes_type> target_map_type;

    ////////////////////////////////////////////////////////////////////////////
    // Construction/initialization
    dynamic_graph();
    ~dynamic_graph();

    void set_distribution(
        id_type const &, 
        distribution_type const &);
    typedef hpx::actions::action2<
        dynamic_graph, 
        dynamic_graph_set_distribution,
            id_type const &, 
            distribution_type const &,
        &dynamic_graph::set_distribution
    > set_distribution_action;

    void replistruct(
        id_type const &, 
        distribution_type const &, 
        ids_type const &);
    typedef hpx::actions::action3<
        dynamic_graph, 
        dynamic_graph_replistruct,
            id_type const &, 
            distribution_type const &, 
            ids_type const &,
        &dynamic_graph::replistruct
    > replistruct_action;

    void request_extension(size_type v);
    typedef hpx::actions::action1<
        dynamic_graph, 
        dynamic_graph_request_extension,
            size_type,
        &dynamic_graph::request_extension
    > request_extension_action;

    void constructed(void);
    typedef hpx::actions::action0<
        dynamic_graph, dynamic_graph_constructed, &dynamic_graph::constructed
    > constructed_action;

    void not_constructed(void);

    void init(void);
    typedef hpx::actions::action0<
        dynamic_graph, 
        dynamic_graph_init, 
        &dynamic_graph::init
    > init_action;

    signal_value_type signal_init(void);
    typedef hpx::actions::result_action0<
        dynamic_graph, 
            signal_value_type,
        dynamic_graph_signal_init, 
        &dynamic_graph::signal_init
    > signal_init_action;

    void finalize_init(
        distribution_type const &, 
        ids_type const &,
        size_type,
        size_type);
    typedef hpx::actions::action4<
        dynamic_graph, 
        dynamic_graph_finalize_init,
            distribution_type const &, 
            ids_type const &,
            size_type,
            size_type,
        &dynamic_graph::finalize_init
    > finalize_init_action;

    void ready(void);
    typedef hpx::actions::action0<
        dynamic_graph, dynamic_graph_ready, &dynamic_graph::ready
    > ready_action;

    void not_ready(void);

    ////////////////////////////////////////////////////////////////////////////
    // Use interface

    size_type add_edge(edge_tuple_type const &);
    typedef hpx::actions::result_action1<
        dynamic_graph, 
            size_type, 
        dynamic_graph_add_edge, 
            edge_tuple_type const &,
        &dynamic_graph::add_edge
    > add_edge_action;

    size_type add_edges(edge_tuples_type const &);
    typedef hpx::actions::result_action1<
        dynamic_graph, 
            size_type, 
        dynamic_graph_add_edges, 
            edge_tuples_type const &,
        &dynamic_graph::add_edges
    > add_edges_action;

    void add_vertex(size_type);
    typedef hpx::actions::action1<
        dynamic_graph, 
        dynamic_graph_add_vertex, 
            size_type,
        &dynamic_graph::add_vertex
    > add_vertex_action;

    size_type order(void);
    typedef hpx::actions::result_action0<
        dynamic_graph, 
            size_type, 
        dynamic_graph_order, 
        &dynamic_graph::order
    > order_action;

    size_type size(void);
    typedef hpx::actions::result_action0<
        dynamic_graph, 
            size_type, 
        dynamic_graph_size, 
        &dynamic_graph::size
    > size_action;

    ////////////////////////////////////////////////////////////////////////////
  private:
    size_type order_;
    size_type size_;

    id_type me_;
    id_type here_;

    distribution_type distribution_;
    sizes_type coverage_map_;

    std::vector<dynamic_graph_type> siblings_;

    target_map_type map_;

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
