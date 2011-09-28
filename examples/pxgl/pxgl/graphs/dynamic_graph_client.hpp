// Copyright (c) 2010-2011 Dylan Stark
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying 
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(PXGL_GRAPHS_DYNAMIC_GRAPH_CLIENT_20110217T0826)
#define PXGL_GRAPHS_DYNAMIC_GRAPH_CLIENT_20110217T0826

#include "../../pxgl/pxgl.hpp"
#include "../../pxgl/util/component.hpp"
#include "../../pxgl/util/hpx.hpp"

#include "../../pxgl/graphs/dynamic_graph.hpp"
#include "../../pxgl/graphs/extension_info.hpp"

#include "../../pxgl/xua/range.hpp"
#include "../../pxgl/xua/arbitrary_distribution.hpp"

////////////////////////////////////////////////////////////////////////////////
typedef unsigned long size_type;

typedef hpx::naming::id_type id_type;
typedef std::vector<id_type> ids_type;

typedef pxgl::xua::arbitrary_distribution<id_type, pxgl::xua::range>
        arbitrary_distribution_type;
typedef arbitrary_distribution_type distribution_type;

typedef pxgl::graphs::server::edge_tuple edge_tuple_type;
typedef std::vector<edge_tuple_type> edge_tuples_type;

typedef hpx::lcos::promise<hpx::util::unused_type> future_void_type;
typedef hpx::lcos::promise<size_type> future_size_type;

////////////////////////////////////////////////////////////////////////////////
// Stubs interface
namespace pxgl { namespace graphs { namespace stubs {
  struct dynamic_graph
    : HPX_STUBS_BASE_0(dynamic_graph)
  {
    ////////////////////////////////////////////////////////////////////////////
    typedef server::dynamic_graph server_type;
    typedef server_type::size_type size_type;
    typedef server_type::sizes_type sizes_type;
    typedef server_type::distribution_type distribution_type;
    typedef server_type::id_type id_type;
    typedef server_type::ids_type ids_type;
    typedef server_type::edge_tuple_type edge_tuple_type;
    typedef server_type::edge_tuples_type edge_tuples_type;

    typedef server::extension_info_type extension_info_type;
    typedef server::signal_value_type signal_value_type;

    typedef hpx::lcos::promise<signal_value_type> future_signal_value_type;

    ////////////////////////////////////////////////////////////////////////////
    // Construction/initialization
    static void set_distribution(
        id_type const & id, 
        id_type const & me, 
        distribution_type const & distribution)
    {
      typedef server_type::set_distribution_action action_type;
      hpx::applier::apply<action_type>(id, me, distribution);
    }

    static void replistruct(
        id_type const & id, 
        id_type const & me, 
        distribution_type const & distribution,
        ids_type const & sibling_ids)
    {
      typedef server_type::replistruct_action action_type;
      hpx::applier::apply<action_type>(
          id, me, distribution, sibling_ids);
    }

    static future_void_type eager_request_extension(
        id_type const & id, 
        size_type v)
    {
      typedef server_type::request_extension_action action_type;
      return hpx::lcos::eager_future<action_type>(id, v);
    }

    static void constructed(id_type id)
    {
      typedef server_type::constructed_action action_type;
      hpx::lcos::eager_future<action_type>(id).get();
    }

    static void init(id_type const & id)
    {
      typedef server_type::init_action action_type;
      hpx::applier::apply<action_type>(id);
    }

    static future_signal_value_type eager_signal_init(id_type const & id)
    {
      typedef server_type::signal_init_action action_type;
      return hpx::lcos::eager_future<action_type>(id);
    }

    static void finalize_init(
        id_type const & id,
        distribution_type const & distribution,
        ids_type const & sibling_ids,
        size_type order,
        size_type size)
    {
      typedef server_type::finalize_init_action action_type;
      hpx::applier::apply<action_type>(
          id, distribution, sibling_ids, order, size);
    }

    static void ready(id_type id)
    {
      typedef server_type::ready_action action_type;
      hpx::lcos::eager_future<action_type>(id).get();
    }

    ////////////////////////////////////////////////////////////////////////////
    // Use

    static future_size_type eager_add_edge(
        id_type const & id, 
        edge_tuple_type const & edge)
    {
      typedef server_type::add_edge_action action_type;
      return hpx::lcos::eager_future<action_type>(id, edge);
    }

    static future_size_type eager_add_edges(
        id_type const & id, 
        edge_tuples_type const & edge)
    {
      typedef server_type::add_edges_action action_type;
      return hpx::lcos::eager_future<action_type>(id, edge);
    }

    static future_void_type eager_add_vertex(
        id_type const & id, 
        size_type v)
    {
      typedef server_type::add_vertex_action action_type;
      return hpx::lcos::eager_future<action_type>(id, v);
    }

    static size_type order(id_type const & id)
    {
      typedef server_type::order_action action_type;
      return hpx::lcos::eager_future<action_type>(id).get();
    }
    
    static size_type size(id_type const & id)
    {
      typedef server_type::size_action action_type;
      return hpx::lcos::eager_future<action_type>(id).get();
    }
  };
}}}

////////////////////////////////////////////////////////////////////////////////
// Client interface
namespace pxgl { namespace graphs {
  class dynamic_graph
    : public HPX_CLIENT_BASE_0(dynamic_graph)
  {
  //////////////////////////////////////////////////////////////////////////////
  private:
    typedef HPX_CLIENT_BASE_0(dynamic_graph) base_type;

  public:
    dynamic_graph()
      : base_type(hpx::naming::invalid_id)
    {}

    dynamic_graph(hpx::naming::id_type id)
      : base_type(id)
    {}

    ////////////////////////////////////////////////////////////////////////////
    // Graph types
    typedef stubs::dynamic_graph stubs_type;
    typedef stubs_type::size_type size_type;
    typedef stubs_type::id_type id_type;
    typedef stubs_type::ids_type ids_type;
    typedef stubs_type::signal_value_type signal_value_type;
    typedef stubs_type::distribution_type distribution_type;
    typedef stubs_type::edge_tuple_type edge_tuple_type;
    typedef stubs_type::edge_tuples_type edge_tuples_type;
 
    ////////////////////////////////////////////////////////////////////////////
    // Construction/initialization
    void set_distribution(
        id_type const & me, 
        distribution_type const & distribution) const
    {
      BOOST_ASSERT(this->gid_);
      this->base_type::set_distribution(this->gid_, me, distribution);
    }

    void replistruct(
        id_type const & me, 
        distribution_type const & distribution, 
        ids_type const & sibling_ids) const
    {
      BOOST_ASSERT(this->gid_);
      this->base_type::replistruct(
          this->gid_, me, distribution, sibling_ids);
    }

    future_void_type eager_request_extension(size_type v) const
    {
      BOOST_ASSERT(this->gid_);
      return this->base_type::eager_request_extension(this->gid_, v);
    }

    void constructed(void)
    {
      BOOST_ASSERT(this->gid_);
      this->base_type::constructed(this->gid_);
    }

    void init(void) const
    {
      BOOST_ASSERT(this->gid_);
      this->base_type::init(this->gid_);
    }

    future_signal_value_type eager_signal_init(void) const
    {
      BOOST_ASSERT(this->gid_);
      return this->base_type::eager_signal_init(this->gid_);
    }

    void finalize_init(
        distribution_type const & distribution, 
        ids_type const & sibling_ids,
        size_type order,
        size_type size) const
    {
      BOOST_ASSERT(this->gid_);
      this->base_type::finalize_init(
          this->gid_, distribution, sibling_ids, order, size);
    }

    void ready(void) const
    {
      BOOST_ASSERT(this->gid_);
      this->base_type::ready(this->gid_);
    }

    ////////////////////////////////////////////////////////////////////////////
    // Use

    future_size_type eager_add_edge(edge_tuple_type const & edge) const
    {
      BOOST_ASSERT(this->gid_);
      return this->base_type::eager_add_edge(this->gid_, edge);
    }

    future_size_type eager_add_edges(edge_tuples_type const & edge) const
    {
      BOOST_ASSERT(this->gid_);
      return this->base_type::eager_add_edges(this->gid_, edge);
    }

    future_void_type eager_add_vertex(size_type v) const
    {
      BOOST_ASSERT(this->gid_);
      return this->base_type::eager_add_vertex(this->gid_, v);
    }

    size_type order(void) const
    {
      BOOST_ASSERT(this->gid_);
      return this->base_type::order(this->gid_);
    }

    size_type size(void) const
    {
      BOOST_ASSERT(this->gid_);
      return this->base_type::size(this->gid_);
    }
  };
}}

#endif

