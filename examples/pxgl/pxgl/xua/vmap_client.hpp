// Copyright (c) 2010-2011 Dylan Stark
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying 
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(PXGL_XUA_VMAP_CLIENT_20110217T0826)
#define PXGL_XUA_VMAP_CLIENT_20110217T0826

#include "../../pxgl/pxgl.hpp"
#include "../../pxgl/util/component.hpp"
#include "../../pxgl/util/hpx.hpp"

#include "../../pxgl/xua/vmap.hpp"

#include "../../pxgl/graphs/extension_info.hpp"

#include "../../pxgl/xua/range.hpp"
#include "../../pxgl/xua/arbitrary_distribution.hpp"

////////////////////////////////////////////////////////////////////////////////
typedef unsigned long size_type;

typedef hpx::naming::id_type id_type;
typedef std::vector<id_type> ids_type;

typedef hpx::naming::gid_type gid_type;
typedef std::vector<hpx::naming::gid_type> gids_type;

typedef pxgl::xua::arbitrary_distribution<id_type, pxgl::xua::range>
        arbitrary_distribution_type;
typedef arbitrary_distribution_type distribution_type;

typedef pxgl::graphs::server::edge_tuple edge_tuple_type;
typedef std::vector<edge_tuple_type> edge_tuples_type;

typedef pxgl::graphs::server::extension_info_type extension_info_type;

typedef hpx::lcos::promise<hpx::util::unused_type> future_void_type;
typedef hpx::lcos::promise<size_type> future_size_type;
typedef hpx::lcos::promise<extension_info_type> future_extension_info_type;

////////////////////////////////////////////////////////////////////////////////
// Stubs interface
namespace pxgl { namespace xua { namespace stubs {
  struct vmap
    : HPX_STUBS_BASE_0(vmap)
  {
    ////////////////////////////////////////////////////////////////////////////
    typedef server::vmap server_type;
    typedef server_type::size_type size_type;
    typedef server_type::sizes_type sizes_type;
    typedef server_type::distribution_type distribution_type;
    typedef server_type::id_type id_type;
    typedef server_type::ids_type ids_type;
    typedef server_type::edge_tuple_type edge_tuple_type;
    typedef server_type::edge_tuples_type edge_tuples_type;

    typedef pxgl::graphs::server::extension_info_type extension_info_type;

    typedef hpx::lcos::promise<size_type> future_size_type;

    ////////////////////////////////////////////////////////////////////////////
    // Construction/initialization
    static void construct(
        id_type const & id, 
        id_type const & me, 
        distribution_type const & distribution,
        id_type const & graph,
        id_type const & subgraph)
    {
      typedef server_type::construct_action action_type;
      hpx::applier::apply<action_type>(id, me, distribution, graph, subgraph);
    }

    static void replistruct(
        id_type const & id, 
        id_type const & me, 
        distribution_type const & distribution,
        ids_type const & sibling_ids,
        id_type const & graph,
        id_type const & subgraph)
    {
      typedef server_type::replistruct_action action_type;
      hpx::applier::apply<action_type>(
          id, me, distribution, sibling_ids, graph, subgraph);
    }

    static future_extension_info_type eager_request_extension(
        id_type const & id, 
        size_type k)
    {
      typedef server_type::request_extension_action action_type;
      return hpx::lcos::eager_future<action_type>(id, k);
    }

    static void constructed(id_type id)
    {
      typedef server_type::constructed_action action_type;
      hpx::lcos::eager_future<action_type>(id).get();
    }

    static future_void_type eager_visit(
        id_type const & id,
        size_type v,
        size_type depth)
    {
      typedef server_type::visit_action action_type;
      return hpx::lcos::eager_future<action_type>(id, v, depth);
    }

    static future_void_type eager_visits(
        id_type const & id,
        sizes_type vs,
        size_type depth)
    {
      typedef server_type::visits_action action_type;
      return hpx::lcos::eager_future<action_type>(id, vs, depth);
    }

    static void sync_visit(
        id_type const & id,
        size_type v,
        size_type depth)
    {
      typedef server_type::visit_action action_type;
      hpx::lcos::eager_future<action_type>(id, v, depth).get();
    }

    static void sync_visits(
        id_type const & id,
        sizes_type vs,
        size_type depth)
    {
      typedef server_type::visits_action action_type;
      hpx::lcos::eager_future<action_type>(id, vs, depth).get();
    }

    static void init(id_type const & id)
    {
      typedef server_type::init_action action_type;
      hpx::applier::apply<action_type>(id);
    }

    static future_size_type eager_signal_init(id_type const & id)
    {
      typedef server_type::signal_init_action action_type;
      return hpx::lcos::eager_future<action_type>(id);
    }

    static void finalize_init(
        id_type const & id,
        distribution_type const & distribution,
        ids_type const & sibling_ids,
        size_type size)
    {
      typedef server_type::finalize_init_action action_type;
      hpx::applier::apply<action_type>(
          id, distribution, sibling_ids, size);
    }

    static void ready(id_type id)
    {
      typedef server_type::ready_action action_type;
      hpx::lcos::eager_future<action_type>(id).get();
    }

    ////////////////////////////////////////////////////////////////////////////
    // Use

    static distribution_type get_distribution(id_type const & id)
    {
      typedef server_type::get_distribution_action action_type;
      return hpx::lcos::eager_future<action_type>(id).get();
    }

    static id_type local_to(id_type const & id, size_type index)
    {
      typedef server_type::local_to_action action_type;
      return hpx::lcos::eager_future<action_type>(id, index).get();
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
namespace pxgl { namespace xua {
  class vmap
    : public HPX_CLIENT_BASE_0(vmap)
  {
  //////////////////////////////////////////////////////////////////////////////
  private:
    typedef HPX_CLIENT_BASE_0(vmap) base_type;

  public:
    vmap()
      : base_type(hpx::naming::invalid_id)
    {}

    vmap(hpx::naming::id_type id)
      : base_type(id)
    {}

    ////////////////////////////////////////////////////////////////////////////
    // Graph types
    typedef stubs::vmap stubs_type;
    typedef stubs_type::size_type size_type;
    typedef stubs_type::id_type id_type;
    typedef stubs_type::ids_type ids_type;
    typedef stubs_type::distribution_type distribution_type;
    typedef stubs_type::edge_tuple_type edge_tuple_type;
    typedef stubs_type::edge_tuples_type edge_tuples_type;
 
    ////////////////////////////////////////////////////////////////////////////
    // Construction/initialization
    void construct(
        id_type const & me, 
        distribution_type const & distribution,
        id_type const & graph,
        id_type const & subgraph) const
    {
      BOOST_ASSERT(this->gid_);
      this->base_type::construct(this->gid_, me, distribution, graph, subgraph);
    }

    void replistruct(
        id_type const & me, 
        distribution_type const & distribution, 
        ids_type const & sibling_ids,
        id_type const & graph,
        id_type const & subgraph) const
    {
      BOOST_ASSERT(this->gid_);
      this->base_type::replistruct(this->gid_, 
          me, distribution, sibling_ids, graph, subgraph);
    }

    future_extension_info_type eager_request_extension(
        size_type v) const
    {
      BOOST_ASSERT(this->gid_);
      return this->base_type::eager_request_extension(this->gid_, v);
    }

    void constructed(void)
    {
      BOOST_ASSERT(this->gid_);
      this->base_type::constructed(this->gid_);
    }

    future_void_type eager_visit(size_type v, size_type depth)
    {
      BOOST_ASSERT(this->gid_);
      return this->base_type::eager_visit(this->gid_, v, depth);
    }

    future_void_type eager_visits(sizes_type vs, size_type depth)
    {
      BOOST_ASSERT(this->gid_);
      return this->base_type::eager_visits(this->gid_, vs, depth);
    }

    void sync_visit(size_type v, size_type depth)
    {
      BOOST_ASSERT(this->gid_);
      this->base_type::sync_visit(this->gid_, v, depth);
    }

    void sync_visits(sizes_type vs, size_type depth)
    {
      BOOST_ASSERT(this->gid_);
      this->base_type::sync_visits(this->gid_, vs, depth);
    }

    void init(void) const
    {
      BOOST_ASSERT(this->gid_);
      this->base_type::init(this->gid_);
    }

    future_size_type eager_signal_init(void) const
    {
      BOOST_ASSERT(this->gid_);
      return this->base_type::eager_signal_init(this->gid_);
    }

    void finalize_init(
        distribution_type const & distribution, 
        ids_type const & sibling_ids,
        size_type size) const
    {
      BOOST_ASSERT(this->gid_);
      this->base_type::finalize_init(
          this->gid_, distribution, sibling_ids, size);
    }

    void ready(void) const
    {
      BOOST_ASSERT(this->gid_);
      this->base_type::ready(this->gid_);
    }

    ////////////////////////////////////////////////////////////////////////////
    // Use

    distribution_type get_distribution(void) const
    {
      BOOST_ASSERT(this->gid_);
      return this->base_type::get_distribution(this->gid_);
    }

    id_type local_to(size_type index) const
    {
      BOOST_ASSERT(this->gid_);
      return this->base_type::local_to(this->gid_, index);
    }

    size_type size(void) const
    {
      BOOST_ASSERT(this->gid_);
      return this->base_type::size(this->gid_);
    }
  };
}}

#endif

