// Copyright (c) 2010-2011 Dylan Stark
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying 
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(EXAMPLES_SGAB_BC_SSSP_CLIENT_20110217T0826)
#define EXAMPLES_SGAB_BC_SSSP_CLIENT_20110217T0826

#include "../../../pxgl/pxgl.hpp"
#include "../../../pxgl/util/component.hpp"
#include "../../../pxgl/util/hpx.hpp"

#include "bc_sssp.hpp"

#include "../../../pxgl/xua/range.hpp"
#include "../../../pxgl/xua/arbitrary_distribution.hpp"

////////////////////////////////////////////////////////////////////////////////
typedef unsigned long size_type;

typedef hpx::naming::id_type id_type;
typedef std::vector<id_type> ids_type;

typedef hpx::naming::gid_type gid_type;
typedef std::vector<hpx::naming::gid_type> gids_type;

typedef pxgl::xua::arbitrary_distribution<id_type, pxgl::xua::range>
        arbitrary_distribution_type;
typedef arbitrary_distribution_type distribution_type;

typedef hpx::lcos::promise<hpx::util::unused_type> future_void_type;
typedef hpx::lcos::promise<size_type> future_size_type;

////////////////////////////////////////////////////////////////////////////////
// Stubs interface
namespace examples { namespace sgab { namespace stubs {
  struct bc_sssp
    : HPX_STUBS_BASE_0(bc_sssp)
  {
    ////////////////////////////////////////////////////////////////////////////
    typedef server::bc_sssp server_type;
    typedef server_type::size_type size_type;
    typedef server_type::sizes_type sizes_type;
    typedef server_type::distribution_type distribution_type;
    typedef server_type::id_type id_type;
    typedef server_type::ids_type ids_type;
    typedef server_type::graph_type graph_type;
    typedef server_type::bc_scores_type bc_scores_type;

    ////////////////////////////////////////////////////////////////////////////
    // Construction/initialization
    static void async_instantiate(
        id_type const & id,
        distribution_type const & distribution,
        id_type const & graph_id,
        id_type const & bc_scores_id)
    {
      typedef server_type::instantiate_action action_type;
      hpx::applier::apply<action_type>(
          id, id, distribution, graph_id, bc_scores_id);
    }

    static void async_replicate(
        id_type const & id,
        distribution_type const & distribution,
        ids_type const & sibling_ids,
        id_type const & graph_id,
        id_type const & bc_scores_id)
    {
      typedef server_type::replicate_action action_type;
      hpx::applier::apply<action_type>(
          id, distribution, sibling_ids, graph_id, bc_scores_id);
    }

    static void constructed(id_type const & id)
    {
      typedef server_type::constructed_action action_type;
      hpx::lcos::eager_future<action_type>(id).get();
    }

    static void ready(id_type const & id)
    {
      typedef server_type::ready_action action_type;
      hpx::lcos::eager_future<action_type>(id).get();
    }

    static void ready_all(id_type const & id)
    {
      typedef server_type::ready_all_action action_type;
      hpx::lcos::eager_future<action_type>(id).get();
    }

    static void ended(id_type const & id)
    {
      typedef server_type::ended_action action_type;
      hpx::lcos::eager_future<action_type>(id).get();
    }

    ////////////////////////////////////////////////////////////////////////////
    // Use
    static void begin(
        id_type const & id,
        size_type start)
    {
      typedef server_type::begin_action action_type;
      hpx::applier::apply<action_type>(id, start);
    }

    static sizes_type sync_expand_source(
        id_type const & id,
        size_type source)
    {
      typedef server_type::expand_source_action action_type;
      return hpx::lcos::eager_future<action_type>(id, source).get();
    }

    static future_size_type eager_expand_target(
        id_type const & id,
        size_type target,
        size_type source,
        long d_source,
        size_type sigma_source)
    {
      typedef server_type::expand_target_action action_type;
      return hpx::lcos::eager_future<action_type>(
          id, target, source, d_source, sigma_source);
    }

    static void sync_contract_target(
        id_type const & id,
        size_type target,
        size_type start)
    {
      typedef server_type::contract_target_action action_type;
      hpx::lcos::eager_future<action_type>(id, target, start).get();
    }

    static void sync_contract_source(
        id_type const & id,
        size_type source,
        size_type target,
        size_type sigma_target,
        double delta_target)
    {
      typedef server_type::contract_source_action action_type;
      hpx::lcos::eager_future<action_type>(
          id, source, target, sigma_target, delta_target).get();
    }
  };
}}}

////////////////////////////////////////////////////////////////////////////////
// Client interface
namespace examples { namespace sgab {
  class bc_sssp
    : public HPX_CLIENT_BASE_0(bc_sssp)
  {
  //////////////////////////////////////////////////////////////////////////////
  private:
    typedef HPX_CLIENT_BASE_0(bc_sssp) base_type;

  public:
    bc_sssp()
      : base_type(hpx::naming::invalid_id)
    {}

    bc_sssp(hpx::naming::id_type id)
      : base_type(id)
    {}

    ////////////////////////////////////////////////////////////////////////////
    // Graph types
    typedef stubs::bc_sssp stubs_type;
    typedef stubs_type::size_type size_type;
    typedef stubs_type::sizes_type sizes_type;
    typedef stubs_type::id_type id_type;
    typedef stubs_type::ids_type ids_type;
    typedef stubs_type::distribution_type distribution_type;
    typedef stubs_type::graph_type graph_type;
    typedef stubs_type::bc_scores_type bc_scores_type;
 
    ////////////////////////////////////////////////////////////////////////////
    // Construction/initialization
    void async_instantiate(
        graph_type const & graph,
        bc_scores_type const & bc_scores) const
    {
      BOOST_ASSERT(this->gid_);

      this->base_type::async_instantiate(
          this->gid_, graph.get_distribution(),
          graph.get_gid(), bc_scores.get_gid());
    }

    void async_replicate(
        distribution_type const & distribution, 
        ids_type const & sibling_ids,
        id_type const & graph_id,
        id_type const & bc_scores_id) const
    {
      BOOST_ASSERT(this->gid_);

      this->base_type::async_replicate(this->gid_, distribution, sibling_ids,
          graph_id, bc_scores_id);
    }
    void constructed(void)
    {
      BOOST_ASSERT(this->gid_);
      this->base_type::constructed(this->gid_);
    }

    void ready(void) const
    {
      BOOST_ASSERT(this->gid_);
      this->base_type::ready(this->gid_);
    }

    void ready_all(void) const
    {
      BOOST_ASSERT(this->gid_);
      this->base_type::ready_all(this->gid_);
    }

    void ended(void) const
    {
      BOOST_ASSERT(this->gid_);
      this->base_type::ended(this->gid_);
    }

    ////////////////////////////////////////////////////////////////////////////
    // Use
    void operator()(size_type start) const
    {
      BOOST_ASSERT(this->gid_);
      this->base_type::begin(this->gid_, start);
    }

    sizes_type sync_expand_source(size_type source) const
    {
      BOOST_ASSERT(this->gid_);
      return this->base_type::sync_expand_source(this->gid_, source);
    }

    future_size_type eager_expand_target(
        size_type target,
        size_type source,
        long d_source,
        size_type sigma_source) const
    {
      BOOST_ASSERT(this->gid_);
      return this->base_type::eager_expand_target(
          this->gid_, target, source, d_source, sigma_source);
    }

    void sync_contract_target(
        size_type target,
        size_type start) const
    {
      BOOST_ASSERT(this->gid_);
      this->base_type::sync_contract_target(this->gid_, target, start);
    }

    void sync_contract_source(
        size_type source,
        size_type target,
        size_type sigma_target,
        double delta_target) const
    {
      BOOST_ASSERT(this->gid_);
      this->base_type::sync_contract_source(
          this->gid_, source, target, sigma_target, delta_target);
    }
  };
}}

#endif

