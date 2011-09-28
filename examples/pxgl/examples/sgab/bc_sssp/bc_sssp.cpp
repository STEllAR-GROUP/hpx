// Copyright (c) 2010-2011 Dylan Stark
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying 
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

// Bring in necessary headers for setting up an HPX component
#include <hpx/runtime/components/component_factory.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

// Bring in graph and aux. definition
#include "../../../pxgl/pxgl.hpp"

#include "../../../examples/sgab/bc_sssp/bc_sssp_client.hpp"

#include "../../../pxgl/util/scoped_use.hpp"
#include "../../../pxgl/util/futures.hpp"

#include "../../../pxgl/xua/range.hpp"
#include "../../../pxgl/xua/arbitrary_distribution.hpp"

#include "../../../pxgl/xua/control.hpp"

////////////////////////////////////////////////////////////////////////////////
#define PXGL_DEBUG_ALIGN

////////////////////////////////////////////////////////////////////////////////
// Define logging helper
#define LBCSSSP_LOG_fatal 1
#define LBCSSSP_LOG_debug 0
#define LBCSSSP_LOG_info  0
#define LBCSSSP_LOG_dot   0
#define LBCSSSP_LOG__ping 0

#if LBCSSSP_LOG_ping == 1
#  define LBCSSSP_ping(major,minor) YAP_now(major,minor)
#else
#  define LBCSSSP_ping(major,minor) do {} while(0)
#endif

#if LBCSSSP_LOG_dot == 1
#define LBCSSSP_dot(str,...) YAPs(str,__VA_ARGS__)
#else
#define LBCSSSP_dot(str,...) do {} while(0)
#endif

#if LBCSSSP_LOG_info == 1
#define LBCSSSP_info(str,...) YAPs(str,__VA_ARGS__)
#else
#define LBCSSSP_info(str,...) do {} while(0)
#endif

#if LBCSSSP_LOG_debug == 1
#define LBCSSSP_debug(str,...) YAPs(str,__VA_ARGS__)
#else
#define LBCSSSP_debug(str,...) do {} while(0)
#endif

#if LBCSSSP_LOG_fatal == 1
#define LBCSSSP_fatal(str,...) YAPs(str,__VA_ARGS__)
#else
#define LBCSSSP_fatal(str,...) do {} while(0)
#endif

////////////////////////////////////////////////////////////////////////////////
typedef unsigned long size_type;
typedef std::vector<size_type> sizes_type;

typedef hpx::naming::id_type id_type;

typedef pxgl::xua::arbitrary_distribution<
    hpx::naming::id_type,
    pxgl::xua::range
> arbitrary_range_type;

typedef examples::sgab::bc_sssp bc_sssp_type;
typedef examples::sgab::server::bc_sssp bc_sssp_member_type;

typedef hpx::components::managed_component<
    bc_sssp_member_type
> bc_sssp_component_type;

typedef hpx::lcos::promise<hpx::util::unused_type> future_void_type;
typedef std::vector<future_void_type> future_voids_type;

typedef hpx::lcos::promise<size_type> future_size_type;
typedef std::vector<future_size_type> future_sizes_type;

////////////////////////////////////////////////////////////////////////////////
namespace examples { namespace sgab { namespace server  {
  bc_sssp::bc_sssp()
    : me_(hpx::naming::invalid_id),
      here_(hpx::get_runtime().get_process().here()),
      siblings_(0),
      constructed_(false),
      initialized_(false),
      ended_(false)
  {
    use_feb_.set(feb_data_type(1));
  }

  bc_sssp::~bc_sssp()
  {
    *((int*)0) = 1;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Construction/initialization
  inline void bc_sssp::initialize_local_variables(void)
  {
    sizes_type const & vertices = *(graph_.sync_vertices());
    size_type const local_size(
        index(*(max_element(vertices.begin(), vertices.end()))) + 1);
        
    LBCSSSP_debug("Local size: %u\n", local_size);

    BC_ptr_ = bc_scores_.sync_init_items(local_size);

    P_ = predecessors_type(local_size);

    sigma_ = sizes_type(local_size, 0);

    d_ = longs_type(local_size, -1);

    delta_ = doubles_type(local_size, 0.0);
  }

  inline size_type const bc_sssp::index(size_type const vertex)
  {
    return (vertex/distribution_.size()) + (vertex % distribution_.size());
  }

  void bc_sssp::instantiate(
      id_type const & me,
      distribution_type const & distribution,
      id_type const & graph_id,
      id_type const & bc_scores_id)
  {
    {
      pxgl::util::scoped_use l(use_feb_);

      assert(!initialized_);

      me_ = me;
      distribution_ = distribution;

      graph_ = graph_type(graph_id);
      bc_scores_ = bc_scores_type(bc_scores_id);

      typedef distribution_type::locality_ids_type locality_ids_type;
      locality_ids_type const & locales = distribution_.coverage();
      size_type const extent = locales.size();

      siblings_ = std::vector<bc_sssp_type>(extent);
      for (size_type i = 0; i < extent; i++)
      {
        if (locales[i] != here_)
        {
          siblings_[i].create(locales[i]);
        }
        else
        {
          siblings_[i] = bc_sssp_type(me_);
        }
      }

      // Collect ids for siblings
      // Note: we include this actor in the collection of siblings
      ids_type sibling_ids(distribution.size());
      for (size_type i =0; i < extent; i++)
      {
        sibling_ids[i] = siblings_[i].get_gid();
      }

      // Construct siblings
      for (size_type i =0; i < extent; i++)
      {
        if (locales[i] != here_)
        {
          siblings_[i].async_replicate(
              distribution, sibling_ids, 
              graph_.local_to(i), bc_scores_.local_to(i));
        }
      }
     
      // Consider this process member constructed
      constructed_feb_.set(feb_data_type(1));
      constructed_ = true;

      initialize_local_variables();

      // Consider this process member initialized
      initialized_feb_.set(feb_data_type(1));
      initialized_ = true;
    }
  }

  void bc_sssp::replicate(
      distribution_type const & distribution, 
      ids_type const & sibling_ids,
      id_type const & graph_id,
      id_type const & bc_scores_id)
  {
    {
      pxgl::util::scoped_use l(use_feb_);

      assert(!initialized_);

      // Set distribution
      distribution_ = distribution;

      graph_ = graph_type(graph_id);
      bc_scores_ = bc_scores_type(bc_scores_id);

      typedef distribution_type::locality_ids_type locality_ids_type;
      locality_ids_type const & locales = distribution_.coverage();
      size_type const extent = locales.size();

      // Set siblings and me
      siblings_ = std::vector<bc_sssp_type>(extent);
      for (size_type i = 0; i < extent; i++)
      {
        siblings_[i] = bc_sssp_type(sibling_ids[i]);

        if (locales[i] == here_)
        {
          me_ = sibling_ids[i];
        }
      }

      // Set as constructed
      constructed_feb_.set(feb_data_type(1));
      constructed_ = true;

      // Initialize
      initialize_local_variables();

      // Set as initialized
      initialized_feb_.set(feb_data_type(1));
      initialized_ = true;
    }
  }

  void bc_sssp::constructed(void)
  {
    while (!constructed_)
    {
      feb_data_type d;
      constructed_feb_.read(d);

      if (1 == d.which())
      {
        error_type e = boost::get<error_type>(d);
        boost::rethrow_exception(e);
      }
    }
  }

  void bc_sssp::not_constructed(void)
  {
    assert(!constructed_);
  }

  void bc_sssp::ready(void)
  {
    while (!initialized_)
    {
      feb_data_type d;
      initialized_feb_.read(d);

      if (1 == d.which())
      {
        error_type e = boost::get<error_type>(d);
        boost::rethrow_exception(e);
      }
    }
  }

  void bc_sssp::not_ready(void)
  {
    assert(!initialized_);
  }

  void bc_sssp::ready_all(void)
  {
    constructed();

    BOOST_FOREACH(bc_sssp_type sibling, siblings_)
    {
      sibling.ready();
    }
  }

  void bc_sssp::ended(void)
  {
    while (!ended_)
    {
      feb_data_type d;
      ended_feb_.read(d);

      if (1 == d.which())
      {
        error_type e = boost::get<error_type>(d);
        boost::rethrow_exception(e);
      }
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Use
  void bc_sssp::begin(size_type start)
  {
    ready();

    // Start
    size_type const start_id(index(start));
    LBCSSSP_info("Start: %u->%u / %u\n", 
        start, start_id, graph_.sync_vertices()->size());
    
    sigma_[start_id] = 1;
    d_[start_id] = 0;
        
    Q_.push(start);
    while (!Q_.empty())
    {
      size_type const source = Q_.front();
      Q_.pop();

      S_.push_back(source);

      size_type const i = distribution_.locale_id(source);
      sizes_type const new_vertices = siblings_[i].sync_expand_source(source);

      BOOST_FOREACH(size_type const & target, new_vertices)
      {
        Q_.push(target);
      }
    }

    LBCSSSP_info("bc_sssp out: %u nodes in shortest paths tree.\n", S_.size());

    // Synthesize BC scores back up the shortest paths tree
    while (S_.size() > 0)
    {
      size_type const target(S_.back());
      S_.pop_back();

      size_type const i = distribution_.locale_id(target);
      siblings_[i].sync_contract_target(target, start);
    }

    // Set bc_scores as initialized
    //{
    //  bc_scores_type::items_type empty_items;
    //  pxgl::xua::for_each_comp<
    //      bc_scores_type, bc_scores_member_type::init_action
    //  >(bc_scores_.get_gid(), empty_items);
    //}

    // Signal the end of this process
    ended_feb_.set(feb_data_type(1));
    ended_ = true;
  }

  sizes_type bc_sssp::expand_source(size_type source)
  {
    ready();

    future_sizes_type outstanding_actions;
    {
      long const d_source = d_[index(source)];
      size_type const sigma_source = sigma_[index(source)];

      graph_type::edge_iterator_type neighbors(graph_.sync_neighbors(source));
      graph_type::edge_tuples_type const & edges(neighbors.edges());
      size_type const begin(neighbors.begin());
      size_type const end(neighbors.end());
      for (size_type i = begin; i < end; i++)
      {
        graph_type::edge_tuple_type const & adj(edges[i]);

        // Only process edges not divisible by 8
        if (((size_type)adj.weight() & 7) != 0)
        {
          size_type const i = distribution_.locale_id(adj.target());
          outstanding_actions.push_back(
              siblings_[i].eager_expand_target(
                  adj.target(), source, d_source, sigma_source));
        }
      }
    }

    sizes_type new_vertices;
    {
      while (outstanding_actions.size() > 0)
      {
        if (graph_type::invalid_vertex() != outstanding_actions.back().get())
        {
          new_vertices.push_back(outstanding_actions.back().get());
        }
        outstanding_actions.pop_back();
      }
    }

    return new_vertices;
  }

  size_type bc_sssp::expand_target(
      size_type target, 
      size_type source,
      long d_source,
      size_type sigma_source)
  {
    ready();

    // Find local index of vertex
    size_type const target_id(index(target));

    size_type new_vertex = graph_type::invalid_vertex();

    if (d_[target_id] < 0)
    {
      new_vertex = target;
      d_[target_id] = d_source + 1;
    }

    if ((d_[target_id] = d_source + 1))
    {
      sigma_[target_id] += sigma_source;

      {
        pxgl::util::scoped_use l(use_feb_);

        P_[target_id].push_back(source);
      }
    }

    return new_vertex;
  }

  void bc_sssp::contract_target(
      size_type target,
      size_type start)
  {
    ready();

    size_type const sigma_target(sigma_[index(target)]);
    double const delta_target(delta_[index(target)]);

    BOOST_FOREACH(size_type const & source, P_[index(target)])
    {
      siblings_[distribution_.locale_id(source)].sync_contract_source(
          source, target, sigma_target, delta_target);
    }

    if (target != start)
    {
      pxgl::util::scoped_use l(use_feb_);

      bc_scores_.sync_init_incr(index(target), delta_[index(target)]);
    }
  }

  void bc_sssp::contract_source(
      size_type source,
      size_type target,
      size_type sigma_target,
      double delta_target)
  {
    ready();

    delta_[index(source)] += 
        (sigma_[index(source)] / sigma_target)
            * (1 + delta_target);
  }
}}}

////////////////////////////////////////////////////////////////////////////////
HPX_REGISTER_COMPONENT_MODULE();

////////////////////////////////////////////////////////////////////////////////
// Register component factory for CSR graphs
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    bc_sssp_component_type, 
    bc_sssp);

////////////////////////////////////////////////////////////////////////////////
// Add serialization support for CSR graph actions
HPX_REGISTER_ACTION_EX(
    bc_sssp_component_type::wrapped_type::instantiate_action,
    bc_sssp_instantiate_action);
HPX_REGISTER_ACTION_EX(
    bc_sssp_component_type::wrapped_type::replicate_action,
    bc_sssp_replicate_action);
HPX_REGISTER_ACTION_EX(
    bc_sssp_component_type::wrapped_type::constructed_action,
    bc_sssp_constructed_action);
HPX_REGISTER_ACTION_EX(
    bc_sssp_component_type::wrapped_type::ready_action,
    bc_sssp_ready_action);
HPX_REGISTER_ACTION_EX(
    bc_sssp_component_type::wrapped_type::ready_all_action,
    bc_sssp_ready_all_action);
HPX_REGISTER_ACTION_EX(
    bc_sssp_component_type::wrapped_type::ended_action,
    bc_sssp_ended_action);
HPX_REGISTER_ACTION_EX(
    bc_sssp_component_type::wrapped_type::begin_action,
    bc_sssp_begin_action);
HPX_REGISTER_ACTION_EX(
    bc_sssp_component_type::wrapped_type::expand_source_action,
    bc_sssp_expand_source_action);
HPX_REGISTER_ACTION_EX(
    bc_sssp_component_type::wrapped_type::expand_target_action,
    bc_sssp_expand_target_action);
HPX_REGISTER_ACTION_EX(
    bc_sssp_component_type::wrapped_type::contract_target_action,
    bc_sssp_contract_target_action);
HPX_REGISTER_ACTION_EX(
    bc_sssp_component_type::wrapped_type::contract_source_action,
    bc_sssp_contract_source_action);

////////////////////////////////////////////////////////////////////////////////
// Define CSR graph component
HPX_DEFINE_GET_COMPONENT_TYPE(bc_sssp_component_type::wrapped_type);

////////////////////////////////////////////////////////////////////////////////
// Add futures support
HPX_REGISTER_FUTURE(hpx::util::unused_type, unused);
HPX_REGISTER_FUTURE(size_type, size);
HPX_REGISTER_FUTURE(sizes_type, sizes);

