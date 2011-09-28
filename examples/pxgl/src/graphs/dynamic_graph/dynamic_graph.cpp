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

#include "../../../pxgl/graphs/dynamic_graph_client.hpp"
#include "../../../pxgl/graphs/extension_info.hpp"
#include "../../../pxgl/graphs/edge_tuple.hpp"

#include "../../../pxgl/util/scoped_use.hpp"
#include "../../../pxgl/util/futures.hpp"
#include "../../../pxgl/xua/range.hpp"
#include "../../../pxgl/xua/arbitrary_distribution.hpp"

////////////////////////////////////////////////////////////////////////////////
//#define PXGL_DEBUG_ALIGN

////////////////////////////////////////////////////////////////////////////////
// Define logging helper
#define LSUBG_LOG_fatal 1
#define LSUBG_LOG_debug 0
#define LSUBG_LOG_info  0
#define LSUBG_LOG_dot   0
#define LSUBG_LOG__ping 0

#if LSUBG_LOG_ping == 1
#  define LSUBG_ping(major,minor) YAP_now(major,minor)
#else
#  define LSUBG_ping(major,minor) do {} while(0)
#endif

#if LSUBG_LOG_dot == 1
#define LSUBG_dot(str,...) YAPs(str,__VA_ARGS__)
#else
#define LSUBG_dot(str,...) do {} while(0)
#endif

#if LSUBG_LOG_info == 1
#define LSUBG_info(str,...) YAPs(str,__VA_ARGS__)
#else
#define LSUBG_info(str,...) do {} while(0)
#endif

#if LSUBG_LOG_debug == 1
#define LSUBG_debug(str,...) YAPs(str,__VA_ARGS__)
#else
#define LSUBG_debug(str,...) do {} while(0)
#endif

#if LSUBG_LOG_fatal == 1
#define LSUBG_fatal(str,...) YAPs(str,__VA_ARGS__)
#else
#define LSUBG_fatal(str,...) do {} while(0)
#endif

////////////////////////////////////////////////////////////////////////////////
typedef unsigned long size_type;

typedef hpx::naming::id_type id_type;

typedef pxgl::xua::arbitrary_distribution<
    hpx::naming::id_type,
    pxgl::xua::range
> arbitrary_range_type;

typedef pxgl::graphs::server::edge_tuple edge_tuple_type;

typedef pxgl::graphs::dynamic_graph dynamic_graph_type;
typedef pxgl::graphs::server::dynamic_graph dynamic_graph_member_type;

typedef hpx::components::managed_component<
    dynamic_graph_member_type
> dynamic_graph_component_type;

typedef pxgl::graphs::server::extension_info_type extension_info_type;
typedef pxgl::graphs::server::signal_value_type signal_value_type;

typedef hpx::lcos::promise<signal_value_type> future_signal_value_type;
typedef std::vector<future_signal_value_type> future_signal_values_type;

typedef hpx::lcos::promise<hpx::util::unused_type> future_void_type;
typedef std::vector<future_void_type> future_voids_type;

////////////////////////////////////////////////////////////////////////////////
namespace pxgl { namespace graphs { namespace server  {
  dynamic_graph::dynamic_graph()
    : order_(0),
      size_(0),
      me_(hpx::naming::invalid_id),
      here_(hpx::get_runtime().get_process().here()),
      siblings_(0),
      map_(0),
      constructed_(false),
      initialized_(false)
  {
    use_feb_.set(feb_data_type(1));
  }

  dynamic_graph::~dynamic_graph()
  {
    *((int*)0) = 1;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Construction/initialization

  void dynamic_graph::set_distribution(
      id_type const & me, 
      distribution_type const & distribution)
  {
    pxgl::util::scoped_use l(use_feb_);

    not_constructed();
    not_ready();

    // Set this member's name
    me_ = me;

    // Set the distribution containing the "logical coverage" (extent)
    distribution_ = distribution;

    // Setup the coverage
    distribution_.extend(distribution.locale_id(here_));
    siblings_.push_back(dynamic_graph_type(me_));

    // Set this member as constructed
    constructed_feb_.set(feb_data_type(1));
    constructed_ = true;

    LSUBG_ping("Subgraph set distribution", "Stop");
  }

  void dynamic_graph::request_extension(size_type v)
  {
    not_ready();

    future_voids_type outstanding_adds;

    {
      pxgl::util::scoped_use l(use_feb_);

      // Extend the dynamic_graph coverage if necessary
      size_type const there_index = distribution_.member_id(v);
      if (distribution_.size() == there_index)
      {
        // Create new member
        dynamic_graph_type new_member;
        new_member.create(distribution_.locale(v));
        
        // Extend the distribution for the dynamic_graph
        siblings_.push_back(new_member);
        distribution_.extend(distribution_.locale_id(v));

        // Set the distribution for the new member
        ids_type sibling_ids(siblings_.size());
        for (size_type i = 0; i < siblings_.size(); i++)
        {
          sibling_ids[i] = siblings_[i].get_gid();
        }

        new_member.replistruct(
            new_member.get_gid(), distribution_, sibling_ids);

        // Add the vertex
        outstanding_adds.push_back(new_member.eager_add_vertex(v));
      }
      else
      {
        // Target member already exists
        assert(0);

      //  outstanding_adds.push_back(
      //      siblings_[there_index].eager_add_vertex(v));
      }
    }

    while (outstanding_adds.size() > 0)
    {
      outstanding_adds.back().get();
      outstanding_adds.pop_back();
    }

    LSUBG_ping("Subgraph request extension", "Stop");
  }

  void dynamic_graph::replistruct(
      id_type const & me,
      distribution_type const & distribution,
      ids_type const & sibling_ids)
  {
    pxgl::util::scoped_use l(use_feb_);

    not_constructed();
    not_ready();

    // Set this member's name
    me_ = me;

    // Set distribution
    distribution_ = distribution;

    siblings_.clear();
    for (size_type i = 0; i < sibling_ids.size(); i++)
    {
      siblings_.push_back(dynamic_graph_type(sibling_ids[i]));
    }

    // Set this member as constructed
    constructed_feb_.set(feb_data_type(1));
    constructed_ = true;
  }
  
  void dynamic_graph::constructed(void)
  {
    while (!constructed_)
    {
      LSUBG_ping("Subgraph", "Waiting for construction");

      feb_data_type d;
      constructed_feb_.read(d);

      if (1 == d.which())
      {
        error_type e = boost::get<error_type>(d);
        boost::rethrow_exception(e);
      }

      LSUBG_ping("Subgraph", "Readied construction");
    }
  }

  void dynamic_graph::not_constructed(void)
  {
    assert(!constructed_);
  }

  // Note: only call when we can guarantee there are no inflight actions
  void dynamic_graph::init(void)
  {
    not_ready();
    constructed();

    {
      pxgl::util::scoped_use l(use_feb_);

      // Build representation
      // ...

      // Signal initialization
      {
        future_signal_values_type outstanding_signals;

        BOOST_FOREACH(dynamic_graph_type const & sibling, siblings_)
        {
          if (sibling.get_gid() != me_)
          {
            outstanding_signals.push_back(
                sibling.eager_signal_init());
          }
        }

        while (outstanding_signals.size() > 0)
        {
          signal_value_type ret = outstanding_signals.back().get();
          outstanding_signals.pop_back();

          order_ += ret.order();
          size_ += ret.size();
        }
      }

      // Finalize initialization
      {
        // Collect sibling GIDs
        ids_type sibling_ids(siblings_.size());
        for (size_type i = 0; i < siblings_.size(); i++)
        {
          sibling_ids[i] = siblings_[i].get_gid();
        }

        // Broadcast global information
        BOOST_FOREACH(dynamic_graph_type const & sibling, siblings_)
        {
          if (sibling.get_gid() != me_)
          {
            sibling.finalize_init(distribution_, sibling_ids, order_, size_);
          }
        }
      }

      // Set as initialized
      initialized_feb_.set(feb_data_type(1));
      initialized_ = true;
    }
  }

  signal_value_type dynamic_graph::signal_init(void)
  {
    pxgl::util::scoped_use l(use_feb_);

    not_ready();
    constructed();

    // Build representation
    // ...

    return signal_value_type(order_, size_);
  }

  void dynamic_graph::finalize_init(
      distribution_type const & distribution,
      ids_type const & sibling_ids,
      size_type order,
      size_type size)
  {
    pxgl::util::scoped_use l(use_feb_);

    not_ready();
    constructed();

    // Finalize the coverage
    distribution_ = distribution;

    siblings_.clear();
    for (size_type i = 0; i < sibling_ids.size(); i++)
    {
      siblings_.push_back(dynamic_graph_type(sibling_ids[i]));
    }

    order_ = order;
    size_ = size;

    // Set this member as constructed
    initialized_feb_.set(feb_data_type(1));
    initialized_ = true;
  }
  
  void dynamic_graph::ready(void)
  {
    while (!initialized_)
    {
      LSUBG_ping("Subgraph", "Waiting");

      feb_data_type d;
      initialized_feb_.read(d);

      if (1 == d.which())
      {
        error_type e = boost::get<error_type>(d);
        boost::rethrow_exception(e);
      }

      LSUBG_ping("Subgraph", "Readied");
    }
  }

  void dynamic_graph::not_ready(void)
  {
    assert(!initialized_);
  }

  //////////////////////////////////////////////////////////////////////////////
  // Use
  size_type dynamic_graph::add_edge(edge_tuple_type const & edge)
  {
    // Guard interface
    not_ready();
    constructed();

#ifdef PXGL_DEBUG_ALIGN
    assert(distribution_.locale(edge.source()) == here_);
#endif

    future_voids_type outstanding_adds;
    {
      // Guard graph while updating
      pxgl::util::scoped_use l(use_feb_);

      size_type const source(edge.source());
      size_type const target(edge.target());

      // Add source vertex here
      if (map_.find(source) == map_.end())
      {
        map_[source];
        order_++;

        LSUBG_dot("\t%u;\n", source);
      }

      // Add edge here
      sizes_type & out_edges = map_[source];
      sizes_type::const_iterator eit =
          find(out_edges.begin(), out_edges.end(), target);
      if (eit == out_edges.end())
      {
        out_edges.push_back(target);
        size_++;

        LSUBG_dot("\t\t%u -> %u;\n", source, target);

        // Add target vertex somewhere
        if (distribution_.locale_id(target)
            == distribution_.locale_id(source))
        {
          // Add target vertex here
          if (map_.find(target) == map_.end())
          {
            map_[target];
            order_++;

            LSUBG_dot("\t%u;\n", target);
          }
        }
        else
        {
          // Add target on remote member
          size_type const there_index = distribution_.member_id(target);
          if (distribution_.size() == there_index)
          {
            // Request dynamic_graph leader to add target vertex to unknown member
            outstanding_adds.push_back(
                siblings_[0].eager_request_extension(target));
          }
          else
          {
            // Request member to add target vertex
            outstanding_adds.push_back(
                siblings_[there_index].eager_add_vertex(target));
          }
        }
      }
      else
      {
        //Already have the edge
      }
    }

    // Wait for acknowledgement that the vertex was added
    while (outstanding_adds.size() > 0)
    {
      outstanding_adds.back().get();
      outstanding_adds.pop_back();
    }

    LSUBG_ping("Subgraph add edge", "Stop");

    return 0;
  }

  size_type dynamic_graph::add_edges(edge_tuples_type const & edges)
  {
    // Guard interface
    not_ready();
    constructed();

#ifdef PXGL_DEBUG_ALIGN
    assert(distribution_.locale(edge.source()) == here_);
#endif

    future_voids_type outstanding_adds;
    {
      BOOST_FOREACH(edge_tuple_type const & edge, edges)
      {
        // Guard graph while updating
        pxgl::util::scoped_use l(use_feb_);

        size_type const source(edge.source());
        size_type const target(edge.target());

        // Add source vertex here
        if (map_.find(source) == map_.end())
        {
          map_[source];
          order_++;

          LSUBG_dot("\t%u;\n", source);
        }

        // Add edge here
        sizes_type & out_edges = map_[source];
        sizes_type::const_iterator eit =
            find(out_edges.begin(), out_edges.end(), target);
        if (eit == out_edges.end())
        {
          out_edges.push_back(target);
          size_++;

          LSUBG_dot("\t\t%u -> %u;\n", source, target);

          // Add target vertex somewhere
          if (distribution_.locale_id(target)
              == distribution_.locale_id(source))
          {
            // Add target vertex here
            if (map_.find(target) == map_.end())
            {
              map_[target];
              order_++;

              LSUBG_dot("\t%u;\n", target);
            }
          }
          else
          {
            // Add target on remote member
            size_type const there_index = distribution_.member_id(target);
            if (distribution_.size() == there_index)
            {
              // Request dynamic_graph leader to add target vertex to unknown member
              outstanding_adds.push_back(
                  siblings_[0].eager_request_extension(target));
            }
            else
            {
              // Request member to add target vertex
              outstanding_adds.push_back(
                  siblings_[there_index].eager_add_vertex(target));
            }
          }
        }
        else
        {
          //Already have the edge
        }
      }
    }

    // Wait for acknowledgement that the vertex was added
    while (outstanding_adds.size() > 0)
    {
      outstanding_adds.back().get();
      outstanding_adds.pop_back();
    }

    LSUBG_ping("Subgraph add edge", "Stop");

    return 0;
  }

  void dynamic_graph::add_vertex(size_type v)
  {
    not_ready();
    constructed();

#ifdef PXGL_DEBUG_ALIGN
    assert(distribution_.locale(v) == here_);
#endif

    {
      pxgl::util::scoped_use l(use_feb_);

      // Add vertex here
      if (map_.find(v) == map_.end())
      {
        map_[v];
        order_++;

        LSUBG_dot("\t%u;\n", v);
      }

      LSUBG_info("Added vertex %u to member %u.\n", v, me_.get_lsb());
    }
  }

  size_type dynamic_graph::order(void)
  {
    ready();

    return order_;
  }

  size_type dynamic_graph::size(void)
  {
    ready();

    return size_;
  }
}}}

////////////////////////////////////////////////////////////////////////////////
HPX_REGISTER_COMPONENT_MODULE();

////////////////////////////////////////////////////////////////////////////////
// Register component factory for CSR graphs
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    dynamic_graph_component_type, 
    dynamic_graph);

////////////////////////////////////////////////////////////////////////////////
// Add serialization support for CSR graph actions
HPX_REGISTER_ACTION_EX(
    dynamic_graph_component_type::wrapped_type::set_distribution_action,
    dynamic_graph_set_distribution_action);
HPX_REGISTER_ACTION_EX(
    dynamic_graph_component_type::wrapped_type::request_extension_action,
    dynamic_graph_request_extension_action);
HPX_REGISTER_ACTION_EX(
    dynamic_graph_component_type::wrapped_type::replistruct_action,
    dynamic_graph_replistruct_action);
HPX_REGISTER_ACTION_EX(
    dynamic_graph_component_type::wrapped_type::constructed_action,
    dynamic_graph_constructed_action);
HPX_REGISTER_ACTION_EX(
    dynamic_graph_component_type::wrapped_type::init_action,
    dynamic_graph_init_action);
HPX_REGISTER_ACTION_EX(
    dynamic_graph_component_type::wrapped_type::signal_init_action,
    dynamic_graph_signal_init_action);
HPX_REGISTER_ACTION_EX(
    dynamic_graph_component_type::wrapped_type::finalize_init_action,
    dynamic_graph_finalize_init_action);
HPX_REGISTER_ACTION_EX(
    dynamic_graph_component_type::wrapped_type::ready_action,
    dynamic_graph_ready_action);
HPX_REGISTER_ACTION_EX(
    dynamic_graph_component_type::wrapped_type::add_edge_action,
    dynamic_graph_add_edge_action);
HPX_REGISTER_ACTION_EX(
    dynamic_graph_component_type::wrapped_type::add_edges_action,
    dynamic_graph_add_edges_action);
HPX_REGISTER_ACTION_EX(
    dynamic_graph_component_type::wrapped_type::add_vertex_action,
    dynamic_graph_add_vertex_action);
HPX_REGISTER_ACTION_EX(
    dynamic_graph_component_type::wrapped_type::order_action,
    dynamic_graph_order_action);
HPX_REGISTER_ACTION_EX(
    dynamic_graph_component_type::wrapped_type::size_action,
    dynamic_graph_size_action);

////////////////////////////////////////////////////////////////////////////////
// Define CSR graph component
HPX_DEFINE_GET_COMPONENT_TYPE(dynamic_graph_component_type::wrapped_type);

////////////////////////////////////////////////////////////////////////////////
// Add futures support
HPX_REGISTER_FUTURE(hpx::util::unused_type, unused);
HPX_REGISTER_FUTURE(id_type, id);
HPX_REGISTER_FUTURE(size_type, size);

