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

#include "../../../pxgl/xua/vmap_client.hpp"
#include "../../../pxgl/xua/vector.hpp"
#include "../../../pxgl/graphs/csr_graph.hpp"
#include "../../../pxgl/graphs/dynamic_graph_client.hpp"
#include "../../../pxgl/graphs/extension_info.hpp"
#include "../../../pxgl/graphs/edge_tuple.hpp"

#include "../../../pxgl/util/scoped_use.hpp"
#include "../../../pxgl/util/futures.hpp"
#include "../../../pxgl/xua/range.hpp"
#include "../../../pxgl/xua/arbitrary_distribution.hpp"

////////////////////////////////////////////////////////////////////////////////
// Define logging helper
#define LVMAP_LOG_fatal 1
#define LVMAP_LOG_debug 0
#define LVMAP_LOG_info  0
#define LVMAP_LOG__ping 0

#if LVMAP_LOG_ping == 1
#  define LVMAP_ping(major,minor) YAP_now(major,minor)
#else
#  define LVMAP_ping(major,minor) do {} while(0)
#endif

#if LVMAP_LOG_info == 1
#define LVMAP_info(str,...) YAPs(str,__VA_ARGS__)
#else
#define LVMAP_info(str,...) do {} while(0)
#endif

#if LVMAP_LOG_debug == 1
#define LVMAP_debug(str,...) YAPs(str,__VA_ARGS__)
#else
#define LVMAP_debug(str,...) do {} while(0)
#endif

#if LVMAP_LOG_fatal == 1
#define LVMAP_fatal(str,...) YAPs(str,__VA_ARGS__)
#else
#define LVMAP_fatal(str,...) do {} while(0)
#endif

////////////////////////////////////////////////////////////////////////////////
typedef unsigned long size_type;

typedef hpx::naming::id_type id_type;

typedef pxgl::xua::vmap vmap_type;
typedef pxgl::xua::server::vmap vmap_member_type;

typedef hpx::components::managed_component<
    vmap_member_type
> vmap_component_type;

typedef pxgl::xua::arbitrary_distribution<
    hpx::naming::id_type,
    pxgl::xua::range
> arbitrary_range_type;

typedef pxgl::xua::vector<
    arbitrary_distribution_type,
    pxgl::graphs::server::edge_tuple_type
> edge_container_type;

typedef pxgl::graphs::csr_graph<
    edge_container_type,
    arbitrary_distribution_type
> graph_type;

typedef pxgl::graphs::dynamic_graph subgraph_type;

typedef pxgl::graphs::server::extension_info_type extension_info_type;
typedef pxgl::graphs::server::adjacency<size_type, double>  adjacency_type;
typedef std::vector<adjacency_type> adjacencies_type;

typedef hpx::lcos::promise<size_type> future_size_type;
typedef std::vector<future_size_type> future_sizes_type;

typedef hpx::lcos::promise<hpx::util::unused_type> future_void_type;
typedef std::vector<future_void_type> future_voids_type;

////////////////////////////////////////////////////////////////////////////////
namespace pxgl { namespace xua { namespace server  {
  vmap::vmap()
    : size_(0),
      me_(hpx::naming::invalid_id),
      here_(hpx::get_runtime().get_process().here()),
      siblings_(0),
      map_(0),
      seen_(0),
      constructed_(false),
      initialized_(false)
  {
    use_feb_.set(feb_data_type(1));
  }

  vmap::~vmap()
  {
    *((int*)0) = 1;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Construction/initialization

  void vmap::construct(
      id_type const & me, 
      distribution_type const & distribution,
      id_type const & graph_id,
      id_type const & subgraph_id)
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
    siblings_.push_back(vmap_type(me_));

    // Setup associated source and target graphs
    graph_ = graph_type(graph_id);
    subgraph_ = subgraph_type(subgraph_id);

    // Set this member as constructed
    constructed_feb_.set(feb_data_type(1));
    constructed_ = true;

    LVMAP_ping("Vmap set distribution", "Stop");
  }

  extension_info_type vmap::request_extension(size_type k)
  {
    not_ready();

    {
      pxgl::util::scoped_use l(use_feb_);

      // Extend the vmap coverage if necessary
      if (distribution_.size() == distribution_.member_id(k))
      {
        assert(distribution_.size() >= siblings_.size()+1);

        size_type const new_index = distribution_.locale_id(k);

        // Create new member
        vmap_type new_member;
        new_member.create(distribution_.locale(new_index));
        
        // Extend the distribution for the vmap
        siblings_.push_back(new_member);
        distribution_.extend(new_index);

        // Set the distribution for the new member
        ids_type sibling_ids(siblings_.size());
        for (size_type i = 0; i < siblings_.size(); i++)
        {
          sibling_ids[i] = siblings_[i].get_gid();
        }

        new_member.replistruct(
            new_member.get_gid(), distribution_, sibling_ids,
            graph_.get_gid(), subgraph_.get_gid());
      
        return extension_info_type(sibling_ids, distribution_.map());
      }

      ids_type sibling_ids(siblings_.size());
      for (size_type i = 0; i < siblings_.size(); i++)
      {
        sibling_ids[i] = siblings_[i].get_gid();
      }

      return extension_info_type(sibling_ids, distribution_.map());
    }
  }

  void vmap::replistruct(
      id_type const & me,
      distribution_type const & distribution,
      ids_type const & sibling_ids,
      id_type const & graph_id,
      id_type const & subgraph_id)
  {
    pxgl::util::scoped_use l(use_feb_);

    not_constructed();
    not_ready();

    // Set this member's name
    me_ = me;

    // Set distribution
    distribution_ = distribution;

    for (size_type i = 0; i < sibling_ids.size(); i++)
    {
      siblings_.push_back(vmap_type(sibling_ids[i]));
    }

    // Setup associated graphs
    graph_ = graph_type(graph_id);
    subgraph_ = subgraph_type(subgraph_id);

    // Set this member as constructed
    constructed_feb_.set(feb_data_type(1));
    constructed_ = true;
  }
  
  void vmap::constructed(void)
  {
    while (!constructed_)
    {
      LVMAP_ping("Vmap", "Waiting for construction");

      feb_data_type d;
      constructed_feb_.read(d);

      if (1 == d.which())
      {
        error_type e = boost::get<error_type>(d);
        boost::rethrow_exception(e);
      }

      LVMAP_ping("Vmap", "Readied construction");
    }
  }

  void vmap::not_constructed(void)
  {
    assert(!constructed_);
  }

  void vmap::visit(size_type v, size_type depth)
  {
    not_ready();
    constructed();

    LVMAP_debug("Executing vmap[%u].visit(%u)\n", v, depth);
    
    // Check that max. depth was not reached
    bool was_seen = false;
    bool continue_search = true;
    {
      pxgl::util::scoped_use l(use_feb_);

      if (depth > seen_[v])
      {
        if (0 == seen_[v])
        {
          LVMAP_debug("%u: first coloring vertex %u with depth %u.\n",
              me_.get_lsb(), v, depth);

          // Color the vertex
          seen_[v] = depth;
        }
        else
        {
          was_seen = true;
        }
      }
      else
      {
        continue_search = false;
      }
    }

    LVMAP_debug(
        "Executing vmap[%u].visit(%u): was_seen(%d), continue_search(%d)\n", 
        v, depth, was_seen, continue_search);
    
    if (continue_search)
    {
      size_type const depth_1 = depth - 1;

      // Visit neighbors
      int payload = 0;
      pxgl::rts::get_ini_option(payload, "vmap.visit_payload");
      if (0 < payload)
      {
        edge_tuples_type edges;
        target_map_type visit_bins;

        graph_type::edge_iterator_type neighbors(graph_.sync_neighbors(v));
        graph_type::edge_tuples_type const & n_edges(neighbors.edges());
        size_type const begin(neighbors.begin());
        size_type const end(neighbors.end());
        for (size_type i = begin; i < end; i++)
        {
          edge_tuple_type const & edge(n_edges[i]);

          if (!was_seen)
          {
            edges.push_back(edge);
          }
         
          {
            if (0 < depth_1)
            {
              // Propogate visiting
              size_type const there_index = 
                  distribution_.member_id(edge.target());

              if (distribution_.size() == there_index)
              {
                // Target member is unknown, request extension
                extension_info_type const update = 
                    siblings_[0].eager_request_extension(
                        edge.target()).get();

                {
                  pxgl::util::scoped_use l(use_feb_);

                  distribution_.remap(update.coverage_map());

                  siblings_.clear();
                  BOOST_FOREACH(id_type const & sibling_id, update.sibling_ids())
                  {
                    siblings_.push_back(vmap_type(sibling_id));
                  }
                }
              }
    
              visit_bins[distribution_.member_id(edge.target())].push_back(
                  edge.target());
            }
          }

        }

        if (edges.size() > 0)
        {
          subgraph_.eager_add_edges(edges).get();
        }

        LVMAP_debug("Visit bins size is %u\n", visit_bins.size());
        future_voids_type outstanding_visits;
        BOOST_FOREACH(target_map_type::value_type const & item, visit_bins)
        {
          LVMAP_debug("\tbin[%u] size is %u\n", 
              item.first, item.second.size());
          outstanding_visits.push_back(
              siblings_[item.first].eager_visits(item.second, depth_1));
          //BOOST_FOREACH(size_type const & target, item.second)
          //{
          //  outstanding_visits.push_back(
          //      siblings_[item.first].eager_visit(target, depth_1));
          //}
        }
        while (outstanding_visits.size() > 0)
        {
          outstanding_visits.back().get();
          outstanding_visits.pop_back();
        }
      }
      else
      {
        future_voids_type outstanding_actions;
        {
          graph_type::edge_iterator_type neighbors(graph_.sync_neighbors(v));
          graph_type::edge_tuples_type const & edges(neighbors.edges());
          size_type const begin(neighbors.begin());
          size_type const end(neighbors.end());
          for (size_type i = begin; i < end; i++)
          {
            edge_tuple_type const & edge((edges)[i]);

            if (!was_seen)
            {
              subgraph_.eager_add_edge(edge).get();
            }
    
            if (0 < depth - 1)
            {
              // Propogate visiting
              size_type const there_index = 
                  distribution_.member_id(edge.target());

              if (distribution_.size() == there_index)
              {
                // Target member is unknown, request extension
                extension_info_type const update = 
                    siblings_[0].eager_request_extension(
                        edge.target()).get();

                {
                  pxgl::util::scoped_use l(use_feb_);

                  distribution_.remap(update.coverage_map());

                  siblings_.clear();
                  BOOST_FOREACH(id_type const & sibling_id, update.sibling_ids())
                  {
                    siblings_.push_back(vmap_type(sibling_id));
                  }
                }
              }
     
              outstanding_actions.push_back(
                  siblings_[distribution_.member_id(edge.target())].eager_visit(
                      edge.target(), depth-1));
            }
          }
        }
        while (outstanding_actions.size() > 0)
        {
          outstanding_actions.back().get();
          outstanding_actions.pop_back();
        }
      }
    }
  }

  void vmap::visits(sizes_type vs, size_type depth)
  {
    not_ready();
    constructed();

    LVMAP_debug("Executing vmap[].visits(%u) with %u\n", depth, vs.size());
   
    BOOST_FOREACH(size_type const & v, vs)
    {
      // Check that max. depth was not reached
      bool was_seen = false;
      bool continue_search = true;
      {
        pxgl::util::scoped_use l(use_feb_);

        if (depth > seen_[v])
        {
          if (0 == seen_[v])
          {
            LVMAP_debug("%u: first coloring vertex %u with depth %u.\n",
                me_.get_lsb(), v, depth);

            // Color the vertex
            seen_[v] = depth;
          }
          else
          {
            was_seen = true;
          }
        }
        else
        {
          continue_search = false;
        }
      }

      LVMAP_debug(
          "Executing vmap[%u].visit(%u): was_seen(%d), continue_search(%d)\n", 
          v, depth, was_seen, continue_search);
      
      if (continue_search)
      {
        size_type const depth_1 = depth - 1;

        // Visit neighbors
        int payload = 0;
        pxgl::rts::get_ini_option(payload, "vmap.visit_payload");
        if (0 < payload)
        {
          edge_tuples_type edges;
          target_map_type visit_bins;

          graph_type::edge_iterator_type neighbors(graph_.sync_neighbors(v));
          graph_type::edge_tuples_type const & n_edges(neighbors.edges());
          size_type const begin(neighbors.begin());
          size_type const end(neighbors.end());
          for (size_type i = begin; i < end; i++)
          {
            edge_tuple_type const & edge(n_edges[i]);
            assert(edge.source() == v);

            if (!was_seen)
            {
              edges.push_back(edge_tuple_type(
                  v, edge.target(), (size_type)edge.weight()));
            }
           
            {
              if (0 < depth_1)
              {
                // Propogate visiting
                size_type const there_index = 
                    distribution_.member_id(edge.target());

                if (distribution_.size() == there_index)
                {
                  // Target member is unknown, request extension
                  extension_info_type const update = 
                      siblings_[0].eager_request_extension(
                          edge.target()).get();

                  {
                    pxgl::util::scoped_use l(use_feb_);

                    distribution_.remap(update.coverage_map());

                    siblings_.clear();
                    BOOST_FOREACH(id_type const & sibling_id, update.sibling_ids())
                    {
                      siblings_.push_back(vmap_type(sibling_id));
                    }
                  }
                }
      
                visit_bins[distribution_.member_id(edge.target())].push_back(
                    edge.target());
              }
            }

          }

          if (edges.size() > 0)
          {
            subgraph_.eager_add_edges(edges).get();
          }

          LVMAP_debug("Visit bins size is %u\n", visit_bins.size());
          future_voids_type outstanding_visits;
          BOOST_FOREACH(target_map_type::value_type const & item, visit_bins)
          {
            LVMAP_debug("\tbin[%u] size is %u\n", 
                item.first, item.second.size());
            outstanding_visits.push_back(
                siblings_[item.first].eager_visits(item.second, depth_1));
          }
          while (outstanding_visits.size() > 0)
          {
            outstanding_visits.back().get();
            outstanding_visits.pop_back();
          }
        }
        else
        {
          future_voids_type outstanding_actions;
          {
            graph_type::edge_iterator_type neighbors(graph_.sync_neighbors(v));
            graph_type::edge_tuples_type const & edges(neighbors.edges());
            size_type const begin(neighbors.begin());
            size_type const end(neighbors.end());
            for (size_type i = begin; i < end; i++)
            {
              edge_tuple_type const & edge(edges[i]);

              if (!was_seen)
              {
                subgraph_.eager_add_edge(edge);
              }
      
              if (0 < depth - 1)
              {
                // Propogate visiting
                size_type const there_index = 
                    distribution_.member_id(edge.target());

                if (distribution_.size() == there_index)
                {
                  // Target member is unknown, request extension
                  extension_info_type const update = 
                      siblings_[0].eager_request_extension(
                          edge.target()).get();

                  {
                    pxgl::util::scoped_use l(use_feb_);

                    distribution_.remap(update.coverage_map());

                    siblings_.clear();
                    BOOST_FOREACH(id_type const & sibling_id, update.sibling_ids())
                    {
                      siblings_.push_back(vmap_type(sibling_id));
                    }
                  }
                }
       
                outstanding_actions.push_back(
                    siblings_[distribution_.member_id(edge.target())].eager_visit(
                        edge.target(), depth-1));
              }
            }
          }
          while (outstanding_actions.size() > 0)
          {
            outstanding_actions.back().get();
            outstanding_actions.pop_back();
          }
        }
      }
    }
  }

  // Note: only call when we can guarantee there are no inflight actions
  void vmap::init(void)
  {
    not_ready();
    constructed();

    {
      pxgl::util::scoped_use l(use_feb_);

      // Set local size
      size_ = seen_.size();

      bool const is_multilocality = (1 < siblings_.size());

      // Signal initialization
      if (is_multilocality)
      {
        future_sizes_type outstanding_signals;

        BOOST_FOREACH(vmap_type const & sibling, siblings_)
        {
          if (sibling.get_gid() != me_)
          {
            outstanding_signals.push_back(
                sibling.eager_signal_init());
          }
        }

        while (outstanding_signals.size() > 0)
        {
          size_ = outstanding_signals.back().get();
          outstanding_signals.pop_back();
        }
      }

      // Finalize initialization
      if (is_multilocality)
      {
        // Collect sibling GIDs
        ids_type sibling_ids(siblings_.size());
        for (size_type i = 0; i < siblings_.size(); i++)
        {
          sibling_ids[i] = siblings_[i].get_gid();
        }

        // Broadcast global information
        BOOST_FOREACH(vmap_type const & sibling, siblings_)
        {
          if (sibling.get_gid() != me_)
          {
            sibling.finalize_init(distribution_, sibling_ids, size_);
          }
        }
      }

      // Construct the updated distribution for this container
      distribution_.finalize_coverage();

      // Set as initialized
      initialized_feb_.set(feb_data_type(1));
      initialized_ = true;
    }
  }

  size_type vmap::signal_init(void)
  {
    pxgl::util::scoped_use l(use_feb_);

    not_ready();
    constructed();

    return seen_.size();
  }

  void vmap::finalize_init(
      distribution_type const & distribution,
      ids_type const & sibling_ids,
      size_type size)
  {
    pxgl::util::scoped_use l(use_feb_);

    not_ready();
    constructed();

    // Finalize the coverage
    distribution_ = distribution;

    for (size_type i = 0; i < sibling_ids.size(); i++)
    {
      siblings_.push_back(vmap_type(sibling_ids[i]));
    }

    // Construct the updated distribution for this container
    distribution_.finalize_coverage();

    // Set global size
    size_ = size;

    // Set this member as constructed
    initialized_feb_.set(feb_data_type(1));
    initialized_ = true;
  }
  
  void vmap::ready(void)
  {
    while (!initialized_)
    {
      LVMAP_ping("Vmap", "Waiting");

      feb_data_type d;
      initialized_feb_.read(d);

      if (1 == d.which())
      {
        error_type e = boost::get<error_type>(d);
        boost::rethrow_exception(e);
      }

      LVMAP_ping("Vmap", "Readied");
    }
  }

  void vmap::not_ready(void)
  {
    assert(!initialized_);
  }

  //////////////////////////////////////////////////////////////////////////////
  // Use

  distribution_type vmap::get_distribution(void)
  {
    ready();

    return distribution_;
  }

  id_type vmap::local_to(size_type index)
  {
    ready();

    return siblings_[index].get_gid();
  }

  size_type vmap::size(void)
  {
    ready();

    return size_;
  }
}}}

////////////////////////////////////////////////////////////////////////////////
HPX_REGISTER_COMPONENT_MODULE();

////////////////////////////////////////////////////////////////////////////////
// Register component factory for property map of Psearch visitors
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    vmap_component_type, 
    vmap);

////////////////////////////////////////////////////////////////////////////////
// Add serialization support for Vmap actions
HPX_REGISTER_ACTION_EX(
    vmap_component_type::wrapped_type::construct_action,
    vmap_construct_action);
HPX_REGISTER_ACTION_EX(
    vmap_component_type::wrapped_type::request_extension_action,
    vmap_request_extension_action);
HPX_REGISTER_ACTION_EX(
    vmap_component_type::wrapped_type::replistruct_action,
    vmap_replistruct_action);
HPX_REGISTER_ACTION_EX(
    vmap_component_type::wrapped_type::constructed_action,
    vmap_constructed_action);
HPX_REGISTER_ACTION_EX(
    vmap_component_type::wrapped_type::visit_action,
    vmap_visit_action);
HPX_REGISTER_ACTION_EX(
    vmap_component_type::wrapped_type::visits_action,
    vmap_visits_action);
HPX_REGISTER_ACTION_EX(
    vmap_component_type::wrapped_type::init_action,
    vmap_init_action);
HPX_REGISTER_ACTION_EX(
    vmap_component_type::wrapped_type::signal_init_action,
    vmap_signal_init_action);
HPX_REGISTER_ACTION_EX(
    vmap_component_type::wrapped_type::finalize_init_action,
    vmap_finalize_init_action);
HPX_REGISTER_ACTION_EX(
    vmap_component_type::wrapped_type::ready_action,
    vmap_ready_action);
HPX_REGISTER_ACTION_EX(
    vmap_component_type::wrapped_type::get_distribution_action,
    vmap_get_distribution_action);
HPX_REGISTER_ACTION_EX(
    vmap_component_type::wrapped_type::local_to_action,
    vmap_local_to_action);
HPX_REGISTER_ACTION_EX(
    vmap_component_type::wrapped_type::size_action,
    vmap_size_action);

////////////////////////////////////////////////////////////////////////////////
// Define Vmap component
HPX_DEFINE_GET_COMPONENT_TYPE(vmap_component_type::wrapped_type);

////////////////////////////////////////////////////////////////////////////////
// Add futures support
HPX_REGISTER_FUTURE(hpx::util::unused_type, unused);
HPX_REGISTER_FUTURE(id_type, id);
HPX_REGISTER_FUTURE(size_type, size);
HPX_REGISTER_FUTURE(extension_info_type, extension_info);

