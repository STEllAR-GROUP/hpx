// Copyright (c) 2010-2011 Dylan Stark
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying 
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <fstream>
#include <sstream>

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

#include <boost/random/linear_congruential.hpp>

#include "../../pxgl/pxgl.hpp"
#include "../../pxgl/xua/control.hpp"
#include "../../pxgl/xua/range.hpp"
#include "../../pxgl/xua/arbitrary_distribution.hpp"
#include "../../pxgl/xua/vector.hpp"
#include "../../pxgl/xua/numeric.hpp"
#include "../../pxgl/xua/vmap_client.hpp"
#include "../../pxgl/graphs/csr_graph.hpp"
#include "../../pxgl/graphs/dynamic_graph_client.hpp"
#include "../../pxgl/graphs/edge_tuple.hpp"

#include "../../pxgl/lcos/have_max.hpp"

#include "../../pxgl/util/apply.hpp"

#include "bc_sssp/bc_sssp_client.hpp"

// Define logging helper
#define LSGAB_LOG_fatal 1
#define LSGAB_LOG_debug 0
#define LSGAB_LOG_info  0
#define LSGAB_LOG_dot   0
#define LSGAB_LOG_ping  0

#if LSGAB_LOG_ping == 1
#define LSGAB_ping(major,minor) YAP_now(major,minor)
#else
#define LSGAB_ping(major,minor) do {} while(0)
#endif

#if LSGAB_LOG_debug == 1
#define LSGAB_debug(str,...) YAPs(str,__VA_ARGS__)
#else
#define LSGAB_debug(str,...) do {} while(0)
#endif

#if LSGAB_LOG_info == 1
#define LSGAB_info(str,...) YAPs(str,__VA_ARGS__)
#else
#define LSGAB_info(str,...) do {} while(0)
#endif

#if LSGAB_LOG_dot == 1
#define LSGAB_dot(str,...) YAPs(str,__VA_ARGS__)
#else
#define LSGAB_dot(str,...) do {} while(0)
#endif

#if LSGAB_LOG_fatal == 1
#define LSGAB_fatal(str,...) YAPs(str,__VA_ARGS__)
#else
#define LSGAB_fatal(str,...) do {} while(0)
#endif

//#define TSGAB_begin(name)
//#define TSGAB_nb_begin(name)
#define TSGAB_begin(name) YAP_T_begin(name)
#define TSGAB_nb_begin(name) YAP_T_nb_begin(name)
//#define TSGAB_end(name)
//#define TSGAB_nb_end(name)
#define TSGAB_end(name) YAP_T_end(name)
#define TSGAB_nb_end(name) YAP_T_nb_end(name)

////////////////////////////////////////////////////////////////////////////////
// Type definitions
typedef unsigned long size_type;
typedef std::vector<size_type> sizes_type;

typedef hpx::naming::id_type id_type;
typedef std::vector<id_type> ids_type;
typedef hpx::naming::gid_type gid_type;
typedef std::vector<gid_type> gids_type;

////////////////////////////////////////////////////////////////////////////////
// Distributions
typedef pxgl::xua::arbitrary_distribution<id_type, pxgl::xua::range>
    arbitrary_distribution_type;

////////////////////////////////////////////////////////////////////////////////
// The edge tuples vector
//
// The arbitrary distribution is used because we do not know a priori how
// many edges will be loaded from the file. Note, the choice of distribution
// acts as a permutation on the edge ordering.
//
typedef pxgl::xua::vector<
    arbitrary_distribution_type,
    pxgl::graphs::server::edge_tuple_type
> edge_container_client_type;
typedef edge_container_client_type::server_type edge_container_server_type;

////////////////////////////////////////////////////////////////////////////////
// The graph
typedef pxgl::graphs::csr_graph<
    edge_container_client_type,
    arbitrary_distribution_type
> graph_type;

typedef edge_container_client_type container_type;
typedef edge_container_server_type container_member_type;
#include "../../pxgl/procs/read_matrix_market.hpp"

typedef int result_type;
typedef boost::exception_ptr error_type;
typedef boost::variant<result_type, error_type> feb_data_type;
static hpx::util::full_empty<feb_data_type> use_feb;

typedef pxgl::lcos::have_max<size_type> have_max_type;
typedef pxgl::lcos::have_max<double> have_max_double_type;

////////////////////////////////////////////////////////////////////////////////
// Kernel 2 types
typedef container_type large_set_type;
typedef container_member_type large_set_member_type;

////////////////////////////////////////////////////////////////////////////////
// Kernel 3 types
typedef pxgl::xua::vector<
    arbitrary_distribution_type,
    id_type
> fco_container_type;
typedef fco_container_type::server_type fco_container_member_type;

typedef fco_container_type subgraphs_type;
typedef fco_container_member_type subgraphs_member_type;

typedef pxgl::graphs::dynamic_graph subgraph_type;
typedef pxgl::graphs::server::dynamic_graph subgraph_member_type;

typedef pxgl::xua::vmap vmap_type;
typedef pxgl::xua::server::vmap vmap_member_type;

typedef hpx::lcos::promise<id_type> future_id_type;
typedef std::vector<future_id_type> future_ids_type;

////////////////////////////////////////////////////////////////////////////////
// Kernel 4 types

typedef pxgl::xua::vector<
    arbitrary_distribution_type,
    size_type
> size_container_type;
typedef size_container_type::server_type size_container_member_type;

typedef size_container_type bc_vertices_type;

typedef pxgl::xua::numeric<
    arbitrary_distribution_type,
    double
> double_container_type;
typedef double_container_type::server_type double_container_member_type;

typedef double_container_type bc_scores_type;
typedef double_container_member_type bc_scores_member_type;

typedef hpx::lcos::promise<hpx::util::unused_type> future_void_type;
typedef std::vector<future_void_type> future_voids_type;

typedef examples::sgab::bc_sssp bc_sssp_type;

////////////////////////////////////////////////////////////////////////////////
// Kernel 2: Filter edges
void filter_edges_part(
    graph_type::edge_tuples_type const * edges_ptr,
    size_type const start,
    size_type const stop,
    graph_type::edge_tuples_type * local_edges,
    sizes_type * max_weights)
{
  graph_type::edge_tuples_type part_edges;
  size_type part_max = 0;
  for (size_type i = start; i < stop; i++)
  {
    graph_type::edge_tuple_type const & e = (*edges_ptr)[i];

    if (e.weight() > part_max)
    {
      part_edges.clear();
      part_edges.push_back(e);

      part_max = e.weight();
    }
    else if (e.weight() == part_max)
    {
      part_edges.push_back(e);
    }
  }

  // Merge local info
  {
    pxgl::util::scoped_use l(use_feb);

    size_type max_weight = (*max_weights)[0];

    if (max_weight < part_max)
    {
      max_weight = part_max;
      (*max_weights)[0] = max_weight;

      local_edges->clear();
      local_edges->insert(
          local_edges->end(), part_edges.begin(), part_edges.end());
    }
    else if (max_weight == part_max)
    {
      local_edges->insert(
          local_edges->end(), part_edges.begin(), part_edges.end());
    }
  }
}

typedef hpx::actions::plain_action5<
      graph_type::edge_tuples_type const *,
      size_type const,
      size_type const,
      graph_type::edge_tuples_type *,
      sizes_type *,
    &filter_edges_part
> filter_edges_part_action;
HPX_REGISTER_PLAIN_ACTION(filter_edges_part_action);

int filter_edges(
    id_type const & graph_id, 
    id_type const & large_set_id, 
    id_type const & is_max_id)
{
  YAP_now("K2", "Starting filter edges");
  LSGAB_ping("Filter edges", "Start");

  use_feb.set(feb_data_type(1));

  // Create client-side views of each FCO
  graph_type graph(graph_id);
  large_set_type large_set(large_set_id);
  have_max_type is_max(is_max_id);

  // Find local collection of maximal-weight edges
  size_type max_weight = 0;

  graph_type::edge_tuples_type local_edges;
  {
    // Get edges info
    graph_type::edge_tuples_type const * edges_ptr = graph.edges();
    size_type const num_edges = edges_ptr->size();

    // Calculate partitions
    size_type num_partitions = 1; // Default: no partitioning
    pxgl::rts::get_ini_option(num_partitions, "sgab.k2.num_partitions");

    size_type chunk_size = num_edges / num_partitions;
    size_type extra_size = num_edges % num_partitions;

    id_type const here(hpx::get_runtime().get_process().here());

    sizes_type max_weights(1,0);

    // Spawn and sync. on partitions 
    future_voids_type outstanding_actions;
    for (size_type i = 0; i < extra_size; i++)
    {
      size_type const start = i * (chunk_size + 1);
      size_type const stop = start + (chunk_size + 1);

      outstanding_actions.push_back(
          hpx::lcos::eager_future<filter_edges_part_action>(
              here, edges_ptr, start, stop, &local_edges, &max_weights));
    }

    size_type const prev_size = extra_size * (chunk_size + 1);
    for (size_type i = 0; i < num_partitions - extra_size; i++)
    {
      size_type const start = prev_size + (i * chunk_size);
      size_type const stop = start + chunk_size;

      outstanding_actions.push_back(
          hpx::lcos::eager_future<filter_edges_part_action>(
              here, edges_ptr, start, stop, &local_edges, &max_weights));
    }

    while (outstanding_actions.size() > 0)
    {
      outstanding_actions.back().get();
      outstanding_actions.pop_back();
    }

    max_weight = max_weights[0];
  }
  LSGAB_debug("Local max weight: %u (%u)\n", max_weight, local_edges.size());

  // Trigger is-max LCO with local maximum value:
  //   If local max. is global max, then add collection to large-set
  //   Else, add an empty collection to large-set
  bool mine_is_max;
  mine_is_max = is_max.signal(max_weight);
  if (!mine_is_max)
  {
    local_edges.clear();
  }
  LSGAB_ping("Filter edges", "Received max weight");

  large_set.init(local_edges);

  LSGAB_ping("Filter edges", "Stop");
  return 0;
}

typedef hpx::actions::plain_result_action3<
    int,
      id_type const &, 
      id_type const &, 
      id_type const &,
    &filter_edges
> filter_edges_action;
HPX_REGISTER_PLAIN_ACTION(filter_edges_action);

////////////////////////////////////////////////////////////////////////////////
// Kernel 3: Test/debug subgraph instances
size_type sum_all_subgraph_sizes(id_type const & subgraphs_id)
{
  size_type local_sum = 0;

  {
    subgraphs_type subgraphs(subgraphs_id);

    subgraphs_type::items_type const * local_items_ptr = subgraphs.items();
    BOOST_FOREACH(id_type subgraph_id, (*local_items_ptr))
    {
      subgraph_type const subgraph(subgraph_id);
      local_sum += subgraph.size();
    }
  }

  return local_sum;
}

typedef hpx::actions::plain_result_action1<
    size_type,
    id_type const &,
    sum_all_subgraph_sizes
> sum_all_subgraph_sizes_action;
HPX_REGISTER_PLAIN_ACTION(sum_all_subgraph_sizes_action);

////////////////////////////////////////////////////////////////////////////////
// Kernel 3: Checky ready statuses 
void check_ready_statuses(id_type const & subgraphs_id)
{
  subgraphs_type subgraphs(subgraphs_id);
  ids_type * items_ptr = subgraphs.items(); // Implicitly waits on member

  BOOST_FOREACH(id_type const & subgraph_id, (*items_ptr))
  {
    subgraph_type const subgraph(subgraph_id);
    subgraph.ready();
  }
}

typedef hpx::actions::plain_action1<
        id_type const &,
    &check_ready_statuses
> check_ready_statuses_action;
HPX_REGISTER_PLAIN_ACTION(check_ready_statuses_action);

////////////////////////////////////////////////////////////////////////////////
// Kernel 3: Find subgraphs
id_type find_subgraph(
    id_type const graph_id,
    large_set_type::item_type const edge,
    size_type const depth)
{
  subgraph_type subgraph;
  {
    graph_type graph(graph_id);

    id_type const here(hpx::get_runtime().get_process().here());
    graph_type::distribution_type const graph_distribution =
        graph.get_distribution();

    // Create a subgraph
    subgraph.create(here);
    subgraph.set_distribution(subgraph.get_gid(), graph_distribution);

    LSGAB_debug("Creating a new vmap.\n",0);

    // Create a property map of Psearch visitors
    // Note: we create it local to the target vertex
    vmap_type vmap;
    vmap.create(graph_distribution.locale(edge.target()));
    vmap.construct(vmap.get_gid(), graph_distribution, 
        graph.get_gid(), subgraph.get_gid());

    // Add start edge to subgraph
    subgraph.eager_add_edge(edge).get();

    // Start off parallel search
    vmap.sync_visit(edge.target(), depth);
    vmap.init(); // Technically not needed, since we will throw away vmap

    LSGAB_debug("%u: vmap created with size %u.\n", 
                vmap.get_gid().get_lsb(), vmap.size());

    // Initialize the subgraph - this is an async. operation: only call "use"
    // interface methods after, and make sure to guarantee that initialization
    // has fully completed before letting HPX shut down.
    subgraph.init();
  }

  LSGAB_fatal("Found subgraph with %lu vertices and %lu edges.\n",
             subgraph.order(), subgraph.size());

  return subgraph.get_gid();
}

typedef hpx::actions::plain_result_action3<
    id_type,
        id_type const, 
        large_set_type::item_type const, 
        size_type const,
    &find_subgraph
> find_subgraph_action;
HPX_REGISTER_PLAIN_ACTION(find_subgraph_action);

////////////////////////////////////////////////////////////////////////////////
// Kernel 3: Extract subgraphs
void extract_subgraphs(
    id_type const & large_set_id,
    id_type const & graph_id,
    id_type const & subgraphs_id)
{
  LSGAB_ping("Extract subgraphs", "Start");

  subgraphs_type subgraphs(subgraphs_id);

  ids_type local_subgraphs;
  {
    // Get depth parameter from configuration file
    size_type depth = 3;
    pxgl::rts::get_ini_option(depth, "sgab.k3.depth");
    assert(0 < depth);

    large_set_type::items_type const * items_ptr(
        large_set_type(large_set_id).items());
    size_type const num_subgraphs = items_ptr->size();

    LSGAB_fatal("Extracting %lu subgraphs to depth %lu.\n", num_subgraphs, depth);

    // Find new subgraphs concurrently
    {
      bool serialize_find_subgraphs;
      pxgl::rts::get_ini_option(
          serialize_find_subgraphs, "sgab.k3.serialize_find_subgraphs");
     
      if (serialize_find_subgraphs)
      {
        BOOST_FOREACH(large_set_type::item_type const & edge, (*items_ptr))
        {
          local_subgraphs.push_back(
              hpx::lcos::eager_future<find_subgraph_action>(
                  large_set_id, graph_id, edge, depth).get());
        }

        LSGAB_fatal("Finished finding subgraph with depth %lu.\n",depth);
      }
      else
      {
        future_ids_type outstanding_actions;
        BOOST_FOREACH(large_set_type::item_type const & edge, (*items_ptr))
        {
          outstanding_actions.push_back(
              hpx::lcos::eager_future<find_subgraph_action>(
                  large_set_id, graph_id, edge, depth));
        }

        // Gather new subgraph GIDs
        while (outstanding_actions.size() > 0)
        {
          local_subgraphs.push_back(outstanding_actions.back().get());
          outstanding_actions.pop_back();
        }
      }
    }
  }

  // Initialize local subgraphs
  subgraphs.init(local_subgraphs);

  LSGAB_ping("Extract subgraphs", "Stop");
}

typedef hpx::actions::plain_action3<
        id_type const &, 
        id_type const &, 
        id_type const &,
    &extract_subgraphs
> extract_subgraphs_action;
HPX_REGISTER_PLAIN_ACTION(extract_subgraphs_action);

////////////////////////////////////////////////////////////////////////////////
// Kernel 4: Select BC vertices

void find_max_bc_scores(
    id_type const & bc_scores_id, 
    id_type const & bc_max_vertices_id,
    id_type const & is_max_id)
{
  bc_scores_type const bc_scores(bc_scores_id);
  have_max_double_type const is_max(is_max_id);
  bc_vertices_type bc_max_vertices(bc_max_vertices_id);

  bc_vertices_type::items_type indices;
  {
    bc_scores_type::item_type max_score =
        std::numeric_limits<bc_scores_type::item_type>::min();

    bc_scores_type::items_type const * scores_ptr = bc_scores.items();
    size_type const num_scores = scores_ptr->size();
    for (size_type i = 0; i < num_scores; i++)
    {
      bc_scores_type::item_type const score = (*scores_ptr)[i];

      if (score >= max_score)
      {
        if (score > max_score)
        {
          indices.clear();
          indices.push_back(i);

          max_score = score;
        }
        else if (score == max_score)
        {
          indices.push_back(i);
        }
      }
    }

    if (!is_max.signal(max_score))
    {
      indices.clear();
    }
  }

  bc_max_vertices.init(indices);
}

typedef hpx::actions::plain_action3<
        id_type const &,
        id_type const &,
        id_type const &,
    &find_max_bc_scores
> find_max_bc_scores_action;
HPX_REGISTER_PLAIN_ACTION(find_max_bc_scores_action);

inline void async_find_max_bc_scores(
    bc_scores_type const & bc_scores,
    bc_vertices_type const & bc_max_vertices)
{
  id_type const here = hpx::get_runtime().get_process().here();

  have_max_double_type is_max;
  is_max.create(here);
  is_max.construct(bc_max_vertices.get_distribution().size());

  pxgl::xua::for_each_aligned_client1<find_max_bc_scores_action>(
      bc_scores, bc_max_vertices, is_max.get_gid());
}

void score_bc_vertices(
    id_type const & graph_id, 
    id_type const & bc_vertices_id,
    id_type const & bc_scores_id)
{
  graph_type const graph(graph_id);
  bc_scores_type const bc_scores(bc_scores_id);

  bc_vertices_type const bc_vertices(bc_vertices_id);
  id_type const here = hpx::get_runtime().get_process().here();

  bc_vertices_type::items_type const & vertices = *(bc_vertices.items());
  size_type const num_vertices = vertices.size();
  std::vector<bc_sssp_type> bc_sssps(num_vertices);
  for (size_type i = 0; i < num_vertices; i++)
  {
    bc_sssps[i].create(here);
    bc_sssps[i].async_instantiate(graph, bc_scores);

    bc_sssps[i](vertices[i]);
  }

  for (size_type i = 0; i < num_vertices; i++)
  {
    bc_sssps[i].ready_all();
    bc_sssps[i].ended();
  }
}

typedef hpx::actions::plain_action3<
        id_type const &,
        id_type const &,
        id_type const &,
    &score_bc_vertices
> score_bc_vertices_action;
HPX_REGISTER_PLAIN_ACTION(score_bc_vertices_action);

inline void async_score_bc_vertices(
    graph_type const & graph,
    bc_vertices_type const & bc_vertices,
    bc_scores_type const & bc_scores)
{
  // Parallelized
  pxgl::xua::blocking_for_each_aligned_client<score_bc_vertices_action>(
      graph, bc_vertices, bc_scores);

  // Set bc_scores as initialized
  {
    bc_scores_type::items_type empty_items;
    pxgl::xua::for_each_comp<
        bc_scores_type, bc_scores_member_type::init_action
    >(bc_scores.get_gid(), empty_items);
  }
}

void select_bc_vertices(
    id_type const & graph_id, 
    id_type const & bc_vertices_id)
{
  graph_type const graph(graph_id);
  bc_vertices_type bc_vertices(bc_vertices_id);

  bc_vertices_type::items_type vertices;
  {
    size_type scale = 0;
    pxgl::rts::get_ini_option(scale, "sgab.scale");
    assert(0 < scale);

    size_type k4_approx = scale/2;
    pxgl::rts::get_ini_option(k4_approx, "sgab.k4.k4_approx");
    assert(k4_approx <= scale);

    LSGAB_info("K4 run with scale %u and k4_approx %u.\n",
        scale, k4_approx);

    if (k4_approx == scale)
    {
      sizes_type const * vertices_ptr = graph.sync_vertices();
      vertices.insert(vertices.begin(), vertices_ptr->begin(), vertices_ptr->end());
    }
    else
    {
      sizes_type const * vertices_ptr = graph.sync_vertices();
      size_type const local_order = vertices_ptr->size();

      size_type const local_portion = 
          (size_type)((local_order/(graph.order()*1.))*(1<<k4_approx));
      LSGAB_info("Selecting %u vertices on this locality.\n", local_portion);

      // Setup random number generator
      boost::rand48 random_number;
      random_number.seed(hpx::get_runtime().get_process().here().get_msb());

      uint32_t const random_min = random_number.min();
      uint32_t const random_diff = random_number.max() - random_min;

      while (vertices.size() < local_portion)
      {
        while (vertices.size() < local_portion)
        {
          // Select random entry in local 
          size_type const random_v = 
              (size_type)(((random_number()-random_min) / ((random_diff)*1.0)) 
                  * local_order);
          
          vertices.push_back((*vertices_ptr)[random_v]);
        }

        sort(vertices.begin(), vertices.end());
        sizes_type::iterator new_end = unique(vertices.begin(), vertices.end());
        vertices.erase(new_end, vertices.end());
      }
    }
  }

  bc_vertices.init(vertices);
}

typedef hpx::actions::plain_action2<
        id_type const &,
        id_type const &,
    &select_bc_vertices
> select_bc_vertices_action;
HPX_REGISTER_PLAIN_ACTION(select_bc_vertices_action);

inline void async_select_bc_vertices(
    graph_type const & graph,
    bc_vertices_type const & bc_vertices)
{
  pxgl::xua::for_each_aligned_client<select_bc_vertices_action>(
      graph, bc_vertices);
}

////////////////////////////////////////////////////////////////////////////////
// Main
int hpx_main(boost::program_options::variables_map &vm)
{
  LSGAB_ping("Main", "Start");

  // Setup for selecting kernels to run from parameter file
  bool run_k2 = true;
  pxgl::rts::get_ini_option(run_k2, "sgab.run_k2");
  bool run_k3 = true;
  pxgl::rts::get_ini_option(run_k3, "sgab.run_k3");
  bool run_k4 = true;
  pxgl::rts::get_ini_option(run_k4, "sgab.run_k4");

  hpx::util::high_resolution_timer main_timer;

  std::stringstream measurements;

  {
    size_type scale = 0;
    pxgl::rts::get_ini_option(scale, "sgab.scale");
    measurements << scale << " ";

    int num_os_threads = -1;
    pxgl::rts::get_ini_option(num_os_threads, "rts.num_os_threads");
    measurements << num_os_threads << " ";

    std::string queue = "invalid";
    pxgl::rts::get_ini_option(queue, "rts.queue");
    measurements << queue << " ";

    int smp_mode = -1;
    pxgl::rts::get_ini_option(smp_mode, "rts.smp_mode");
    measurements << smp_mode << " ";
  }

  {
    hpx::process my_proc(hpx::get_runtime().get_process());
    id_type here = my_proc.here();
    ids_type localities = my_proc.localities();

    graph_type graph;
    {
      // Process application-specific command-line options
      std::string input_file;
      hpx::get_option(vm, "input_file", input_file, "sgab.input_file");
      assert(!input_file.empty());

      // Log SGAB banner
      LSGAB_info("Scalable Graph Analysis Benchmark v1.0\n",0);
      LSGAB_info("Input file: %s\n", input_file.c_str());

      //////////////////////////////////////////////////////////////////////////
      // Prepare list of edge tuples
      LSGAB_ping("Load edge data", "Start");
      container_type edge_tuples;
      {
        hpx::util::high_resolution_timer timer;

        // Construct the edge list
        edge_tuples.create(here);
        pxgl::xua::arbitrary_distribution<id_type, pxgl::xua::range>
            et_dist(localities);
        edge_tuples.construct(edge_tuples.get_gid(), et_dist);

        // Read in the edge list data
        pxgl::xua::for_each<
            container_type, read_matrix_market_action
        >(edge_tuples.get_gid(), input_file);

        // This provides phasing required by SGAB spec.
        edge_tuples.ready_all();

        LSGAB_fatal("Completed SDG in %f sec\n", timer.elapsed());
      }
      LSGAB_ping("Load edge data", "Stop");

      // Check sanity
      LSGAB_info("Edge tuples list initialzed with %u edges\n", 
                 edge_tuples.size());

      //////////////////////////////////////////////////////////////////////////
      // Kernel 1
      LSGAB_ping("Kernel 1", "Start");
      {
        hpx::util::high_resolution_timer timer;

        // Build distribution for graph
        pxgl::xua::arbitrary_distribution<id_type, pxgl::xua::range>
            arb_dist(localities);

        // Construct graph
        graph.create(here);
        graph.construct(graph.get_gid(), arb_dist);

        // Initialize graph
        graph.init(edge_tuples.get_gid());
        
        graph.ready_all(); // This provides phasing required by SGAB spec.

        measurements << timer.elapsed() << " ";
        LSGAB_fatal("Completed Kernel1 in %f sec\n", timer.elapsed());
      }
      LSGAB_ping("Kernel 1", "Stop");

      LSGAB_info("Graph initialized with %u vertices and %u edges\n",
                 graph.order(), graph.size());
    }

    if (run_k2 || run_k3)
    {
      size_type num_partitions = 0;
      pxgl::rts::get_ini_option(num_partitions, "sgab.k2.num_partitions");
      measurements << num_partitions << " ";

      //////////////////////////////////////////////////////////////////////////
      // Kernel 2
      LSGAB_ping("Kernel 2", "Start");
      large_set_type large_set;
      {
        hpx::util::high_resolution_timer timer;

        // Construct the large set (edge list)
        large_set.create(here);
        pxgl::xua::arbitrary_distribution<id_type, pxgl::xua::range>
            ls_dist(localities);
        large_set.construct(large_set.get_gid(), ls_dist);

        // Setup have-max LCO
        have_max_type is_max;
        is_max.create(here);
        is_max.construct(ls_dist.size());

        // Generate large set
        YAP_now("K2", "Spawning filter edges");
        pxgl::xua::for_each_aligned_client1<
            filter_edges_action,
            graph_type, large_set_type
        >(graph, large_set, is_max.get_gid());

        // Provide phasing required by SGAB spec.
        LSGAB_fatal("Checking large_set.ready() after %f sec\n", 
            timer.elapsed());
        large_set.ready();

        measurements << timer.elapsed() << " ";
        LSGAB_fatal("Completed Kernel2 in %f sec\n", timer.elapsed());

        // Provide full phasing for HPX shutdown
        pxgl::xua::blocking_for_each_comp<
            large_set_type, 
            large_set_member_type::ready_action
        >(large_set.get_gid());
      }
      LSGAB_ping("Kernel 2", "Stop");
      
      LSGAB_info("Large set has %u items.\n", large_set.size());

      if (run_k3)
      {
        ////////////////////////////////////////////////////////////////////////
        // Kernel 3
        LSGAB_ping("Kernel 3", "Start");
        subgraphs_type subgraphs;
        {
          hpx::util::high_resolution_timer timer;

          subgraphs.create(here);
          subgraphs.construct(subgraphs.get_gid(),large_set.get_distribution());

          // Generate tasks to extract subgraphs from each large set member
          pxgl::xua::for_each_aligned<
              large_set_type, graph_type, subgraphs_type,
              extract_subgraphs_action
          >(large_set.get_gid(), graph.get_gid(), subgraphs.get_gid());
          LSGAB_ping("Main", "Finished generating extract_subgraph tasks");

          measurements << subgraphs.size() << " ";
          LSGAB_fatal("Found %lu subgraphs.\n", subgraphs.size());

          // Provide phasing required by SGAB spec.
          pxgl::xua::blocking_for_each<
              subgraphs_type,
              check_ready_statuses_action
          >(subgraphs.get_gid());

          measurements << timer.elapsed() << " ";
          LSGAB_fatal("Completed Kernel3 in %f sec\n", timer.elapsed());

          {
            int depth;
            pxgl::rts::get_ini_option(depth, "sgab.k3.depth");
            measurements << depth << " ";

            bool sfs;
            pxgl::rts::get_ini_option(sfs, "sgab.k3.serialize_find_subgraphs");
            measurements << sfs << " ";

            int payload;
            pxgl::rts::get_ini_option(payload, "vmap.visit_payload");
            measurements << payload << " ";
          }
        }
        LSGAB_ping("Kernel 3", "Stop");

        // Test subgraphs from Kernel 3
        {
          size_type sum_of_all_subgraphs = 0;
          sum_of_all_subgraphs =
              pxgl::xua::blocking_reduce<
                  subgraphs_type,
                  sum_all_subgraph_sizes_action
              >(subgraphs.get_gid(), sum_of_all_subgraphs);

          measurements << sum_of_all_subgraphs << " ";
          LSGAB_info("Sum of all subgraphs is %u.\n", sum_of_all_subgraphs);
        }
      }
    }
   
    if (run_k4)
    {
      //////////////////////////////////////////////////////////////////////////
      // Kernel 4
      LSGAB_ping("Kernel 4", "Start");
      {
        hpx::util::high_resolution_timer timer;

        bc_vertices_type bc_vertices;
        {
          bc_vertices.create(here);
          bc_vertices.construct(bc_vertices.get_gid(),graph.get_distribution());

          async_select_bc_vertices(graph, bc_vertices);

          LSGAB_info("BC: selected %u vertices.\n", bc_vertices.size());
        }

        bc_scores_type bc_scores;
        {
          bc_scores.create(here);
          bc_scores.construct(
              bc_scores.get_gid(), bc_vertices.get_distribution());

          async_score_bc_vertices(graph, bc_vertices, bc_scores);
        }

        bc_vertices_type bc_max_vertices;
        {
          bc_max_vertices.create(here);
          bc_max_vertices.construct(
              bc_max_vertices.get_gid(), bc_vertices.get_distribution());

          async_find_max_bc_scores(bc_scores, bc_max_vertices);
          LSGAB_info("BC: num. max scores is %u.\n", bc_max_vertices.size());
        }

        measurements << timer.elapsed() << " ";
        LSGAB_fatal("Completed Kernel4 in %f sec\n", timer.elapsed());

        // Note: sync in reverse order of instantiation to reduce chance of
        // suspending this thread
        bc_max_vertices.ready_all();
        bc_scores.ready_all();
        bc_vertices.ready_all();

      }
    }
  }

  LSGAB_fatal("Completed SGAB benchmark in %f sec\n", main_timer.elapsed());

  fprintf(stderr, "RUN: %s\n", measurements.str().c_str());

  LSGAB_ping("Main", "Stop");

  hpx::finalize();

  return 0;
}

////////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
  boost::program_options::options_description
      desc_commandline("Usage: sgab [hpx_options]");

  desc_commandline.add_options()
      ("input_file,f", boost::program_options::value<std::string>(),
       "the input data file")
      ;

  int retcode = hpx::init(desc_commandline, argc, argv);
  return retcode;
}

