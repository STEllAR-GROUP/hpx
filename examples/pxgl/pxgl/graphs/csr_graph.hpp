// Copyright (c) 2010-2011 Dylan Stark
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying 
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(PXGL_GRAPHS_CSR_GRAPH_20100817T2013)
#define PXGL_GRAPHS_CSR_GRAPH_20100817T2013

#include <algorithm>
#include <boost/unordered_map.hpp>

#include <hpx/hpx.hpp>
#include <hpx/hpx_fwd.hpp>

#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>

#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>

#include <hpx/runtime/components/client_base.hpp>

#include "../../pxgl/pxgl.hpp"
#include "../../pxgl/util/hpx.hpp"
#include "../../pxgl/util/component.hpp"
#include "../../pxgl/util/scoped_use.hpp"

#include "../../pxgl/graphs/signal_value.hpp"

// Define logging helper
#define LCSR_LOG_fatal 1
#define LCSR_LOG_info  0
#define LCSR_LOG_debug 0
#define LCSR_LOG__ping 0

#if LCSR_LOG_ping == 1
#  define LCSR_ping(major,minor) YAP_now(major,minor)
#else
#  define LCSR_ping(major,minor) do {} while(0)
#endif

#if LCSR_LOG_debug == 1
#define LCSR_debug(str,...) YAPs(str,__VA_ARGS__)
#else
#define LCSR_debug(str,...) do {} while(0)
#endif

#if LCSR_LOG_info == 1
#define LCSR_info(str,...) YAPs(str,__VA_ARGS__)
#else
#define LCSR_info(str,...) do {} while(0)
#endif

#if LCSR_LOG_fatal == 1
#define LCSR_fatal(str,...) YAPs(str,__VA_ARGS__)
#else
#define LCSR_fatal(str,...) do {} while(0)
#endif

////////////////////////////////////////////////////////////////////////////////
// Prototypes
namespace pxgl { 
  namespace graphs {
    template <typename EdgeTuples, typename Distribution>
    class csr_graph;
  }
}

////////////////////////////////////////////////////////////////////////////////
namespace pxgl { namespace graphs { namespace server {

  //////////////////////////////////////////////////////////////////////////////
  // Neighbors types
  template <typename EdgeTuples, typename Size>
  struct edge_iterator
  {
    typedef EdgeTuples edge_tuples_type;
    typedef typename edge_tuples_type::value_type edge_tuple_type;

    typedef Size size_type;

    edge_iterator()
    {}

    edge_iterator(
        EdgeTuples * edges, 
        Size begin, 
        Size end)
      : edges_(edges),
        begin_(begin),
        end_(end)
    {}

    EdgeTuples const & edges(void) const
    {
      return *edges_;
    }

    size_type const begin(void) const
    {
      return begin_;
    }

    size_type const end(void) const
    {
      return end_;
    }

  private:
    // Serialization support
    friend class boost::serialization::access;

    template <typename Archive>
    void serialize(Archive & ar, const unsigned int)
    {
      ar & edges_ & begin_ & end_;
    }

    EdgeTuples * edges_;
    size_type begin_;
    size_type end_;
  };

  template <typename VertexId, typename Weight>
  struct adjacency
  {
    adjacency()
    {}

    adjacency(VertexId target, Weight weight)
      : target_(target),
        weight_(weight)
    {}

    bool operator==(adjacency const & rhs) const
    {
      return rhs.target() == target_;
    }

    bool operator<(adjacency const & rhs) const
    {
      return rhs.target() < target_;
    }

    VertexId const & target(void) const
    {
      return target_;
    }

    Weight const & weight(void) const
    {
      return weight_;
    }

  private:
    // Serialization support
    friend class boost::serialization::access;

    template <typename Archive>
    void serialize(Archive & ar, const unsigned int)
    {
      ar & target_ & weight_;
    }

    VertexId target_;
    Weight weight_;
  };

  //////////////////////////////////////////////////////////////////////////////
  // Server interface
  template <typename EdgeTuples, typename Distribution>
  class csr_graph
    : public HPX_MANAGED_BASE_2(csr_graph, EdgeTuples, Distribution)
  {
  public:
    enum actions
    {
      // Construction
      csr_graph_construct,
      csr_graph_replicate,
      // Initialization
      csr_graph_init,
      csr_graph_init_local,
      csr_graph_signal_init,
      csr_graph_finalize_init,
      csr_graph_aligned_init,
      csr_graph_add_local_vertices,
      // Use
      csr_graph_ready,
      csr_graph_ready_all,
      csr_graph_order,
      csr_graph_size,
      csr_graph_vertices,
      csr_graph_edges,
      csr_graph_neighbors,
      csr_graph_get_distribution,
      csr_graph_local_to,
    };

    ////////////////////////////////////////////////////////////////////////////
    // Associated types
    typedef hpx::naming::id_type id_type;
    typedef std::vector<id_type> ids_type;

    typedef hpx::lcos::promise<int> future_int_type;
    typedef std::vector<future_int_type> future_ints_type;

    typedef unsigned long size_type;
    typedef std::vector<size_type> sizes_type;

    typedef double weight_type;
    typedef std::vector<weight_type> weights_type;

    typedef hpx::lcos::promise<size_type> future_size_type;
    typedef std::vector<future_size_type> future_sizes_type;

    typedef hpx::lcos::promise<signal_value_type> future_signal_value_type;
    typedef std::vector<future_signal_value_type> future_signal_values_type;
      
    typedef Distribution distribution_type;

    typedef typename EdgeTuples::item_type edge_tuple_type;
    typedef typename EdgeTuples::items_type edge_tuples_type;

    typedef pxgl::graphs::csr_graph<EdgeTuples, distribution_type> 
        csr_graph_client_type;

    typedef pxgl::graphs::server::edge_iterator<edge_tuples_type, size_type>
        edge_iterator_type;
    typedef pxgl::graphs::server::adjacency<size_type, double>  adjacency_type;
    typedef std::vector<adjacency_type> adjacencies_type;
    typedef boost::unordered_map<size_type, adjacencies_type> map_type;
    typedef boost::unordered_map<size_type, size_type> vertex_map_type;

    csr_graph()
      : order_(0),
        size_(0),
        me_(hpx::naming::invalid_id),
        here_(hpx::get_runtime().get_process().here()),
        siblings_(0),
        local_edge_list_(0),
        remote_updates_(0),
        local_map_(0),
        vertices_(0),
        index_(0),
        vertex_map_(0),
        edges_(0),
        constructed_(false),
        initialized_(false)
    {
      use_feb_.set(feb_data_type(1));
    }

    ////////////////////////////////////////////////////////////////////////////
    // Construction interface

    ///
    /// \brief Construct a distributed graph
    ///
    /// This is the process for building a decentralized distributed graph
    /// structure.
    ///
    /// \param me the name of this local portion of the graph. 
    /// \param distribution the distribution to use for the graph.
    ///
    /// \pre Only a local graph structure exists, with an associated name
    ///      (GID), but no distribution.
    /// \post The distributed graph structure has been constructed, but not
    ///       initialized.
    ///
    /// \note Parameter \a me is required because an HPX component action
    ///       cannot query for the GID of the component on which it is acting.
    ///
    void construct(
        id_type const & me, 
        distribution_type const & distribution)
    {
      {
        pxgl::util::scoped_use l(use_feb_);

        // Check that the graph was not already constructed
        assert(!constructed_);

        // Set name and distribution for this component
        me_ = me;
        distribution_ = distribution;

        // Create a new graph component on each locality (excluding this one)
        // that is covered by the graph distribution.
        typedef typename distribution_type::locality_ids_type locality_ids_type;
        locality_ids_type const & locales = distribution_.coverage();
        size_type const extent = locales.size();

        siblings_ = std::vector<csr_graph_client_type>(extent);
        for (size_type i = 0; i < extent; i++)
        {
          if (locales[i] != here_)
          {
            siblings_[i].create(locales[i]);
          }
          else
          {
            siblings_[i] = csr_graph_client_type(me_);
          }
        }
     
        // Collect ids for all sibling components comprising this distributed
        // graph.
        ids_type sibling_ids(extent);
        for (size_type i =0; i < extent; i++)
        {
          sibling_ids[i] = siblings_[i].get_gid();
        }

        // Finally, initiate construction of sibling components
        for (size_type i =0; i < extent; i++)
        {
          if (locales[i] != here_)
          {
            siblings_[i].replicate(distribution, sibling_ids);
          }
        }

        // Set this local component as constructed
        constructed_feb_.set(feb_data_type(1));
        constructed_ = true;
      }
    }

    typedef hpx::actions::action2<
        csr_graph, 
        csr_graph_construct, 
            id_type const &, 
            distribution_type const &, 
        &csr_graph::construct
    > construct_action;

    ///
    /// \brief Construct a local portion of a distributed graph
    ///
    /// This procedure constructs a local portion of the graph.
    ///
    /// \param distribution the distribution of the graph.
    /// \param sibling_ids the collection of names of all local portions of
    ///                    the distributed graph
    ///
    /// \post This local portion of the graph has been constructed, but not
    ///       initialized.
    ///
    void replicate(
        distribution_type const & distribution, 
        ids_type const & sibling_ids)
    {
      {
        pxgl::util::scoped_use l(use_feb_);

        // Check that the graph was not already constructed
        assert(!constructed_);

        // Set local information for this component
        distribution_ = distribution;

        typedef typename distribution_type::locality_ids_type locality_ids_type;
        locality_ids_type const & locales = distribution_.coverage();
        size_type const extent = locales.size();

        siblings_ = std::vector<csr_graph_client_type>(extent);
        for (size_type i = 0; i < extent; i++)
        {
          siblings_[i] = csr_graph_client_type(sibling_ids[i]);

          if (locales[i] == here_)
          {
            me_ = sibling_ids[i];
          }
        }

        // Set this local component as constructed
        constructed_feb_.set(feb_data_type(1));
        constructed_ = true;
      }
    }

    typedef hpx::actions::action2<
        csr_graph, 
        csr_graph_replicate, 
            distribution_type const &, 
            ids_type const &,
        &csr_graph::replicate
    > replicate_action;

    ////////////////////////////////////////////////////////////////////////////
    // Initialization interface

  private:

    ///
    /// \brief Initialize the graph from an aligned edge container
    ///
    /// This assumes that the graph and edge container are aligned, so no
    /// redistribution of edges is required.
    ///
    /// \param edges name ofa  distributed container of edges
    ///
    void initialize_aligned(id_type const & edges_id)
    {
      // Tell each local graph to initialize with corresponding local edge
      // container
      {
        EdgeTuples edges(edges_id);
        future_signal_values_type outstanding_signals;

        for (size_type i = 0; i < siblings_.size(); i++)
        {
          outstanding_signals.push_back(
              siblings_[i].eager_aligned_init(edges.local_to(i)));
        }

        while (outstanding_signals.size() > 0)
        {
          signal_value_type ret = outstanding_signals.back().get();
          outstanding_signals.pop_back();

          order_ += ret.order();
          size_ += ret.size();
        }
      }

      // Send total order and size to each sibling; this serves as the signal
      // for them to complete initialization
      {
        for (size_type i = 0; i < siblings_.size(); i++)
        {
          siblings_[i].finalize_init(order_, size_);
        }
      }
    }

    ///
    /// \brief Initialize the graph.
    ///
    /// This implements a particular strategy for initializing the graph when
    /// the edge container resides on a single locality. Edges are distributed
    /// based on the source vertex. A simple message coalescing algorithm is
    /// controlled by a runtime parameter.
    ///
    /// \param edges name of a distributed container of edges
    ///
    void initialize_constant(id_type const & edges_id)
    {
      // FIXME: Update this with use_feb_ and consts
      assert(0);

      // Distribute edges to local subgraphs
      {
        EdgeTuples edges(edges_id);
        future_sizes_type outstanding_applies;

        // Create bins and determine payload size for message coalescing
        std::vector<edge_tuples_type> bins(distribution_.coverage().size());

        size_type payload = 1024;
        pxgl::rts::get_ini_option(payload, "csr_graph.bin_size");
        LCSR_debug("Payload: %u", payload);

        // Iterate through the container of edges, loading edges into
        // appropriate bins, and sending full payloads to targets
        edge_tuples_type const * edge_tuples_ptr = edges.items();
        for (size_type i = 0; i < edge_tuples_ptr->size(); i++)
        {
          edge_tuple_type edge_tuple = (*edge_tuples_ptr)[i];
          size_type src = edge_tuple.source();

          size_type dest = distribution_.locale_id(src);

          bins[dest].push_back(edge_tuple);
          if (bins[dest].size() >= payload)
          {
            for (; i < edge_tuples_ptr->size() && (*edge_tuples_ptr)[i+1].source()==src; i++)
            {
              bins[dest].push_back((*edge_tuples_ptr)[i+1]);
            }

            outstanding_applies.push_back(
                siblings_[dest].eager_init_local(bins[dest]));

            bins[dest].clear();
          }
        }

        // Send any remaining partial payloads
        for (size_type i = 0; i < bins.size(); i++)
        {
          if (bins[i].size() > 0)
          {
            outstanding_applies.push_back(
                siblings_[i].eager_init_local(bins[i]));
            bins[i].clear();
          }
        }

        // Wait until all actions have completed
        while (outstanding_applies.size() > 0)
        {
          outstanding_applies.back().get();
          outstanding_applies.pop_back();
        }
      }

      // Tell each sibling they have received all of their edges
      // so that they know they have received all of their edges
      {
        future_signal_values_type outstanding_signals;

        for (size_type i = 0; i < siblings_.size(); i++)
        {
            outstanding_signals.push_back(
                siblings_[i].eager_signal_init());
        }

        while (outstanding_signals.size() > 0)
        {
          signal_value_type ret = outstanding_signals.back().get();
          outstanding_signals.pop_back();

          order_ += ret.order();
          size_ += ret.size();
        }
      }

      // Send total order and size to each sibling; this serves as the
      // signal for them to complete initialization
      {
        for (size_type i = 0; i < siblings_.size(); i++)
        {
          siblings_[i].finalize_init(order_, size_);
        }
      }
    }

    ///
    /// \brief Build this local portion of the CSR representation.
    ///
    /// \return the number of local vertices and edges.
    ///
    signal_value_type build_representation(id_type edges_id)
    {
      size_type order = 0;
      size_type size = 0;

      {
        pxgl::util::scoped_use l(use_feb_);

        boost::unordered_map<size_type, sizes_type> remote_vertices;

        edge_tuples_type const * edges_ptr = EdgeTuples(edges_id).items();
        BOOST_FOREACH(edge_tuple_type const & edge, (*edges_ptr))
        {
          size_type const & source(edge.source());
          size_type const & target(edge.target());

          // Do not count self-loops
          if (source != target)
          {
            if (distribution_.locale(source) == here_)
            {
              if (distribution_.locale_id(source)
                  == distribution_.locale_id(target))
              {
                // Handle local source and target
                local_map_[source].push_back(
                    adjacency_type(target, edge.weight()));
                local_map_[target];
              }
              else if (distribution_.locale_id(source)
                       != distribution_.locale_id(target))
              {
                // Handle local source, remote target
                local_map_[source].push_back(
                    adjacency_type(target, edge.weight()));

                remote_vertices[distribution_.locale_id(target)].push_back(
                    target);
              }
            }
            else
            {
              // Handle remote source or target
              assert(0);
            }
          }
        }

        // Handle remote targets with degree = 0
        if (siblings_.size() > 1)
        {
          for (size_type i = 0; i < siblings_.size(); i++)
          {
            if (siblings_[i].get_gid() != me_)
            {
              siblings_[i].async_add_local_vertices(remote_vertices[i]);
            }
          }

          remote_updates_ += 1;
        }
      }

      // Wait for all updates
      if (siblings_.size() > 1 && remote_updates_ < siblings_.size())
      {
        feb_data_type d;
        single_feb_.read(d);

        if (1 == d.which())
        {
          error_type e = boost::get<error_type>(d);
          boost::rethrow_exception(e);
        }
      }

      {
        pxgl::util::scoped_use l(use_feb_);

        // Remove duplicate edges and count edges, constructs vertices list
        BOOST_FOREACH(map_type::value_type item, local_map_)
        {
          // Add vertex
          vertices_.push_back(item.first);

          // Clean adjacencies
          std::sort(item.second.begin(), item.second.end());
          
          adjacencies_type::iterator new_end_pos;
          new_end_pos = std::unique(item.second.begin(), item.second.end());

          item.second.erase(new_end_pos, item.second.end());

          size += item.second.size();
        }
        order = vertices_.size();
      }

      // Build simple CSR representation
      index_.resize(order + 1);
      {
        index_[0] = 0;
        size_type v_base = 0;

        BOOST_FOREACH(map_type::value_type const & item, local_map_)
        {
          index_[v_base+1] = index_[v_base] + item.second.size();
          vertex_map_[item.first] = v_base;
          LCSR_debug("deg(%u)=%u; index_[%u]=%u; vertex_map_[%u]=%u\n", 
              item.first, item.second.size(),
              v_base, index_[v_base],
              item.first, vertex_map_[item.first]);

          BOOST_FOREACH(adjacency_type const & adj, item.second)
          {
            edges_.push_back(edge_tuple_type(
                item.first, adj.target(), (size_type)adj.weight()));
          }

          v_base++;
        }

#ifdef PXGL_DEBUG_CSR
        assert(edges_.size() == size);
        assert(vertex_map_.size() == order);
        assert(index_.size() == order + 1);
#endif
      }

      // Throw away local_map_
      local_map_.clear();

      return signal_value_type(order, size);
    }

  public:
    void add_local_vertices(sizes_type vertices)
    {
      constructed();

      {
        pxgl::util::scoped_use l(use_feb_);

        size_type count = 0;
        BOOST_FOREACH(size_type const & vertex, vertices)
        {
          if (local_map_.find(vertex) == local_map_.end())
          {
            local_map_[vertex];
            count++;
          }
        }

        remote_updates_ += 1;

        if (remote_updates_ == siblings_.size())
        {
          // Resume build_representation()
          single_feb_.set(feb_data_type(1));
        }
      }
    }

    typedef hpx::actions::action1<
        csr_graph, 
        csr_graph_add_local_vertices, 
            sizes_type, 
        &csr_graph::add_local_vertices
    > add_local_vertices_action;

    ///
    /// \brief Initialize this local portion of the graph
    ///
    /// \param edges name of local portion of global edge container.
    ///
    /// \return the number of local vertices and edges.
    ///
    signal_value_type aligned_init(id_type const & edges)
    {
      // Setup internal representation
      return build_representation(edges);
    }

    ///
    /// \brief Finalize the initialization process.
    ///
    /// \param order the number of vertices in the graph.
    /// \param size the number of edges in the graph.
    ///
    void finalize_init(size_type order, size_type size)
    {
      pxgl::util::scoped_use l(use_feb_);

      order_ = order;
      size_ = size;

      initialized_feb_.set(feb_data_type(1));
      initialized_ = true;
    }

    ///
    /// \brief Notifies a local portion of the graph that it has received
    ///        all of its edges.
    ///
    /// This process is responsible for building the local portion of the
    /// graph representation.
    ///
    /// \return the order and size of the graph.
    ///
    signal_value_type signal_init(void)
    {
      // The local edges list was already constructed by the init_local()
      // calls.

      return build_representation(hpx::naming::invalid_id);
    }

    size_type init_local(edge_tuples_type edges)
    {
      scoped_lock l(&local_edge_list_);

      // Just collecting edges for now
      local_edge_list_.insert(
          local_edge_list_.end(), edges.begin(), edges.end());

      return 42;
    }

    /// \brief Initialize the graph structure from a list of edge tuples
    ///
    /// The graph is initialized from a given edge list. This command
    /// must only be given once. The exact sequence and synchronization
    /// of actions performed during the initialization depend on the
    /// distributions of the input list of edge tuples and the specified
    /// distribution of the graph.
    ///
    /// Any attempt to initialize the graph multiple times, either on
    /// the same locality, or from different localities, will cause
    /// the enclosing process to abort. Rationale: guaranteeing no
    /// concurrent initializations in a distributed context would incur
    /// a potentially costly upfront synchronization phase; and, it is
    /// unlikely that the programmer intended to cause such a race
    /// condition.
    ///
    /// \param edges the list of edge tuples.
    ///
    /// \pre The graph has been constructed, but not initialized.
    ///
    /// \post The graph has been initialized.
    ///
    void init(id_type const & edges)
    {
      // Wait for the graph to be constructed.
      constructed();
  
      // Select appropriate initialization routine based on the mapping
      // between the graph distribution and the edge container distribution.
      typename EdgeTuples::distribution_type et_dist = 
          EdgeTuples(edges).get_distribution();
      if (distribution_ == et_dist)
      {
        // Initialize graph from an aligned container.
        initialize_aligned(edges);
      }
      else if (et_dist.size() == 1)
      {
        // Initiate 1-N initialization
        initialize_constant(edges);
      }
      else
      {
        // Unsupported mapping
        assert(0);
      }
    }

    /// \brief Wait for the graph to constructed
    ///
    /// Suspends the caller until the graph has been constructed.
    ///
    /// \pre N/A
    /// \post The graph has been constructed.
    ///
    void constructed(void)
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

    ////////////////////////////////////////////////////////////////////////////
    // Use

    /// \brief Get the distribution of the graph
    ///
    /// Suspends until the graph has been constructed.
    ///
    /// \pre  The graph was constructed.
    /// \post N/A
    ///
    /// \return The distribution of the graph.
    distribution_type get_distribution(void)
    {
      // TODO: document whether this needs to be "ready" or "constructed".
      // Switching to constructed case for sgab::sub_k1.
      constructed();

      return distribution_;
    }

    /// \brief Wait for the graph to be initialized
    ///
    /// Suspends the caller if the graph has not been initialized.
    ///
    void ready(void)
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

    void ready_all(void)
    {
      ready();

      BOOST_FOREACH(csr_graph_client_type sibling, siblings_)
      {
        if (sibling.get_gid() != me_)
        {
          sibling.ready();
        }
      }
    }

    /// \brief Get the number of vertices in the graph
    ///
    /// This is a local operation returning the number of vertices in the
    /// graph.
    ///
    /// \return The number of vertices in the graph
    size_type order(void)
    {
      ready();

      return order_;
    }

    /// \brief Get the number of edges in the graph
    ///
    /// This is a local operation returning the number of edges in the
    /// graph.
    ///
    /// \return The number of edges in the graph
    size_type size(void)
    {
      ready();

      return size_;
    }

    sizes_type * vertices(void)
    {
      ready();
      
      return &vertices_;
    }

    typedef hpx::actions::result_action0<
        csr_graph, 
            sizes_type *, 
        csr_graph_vertices, 
        &csr_graph::vertices
    > vertices_action;

    edge_tuples_type * edges(void)
    {
      ready();

      return &edges_;
    }

    typedef hpx::actions::result_action0<
        csr_graph, 
            edge_tuples_type *, 
        csr_graph_edges, 
        &csr_graph::edges
    > edges_action;

    edge_iterator_type neighbors(size_type v)
    {
      ready();

      size_type const i = vertex_map_[v];

      assert(0 < i);
      assert(i+1 < index_.size());

      return edge_iterator_type(&edges_, index_[i], index_[i+1]);
    }

    typedef hpx::actions::result_action1<
        csr_graph, 
            edge_iterator_type, 
        csr_graph_neighbors, 
            size_type,
        &csr_graph::neighbors
    > neighbors_action;

    id_type local_to(size_type index)
    {
      constructed();

      return siblings_[index].get_gid();
    }

    typedef hpx::actions::result_action1<
        csr_graph, 
            id_type, 
        csr_graph_local_to,
            size_type,
        &csr_graph::local_to
    > local_to_action;

    ////////////////////////////////////////////////////////////////////////////
    // Action types
    typedef hpx::actions::result_action1<
        csr_graph,
            signal_value_type,
        csr_graph_aligned_init,
            id_type const &,
        &csr_graph::aligned_init
    > aligned_init_action;
        
    typedef hpx::actions::action2<
        csr_graph,
        csr_graph_finalize_init,
            size_type, 
            size_type,
        &csr_graph::finalize_init
    > finalize_init_action;

    typedef hpx::actions::result_action0<
        csr_graph, 
        signal_value_type,
        csr_graph_signal_init, 
        &csr_graph::signal_init
    > signal_init_action;

    typedef hpx::actions::result_action1<
        csr_graph, 
        size_type,
        csr_graph_init_local, 
        edge_tuples_type,
        &csr_graph::init_local
    > init_local_action;

    typedef hpx::actions::action1<
        csr_graph, 
        csr_graph_init, 
        id_type const &,
        &csr_graph::init
    > init_action;

    typedef hpx::actions::action0<
        csr_graph, csr_graph_ready, &csr_graph::ready
    > ready_action;

    typedef hpx::actions::action0<
        csr_graph, csr_graph_ready_all, &csr_graph::ready_all
    > ready_all_action;

    typedef hpx::actions::result_action0<
        csr_graph, size_type, csr_graph_order, &csr_graph::order
    > order_action;

    typedef hpx::actions::result_action0<
        csr_graph, size_type, csr_graph_size, &csr_graph::size
    > size_action;

    typedef hpx::actions::result_action0<
        csr_graph, distribution_type, csr_graph_get_distribution, 
        &csr_graph::get_distribution
    > get_distribution_action;

  private:
    ////////////////////////////////////////////////////////////////////////////
    // Graph data
    size_type order_;
    size_type size_;

    id_type me_; ///< the name (GID) of this component.
    id_type here_; ///< the location of this component.

    distribution_type distribution_; ///< the distribution of this graph.

    /// \brief The collection of sibling components comprising the decentralized
    /// distributed graph structure.
    std::vector<csr_graph_client_type> siblings_;

    // For initialization
    std::vector<edge_tuple_type> local_edge_list_;
    size_type remote_updates_;

    // Representation
    map_type local_map_;
    sizes_type vertices_;

    sizes_type index_;
    vertex_map_type vertex_map_;
    edge_tuples_type edges_;

    ////////////////////////////////////////////////////////////////////////////
    // Synchronization members
    struct tag {};
    typedef hpx::util::spinlock_pool<tag> mutex_type;
    typedef typename mutex_type::scoped_lock scoped_lock;

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

    // Use as an arb. block
    hpx::util::full_empty<feb_data_type> single_feb_;
  };
}}}

////////////////////////////////////////////////////////////////////////////////
// Stubs interface
namespace pxgl { namespace graphs { namespace stubs {
  template <typename EdgeTuples, typename Distribution>
  struct csr_graph
    : HPX_STUBS_BASE_2(csr_graph, EdgeTuples, Distribution)
  {
    typedef typename server::signal_value_type signal_value_type;
    typedef typename server::csr_graph<EdgeTuples, Distribution> server_type;
    typedef typename server_type::size_type size_type;
    typedef typename server_type::sizes_type sizes_type;
    typedef typename server_type::distribution_type distribution_type;
    typedef typename server_type::id_type id_type;
    typedef typename server_type::ids_type ids_type;
    typedef typename server_type::adjacency_type adjacency_type;
    typedef typename server_type::edge_iterator_type edge_iterator_type;
    typedef typename server_type::adjacencies_type adjacencies_type;

    typedef typename EdgeTuples::item_type edge_tuple_type;
    typedef typename EdgeTuples::items_type edge_tuples_type;


    static void construct(
        id_type const & id,
        id_type const & me,
        distribution_type const & distribution)
    {
      typedef typename server_type::construct_action action_type;
      hpx::applier::apply<action_type>(id, me, distribution);
    }

    static void replicate(
        id_type const & id,
        distribution_type const & distribution,
        ids_type const & sibling_ids)
    {
      typedef typename server_type::replicate_action action_type;
      hpx::applier::apply<action_type>(id, distribution, sibling_ids);
    }

    static signal_value_type aligned_init(
        id_type const & id,
        id_type const & me)
    {
      typedef typename server_type::aligned_init_action action_type;
      return hpx::lcos::eager_future<action_type>(id, me).get();
    }

    static hpx::lcos::promise<signal_value_type> 
        eager_aligned_init(
            id_type const & id,
            id_type const & me)
    {
      typedef typename server_type::aligned_init_action action_type;
      return hpx::lcos::eager_future<action_type>(id, me);
    }

    static void finalize_init(
        hpx::naming::id_type const & id,
        size_type order,
        size_type size)
    {
      typedef typename server_type::finalize_init_action action_type;
      hpx::applier::apply<action_type>(id, order, size);
    }

    static signal_value_type signal_init(hpx::naming::id_type id)
    {
      typedef typename server_type::signal_local_action action_type;
      return hpx::lcos::eager_future<action_type>(id).get();
    }

    static hpx::lcos::promise<signal_value_type>
        eager_signal_init(hpx::naming::id_type id)
    {
      typedef typename server_type::signal_init_action action_type;
      return hpx::lcos::eager_future<action_type>(id);
    }

    static size_type init_local(hpx::naming::id_type id, edge_tuples_type edges)
    {
      typedef typename server_type::init_local_action action_type;
      hpx::applier::apply<action_type>(id, edges);
    }

    static hpx::lcos::promise<size_type> 
        eager_init_local(hpx::naming::id_type id, edge_tuples_type edges)
    {
      typedef typename server_type::init_local_action action_type;
      return hpx::lcos::eager_future<action_type>(id, edges);
    }

    static void init(
        id_type const & id, 
        id_type const & edges)
    {
      typedef typename server_type::init_action action_type;
      hpx::applier::apply<action_type>(id, edges);
    }

    static void async_add_local_vertices(
        id_type const & id,
        sizes_type const & vertices)
    {
      typedef typename server_type::add_local_vertices_action action_type;
      hpx::applier::apply<action_type>(id, vertices);
    }

    static void ready(hpx::naming::id_type const & id)
    {
      typedef typename server_type::ready_action action_type;
      hpx::lcos::eager_future<action_type>(id).get();
    }

    static void ready_all(hpx::naming::id_type const & id)
    {
      typedef typename server_type::ready_all_action action_type;
      hpx::lcos::eager_future<action_type>(id).get();
    }

    static size_type order(hpx::naming::id_type const & id)
    {
      typedef typename server_type::order_action action_type;
      return hpx::lcos::eager_future<action_type>(id).get();
    }

    static size_type size(hpx::naming::id_type const & id)
    {
      typedef typename server_type::size_action action_type;
      return hpx::lcos::eager_future<action_type>(id).get();
    }

    static sizes_type * sync_vertices(id_type const & id)
    {
      typedef typename server_type::vertices_action action_type;
      return hpx::lcos::eager_future<action_type>(id).get();
    }

    static edge_tuples_type * edges(hpx::naming::id_type const & id)
    {
      typedef typename server_type::edges_action action_type;
      return hpx::lcos::eager_future<action_type>(id).get();
    }

    static edge_iterator_type sync_neighbors(
        id_type const & id,
        size_type v)
    {
      typedef typename server_type::neighbors_action action_type;
      return hpx::lcos::eager_future<action_type>(id, v).get();
    }

    static Distribution get_distribution(
        id_type const & id)
    {
      typedef typename server_type::get_distribution_action action_type;
      return hpx::lcos::eager_future<action_type>(id).get();
    }

    static id_type local_to(
        id_type const & id, 
        size_type index)
    {
      typedef typename server_type::local_to_action action_type;
      return hpx::lcos::eager_future<action_type>(id, index).get();
    }
  };
}}}

////////////////////////////////////////////////////////////////////////////////
// Client interface
namespace pxgl { namespace graphs {
  template <typename EdgeTuples, typename Distribution>
  class csr_graph
    : public HPX_CLIENT_BASE_2(csr_graph, EdgeTuples, Distribution)
  {
  public:
    typedef typename EdgeTuples::item_type edge_tuple_type;
    typedef typename EdgeTuples::items_type edge_tuples_type;

  //////////////////////////////////////////////////////////////////////////////
  // Component setup
  private:
    typedef HPX_CLIENT_BASE_2(csr_graph, EdgeTuples, Distribution) base_type;

  public:
    csr_graph()
      : base_type(hpx::naming::invalid_id)
    {}

    csr_graph(hpx::naming::id_type id)
      : base_type(id)
    {}

    ////////////////////////////////////////////////////////////////////////////
    // Graph types
    typedef typename stubs::csr_graph<EdgeTuples, Distribution> stubs_type;
    typedef typename stubs_type::size_type size_type;
    typedef typename stubs_type::sizes_type sizes_type;
    typedef typename stubs_type::id_type id_type;
    typedef typename stubs_type::ids_type ids_type;
    typedef typename stubs_type::signal_value_type signal_value_type;
    typedef typename stubs_type::adjacency_type adjacency_type;
    typedef typename stubs_type::edge_iterator_type edge_iterator_type;
    typedef typename stubs_type::adjacencies_type adjacencies_type;
  
    ////////////////////////////////////////////////////////////////////////////
    // Graph interface
    void construct(
        id_type const & me, 
        Distribution const & distribution) const
    {
      BOOST_ASSERT(this->gid_);
      this->base_type::construct(this->gid_, me, distribution);
    }

    void replicate(
        Distribution const & distribution, 
        ids_type const & sibling_ids) const
    {
      BOOST_ASSERT(this->gid_);
      this->base_type::replicate(this->gid_, distribution, sibling_ids);
    }

    signal_value_type aligned_init(id_type const & me) const
    {
      BOOST_ASSERT(this->gid_);
      return this->base_type::aligned_init(this->gid_, me);
    }

    hpx::lcos::promise<signal_value_type> 
        eager_aligned_init(id_type const & me) const
    {
      BOOST_ASSERT(this->gid_);
      return this->base_type::eager_aligned_init(this->gid_, me);
    }

    void finalize_init(size_type order, size_type size) const
    {
      BOOST_ASSERT(this->gid_);
      this->base_type::finalize_init(this->gid_, order, size);
    }

    signal_value_type signal_init(void)
    {
      BOOST_ASSERT(this->gid_);
      return this->base_type::signal_init(this->gid_);
    }

    hpx::lcos::promise<signal_value_type> eager_signal_init(void)
    {
      BOOST_ASSERT(this->gid_);
      return this->base_type::eager_signal_init(this->gid_);
    }

    void init_local(edge_tuples_type edges)
    {
      BOOST_ASSERT(this->gid_);
      this->base_type::init(this->gid_, edges);
    }

    hpx::lcos::promise<size_type> eager_init_local(edge_tuples_type edges)
    {
      BOOST_ASSERT(this->gid_);
      return this->base_type::eager_init_local(this->gid_, edges);
    }

    void init(id_type const & edges) const
    {
      BOOST_ASSERT(this->gid_);
      this->base_type::init(this->gid_, edges);
    }

    void async_add_local_vertices(sizes_type const & vertices) const
    {
      BOOST_ASSERT(this->gid_);
      this->base_type::async_add_local_vertices(this->gid_, vertices);
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

    sizes_type * sync_vertices(void) const
    {
      BOOST_ASSERT(this->gid_);
      return this->base_type::sync_vertices(this->gid_);
    }

    edge_tuples_type * edges(void) const
    {
      BOOST_ASSERT(this->gid_);
      return this->base_type::edges(this->gid_);
    }

    edge_iterator_type sync_neighbors(size_type v) const
    {
      BOOST_ASSERT(this->gid_);
      return this->base_type::sync_neighbors(this->gid_, v);
    }

    Distribution get_distribution(void) const
    {
      BOOST_ASSERT(this->gid_);
      return this->base_type::get_distribution(this->gid_);
    }

    id_type local_to(size_type index) const
    {
      BOOST_ASSERT(this->gid_);
      return this->base_type::local_to(this->gid_, index);
    }

    ////////////////////////////////////////////////////////////////////////////
    static size_type const invalid_vertex(void)
    {
      return std::numeric_limits<size_type>::max();
    }
  };
}}

#endif

