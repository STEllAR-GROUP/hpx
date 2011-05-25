// Copyright (c) 2010-2011 Dylan Stark
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying 
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(PXGL_PROCS_GENERATE_RMAT_GRAPH_20101102T1834)
#define PXGL_PROCS_GENERATE_RMAT_GRAPH_20101102T1834

#include <boost/unordered_map.hpp>

#include "../../pxgl/xua/control.hpp"

#define LGENRMAT_info(str)
#define LGENRMAT_timer(str,timer)

typedef std::vector<double> doubles_type;

typedef pxgl::graphs::server::edge_tuple_type edge_tuple_type;
typedef pxgl::graphs::server::edge_tuples_type edge_tuples_type;

typedef hpx::naming::id_type id_type;
typedef std::vector<id_type> ids_type;

////////////////////////////////////////////////////////////////////////////////
// From Accelerated C++, p 135
int nrand(int n)
{
  assert(n > 0 || n <= RAND_MAX);

  const int bucket_size = RAND_MAX / n;
  int r;

  do r = rand() / bucket_size;
  while (r >= n);

  return r;
}

////////////////////////////////////////////////////////////////////////////////
//! Generate R-MAT data
//
// Context: locality
//
size_type generate_rmat_graph_local(id_type, int, doubles_type, int);
typedef hpx::actions::plain_result_action4<
    size_type,
    id_type, int, doubles_type, int,
    generate_rmat_graph_local
> generate_rmat_graph_local_action;
HPX_REGISTER_PLAIN_ACTION(generate_rmat_graph_local_action);

size_type generate_rmat_graph_local(id_type container_id, int scale, 
                                    doubles_type parameters, int edge_factor)
{
  hpx::util::high_resolution_timer rmat_local_timer;

  LGENRMAT_info("Executing local R-MAT action")
  LGENRMAT_info("Local container: " << container_id)

  double a = parameters[0];
  double b = parameters[1];
  double c = parameters[2];
  double d = parameters[3];

  boost::unordered_map<size_type, size_type> known_edges;

  container_type container(container_id);

  container_type::distribution_type
      distribution(container.get_distribution());
  ids_type coverage(distribution.coverage());

  id_type here = hpx::get_runtime().get_process().here();

  srand(time(0));

  size_type order = 1 << scale;
  size_type size = edge_factor * order;
  size_type type_max = 1 << scale;

  double alpha = 0.05;
  hpx::get_option(alpha, "proc.generate_rmat_graph.alpha");
  size_type  tolerance = (int)((1.0-alpha) * size);
  LGENRMAT_info("Alpha: " << alpha)

  size_type num_edges_added = 0;

  edge_tuples_type edges;

  while (num_edges_added < tolerance)
  {
    size_type x = 1;
    size_type y = 1;
    size_type step = order / 2;

    for (int i=0; i<scale; i++)
    {
      double p = rand()*1.0 / RAND_MAX;

      if (p < a)
      {
        // Do nothing
      }
      else if ((p >= a) && (p < a+b))
      {
        y += step;
      }
      else if ((p >= a+b) && (p < a+b+c))
      {
        x += step;
      }
      else if ((p >= a+b+c) && (p < a+b+c+d))
      {
        x += step;
        y += step;
      }
      step = step/2;
    }

    size_type key = (x-1)*order + (y-1);
    if (x-1 != y-1 && known_edges.find(key) == known_edges.end())
    {
      known_edges[key] = key;
      size_type w = nrand(type_max);

      if (distribution.locale(x-1) == here)
      {
        edges.push_back(edge_tuple_type(x-1, y-1, w));
      }
      num_edges_added += 1;
    }
  }

  container.init(edges);

  LGENRMAT_timer("rmat_local", rmat_local_timer)
  return edges.size();
}

////////////////////////////////////////////////////////////////////////////////
//! \brief Generate R-MAT data
void generate_rmat_graph(int, doubles_type, int, id_type);
typedef hpx::actions::plain_action4<
    int, doubles_type, int, id_type,
    generate_rmat_graph
> generate_rmat_graph_action;
HPX_REGISTER_PLAIN_ACTION(generate_rmat_graph_action);

void generate_rmat_graph(int scale, doubles_type parameters, int edge_factor, 
                         id_type container_id)
{
  hpx::util::high_resolution_timer total_timer;

  LGENRMAT_info("Starting data generator")
  LGENRMAT_info("Scale = " << scale)
  LGENRMAT_info("(A,B,C,D) = (" << parameters[0] << "," << parameters[1] << ", "
                             << parameters[2] << "," << parameters[2] << ")")
  LGENRMAT_info("Edge-factor = " << edge_factor)

  pxgl::xua::for_each<
      container_type, generate_rmat_graph_local_action
  >(container_id, scale, parameters, edge_factor);

  LGENRMAT_timer("generate_rmat_graph.total", total_timer)
}

#endif

