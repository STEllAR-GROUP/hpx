// Copyright (c) 2010-2011 Dylan Stark
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying 
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(PXGL_PROCS_READ_MATRIX_MARKET_20101103T1545)
#define PXGL_PROCS_READ_MATRIX_MARKET_20101103T1545

#define READ_BINARY 1

#ifdef READ_BINARY
#include <stdio.h>
#else
#include <iostream>
#include <fstream>
#include <sstream>
#endif

#include <pxgl/xua/control.hpp>

#define LREADMM_info(str)
#define LREADMM_timer(str,timer)

typedef hpx::naming::id_type id_type;
typedef container_type::item_type edge_tuple_type;
typedef container_type::items_type edge_tuples_type;

////////////////////////////////////////////////////////////////////////////////
//! \brief Read matrix market file into an edge-tuple container
//
// Context: Locality
//

void read_matrix_market(
    id_type const & container_id, 
    std::string const & file_prefix)
{

  edge_tuples_type edges;
  container_type container(container_id);
  {
    std::stringstream filename;
    if (hpx::get_runtime().get_process().size() == 1)
    {
      filename << file_prefix;
    }
    else
    {
      filename << file_prefix << "." 
               << hpx::get_runtime().get_process().here_index();
    }
    LREADMM_info("Reading file local: " << filename.str())

    std::ifstream file;
#ifdef READ_BINARY
    FILE* fp = fopen(filename.str().c_str(), "rb");
    assert(fp != NULL);

    int read_scale;
    fread(&read_scale, sizeof(int), 1, fp);

    int read_n;
    fread(&read_n, sizeof(int), 1, fp);

    int read_m;
    fread(&read_m, sizeof(int), 1, fp);

    for (int i = 0; i < read_m; i++)
    {
      int e[3];
      fread(&e, sizeof(int), 3, fp);

      edges.push_back(edge_tuple_type(e[0],e[1],e[2]));
    }

    fclose(fp);
#else
    file.open(filename.str().c_str());
    if (!file) assert(0);

    std::string line;
    while (std::getline(file, line))
    {
      std::istringstream lines(line);
      
      size_type src, tgt, wt;
      lines >> src >> tgt >> wt;

      edges.push_back(edge_tuple_type(src, tgt, wt));
    }

    file.close();
#endif
  }

  container.init(edges);
}

typedef hpx::actions::plain_action2<
    id_type const &, 
    std::string const &,
    read_matrix_market
> read_matrix_market_action;
HPX_REGISTER_PLAIN_ACTION(read_matrix_market_action);

#endif

