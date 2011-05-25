// Copyright (c) 2010-2011 Dylan Stark
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying 
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(PXGL_PROCS_WRITE_MATRIX_MARKET_20101103T1151)
#define PXGL_PROCS_WRITE_MATRIX_MARKET_20101103T1151

#include <iostream>
#include <fstream>
#include <sstream>

#include <pxgl/xua/control.hpp>

#define LWRITERMAT_info(str)
#define LWRITERMAT_timer(str,timer)

////////////////////////////////////////////////////////////////////////////////
//! \brief Write contents of an edge-tuple container to file in
//! matrix market format
//
// Context: Locality
//
int write_matrix_market(id_type, std::string);
typedef hpx::actions::plain_result_action2<
    int,
    id_type, std::string,
    write_matrix_market
> write_matrix_market_action;
HPX_REGISTER_PLAIN_ACTION(write_matrix_market_action);

int write_matrix_market(id_type container_id, std::string file_prefix)
{
  hpx::util::high_resolution_timer write_mm_timer;

  std::stringstream filename;
  filename << file_prefix << "." << hpx::get_runtime().get_process().here_index();
  LWRITERMAT_info("Writing file local: " << filename.str())

  container_type container(container_id);
  container_type::items_type const * edges(container.items());

  {
    hpx::util::high_resolution_timer file_timer;

    std::ofstream file;
    file.open(filename.str().c_str());

    BOOST_FOREACH(container_type::item_type edge, *edges)
    {
      file << edge.source() << " "
           << edge.target() << " "
           << edge.weight() << "\n";
    }

    file.close();

    LWRITERMAT_timer("write_mm.file", file_timer);
  }

  LWRITERMAT_timer("write_mm", write_mm_timer);

  return 0;
}

#endif

