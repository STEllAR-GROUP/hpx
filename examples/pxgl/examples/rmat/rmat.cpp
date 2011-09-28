// Copyright (c) 2010-2011 Dylan Stark
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying 
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

#include "../../pxgl/pxgl.hpp"
#include "../../pxgl/util/futures.hpp"

#include "../../pxgl/xua/range.hpp"
#include "../../pxgl/xua/constant_distribution.hpp"
#include "../../pxgl/xua/arbitrary_distribution.hpp"

#include "../../pxgl/xua/vector.hpp"
#include "../../pxgl/xua/control.hpp"

#include "../../pxgl/graphs/edge_tuple.hpp"

// Define logging helper
#define LRMAT_info(str)
#define LRMAT_fatal(str)
#define LRMAT_timer(str,timer)
#define LRMAT_data(str,...)

////////////////////////////////////////////////////////////////////////////////
// Type definitions
typedef unsigned long size_type;
typedef std::vector<size_type> sizes_type;

typedef hpx::lcos::promise<size_type> future_size_type;
typedef std::vector<future_size_type> future_sizes_type;

typedef std::vector<double> doubles_type;

typedef hpx::naming::id_type id_type;
typedef std::vector<id_type> ids_type;

typedef pxgl::xua::vector<
    pxgl::xua::arbitrary_distribution<id_type, pxgl::xua::range>,
    pxgl::graphs::server::edge_tuple_type
> container_type;

// FIXME: remove dependency between the above typedefs and the below processes
// This must go below the container typedef
#include "../../pxgl/procs/generate_rmat_graph.hpp"
#include "../../pxgl/procs/write_matrix_market.hpp"

HPX_REGISTER_FUTURE(size_type, size);
HPX_REGISTER_FUTURE(sizes_type, sizes);
HPX_REGISTER_FUTURE(doubles_type, doubles);

////////////////////////////////////////////////////////////////////////////////
// Main
int hpx_main(boost::program_options::variables_map &vm)
{
  hpx::util::high_resolution_timer hpx_main_timer;

  int scale = 4;
  hpx::get_option(vm, "scale", scale, "rmat.scale");
  int edge_factor = 8;
  hpx::get_option(vm, "edge_factor", edge_factor, "rmat.edge_factor");

  double a = 0.57, b = 0.19, c = 0.19, d = 0.05;
  {
    std::string parameters;
    hpx::get_option(vm, "parameters", parameters, "rmat.parameters");

    if (parameters.size() > 0)
      assert(std::istringstream(parameters) >> a >> b >> c >> d);
  }
  
  LRMAT_info("Distributed R-MAT Data Generator")

  // Generate R-MAT data
  container_type edge_tuples;
  {
    hpx::util::high_resolution_timer rmat_timer;

    hpx::process my_proc(hpx::get_runtime().get_process());
    id_type here = my_proc.here();

    pxgl::xua::arbitrary_distribution<id_type, pxgl::xua::range> 
        et_dist(hpx::get_runtime().get_process().localities());

    edge_tuples.create(here);
    edge_tuples.construct(edge_tuples.get_gid(), et_dist);

    std::vector<double> params(4);
    params[0] = a;
    params[1] = b;
    params[2] = c;
    params[3] = d;

    hpx::applier::apply<generate_rmat_graph_action> (
        here, scale, params, edge_factor, edge_tuples.get_gid());

    LRMAT_timer("hpx_main.rmat", rmat_timer)
  }

  // Write out R-MAT data to file
  std::string filename;
  hpx::get_option(filename, "rmat.output_filename");
  if (filename.size() > 0)
  {
    hpx::util::high_resolution_timer writer_timer;

    pxgl::xua::blocking_for_each<
        container_type, write_matrix_market_action
    >(edge_tuples.get_gid(), filename);

    LRMAT_timer("hpx_main.writer", writer_timer)
  }

  LRMAT_data("num_edges(%d)", edge_tuples.size())

  LRMAT_timer("hpx_main", hpx_main_timer)

  hpx::finalize();

  return 0;
}

////////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
  // Configure application-specific options
  boost::program_options::options_description
      desc_commandline("Usage: gen_rmat_graph --params \"A B C D\" "
                       "[--edge_factor E] "
                       "[hpx_options]");

  desc_commandline.add_options()
      ("scale,s", boost::program_options::value<int>(),
          "the scale of the graph; default: 4")
      ("parameters,p", boost::program_options::value<std::string>(),
          "the (A, B, C, D) R-MAT paramters; default: \"0.57 0.19 0.19 0.05\"")
      ("edge_factor,e", boost::program_options::value<int>(),
          "the edge-factor; default: 8")
      ;

  // Spawn main task
  return hpx::init(desc_commandline, argc, argv);
}

