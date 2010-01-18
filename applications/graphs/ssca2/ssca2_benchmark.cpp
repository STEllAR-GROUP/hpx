//  Copyright (c) 2008-2010 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include <iostream>

#include <stdlib.h>
#include <time.h>

#include <hpx/hpx.hpp>
#include <hpx/util/logging.hpp>

#include <hpx/runtime/actions/plain_action.hpp>

#include <hpx/components/graph/graph.hpp>
#include <hpx/components/graph/edge.hpp>
#include <hpx/components/graph/vertex.hpp>
#include <hpx/components/distributed_set/distributed_set.hpp>
#include <hpx/components/distributed_set/local_set.hpp>

#include <hpx/components/distributed_map/distributed_map.hpp>
#include <hpx/components/distributed_map/local_map.hpp>

#include "props/props.hpp"

#include <hpx/lcos/eager_future.hpp>
#include <hpx/lcos/future_wait.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/export.hpp>

#include <boost/unordered_map.hpp>

#include "ssca2_benchmark.hpp"

#define LSSCA_(lvl) LAPP_(lvl) << " [SSCA] "

using namespace hpx;
using namespace std;
namespace po = boost::program_options;

///////////////////////////////////////////////////////////////////////////////

int hpx_main(int depth, std::string input_file, int k4_approx)
{
    // SSCA#2 Graph Analysis Benchmark
    LSSCA_(info) << "Starting SSCA2 Graph Analysis Benchmark";

    std::cout.setf(std::ios::dec);

    int scale;

    double total_time;
    gid_type here = find_here();

    // Create the graph used for with all kernels
    client_graph_type G;
    G.create(here);

    // Kernel 1: graph construction (see above)
    // Input:
    //    infile - the file containing the graph data
    // Output:
    //    G = the graph containing the graph data
    //
    // for_each(infile.lines(), add_edge_from_line)

    LSSCA_(info) << "Starting Kernel 1";

    /* Begin: timed execution of Kernel 1 v1 */
    hpx::util::high_resolution_timer k1_v1_t;
    lcos::eager_future<kernel1_v1_action>
        k1_v1(here, G.get_gid(), input_file);
    k1_v1.get();
    total_time = k1_v1_t.elapsed();
    /* End: timed execution of Kernel 1  v1*/
    LSSCA_(info) << "Completed Kernel 1 v1 in " << total_time << " sec";
    std::cout << "Completed Kernel 1 v1 in " << total_time << " sec" << std::endl;

    {
        /* Begin: timed execution of Kernel 1 v1 */
        client_graph_type G_K2;
        G_K2.create(here);
        hpx::util::high_resolution_timer k1_v2_t;
        lcos::eager_future<kernel1_v2_action>
            k1_v2(here, G_K2.get_gid(), input_file);
        k1_v2.get();
        total_time = k1_v2_t.elapsed();
        /* End: timed execution of Kernel 1  v2*/
        LSSCA_(info) << "Completed Kernel 1 v2 in " << total_time << " sec";
        std::cout << "Completed Kernel 1 v2 in " << total_time << " sec" << std::endl;
        G_K2.free();
    }

    {
        /* Begin: timed execution of Kernel 1 v1 */
        client_graph_type G_K3;
        G_K3.create(here);
        hpx::util::high_resolution_timer k1_v3_t;
        lcos::eager_future<kernel1_v2_action>
            k1_v3(here, G_K3.get_gid(), input_file);
        k1_v3.get();
        total_time = k1_v3_t.elapsed();
        /* End: timed execution of Kernel 1  v2*/
        LSSCA_(info) << "Completed Kernel 1 v3 in " << total_time << " sec";
        std::cout << "Completed Kernel 1 v3 in " << total_time << " sec" << std::endl;
        G_K3.free();
    }

    // Derive scale from order of the input graph
    scale = log2(G.order());
    LSSCA_(info) << "Input file scale: " << scale;

    // Kernel 2: classify large sets
    // Input:
    //    G - the graph read in from Kernel 1
    // Output:
    //    edge_set - the list of maximal edges
    //
    // edges = filter(G.edges(), max_edge)

    LSSCA_(info) << "Starting Kernel 2";

    /* Begin: timed execution of Kernel 2 */
    hpx::util::high_resolution_timer k2_t;
    client_dist_edge_set_type edge_set;
    edge_set.create(here);
    lcos::eager_future<kernel2_action>
        k2(here, G.get_gid(), edge_set.get_gid());
    k2.get();
    total_time = k2_t.elapsed();
    /* End: timed execution of Kernel 2 */
    LSSCA_(info) << "Completed Kernel 2 in " << total_time << " sec";
    std::cout << "Completed Kernel 2 in " << total_time << " sec" << std::endl;

    LSSCA_(info) << "Large set:";

    int num_edges = 0;

    gids_type locals = edge_set.locals();

    gids_type::const_iterator lend = locals.end();
    for (gids_type::const_iterator lit = locals.begin();
         lit != lend; ++lit)
    {
        gids_type edges = stub_local_edge_set_type::get(*lit);

        gids_type::const_iterator eend = edges.end();
        for (gids_type::const_iterator eit = edges.begin();
             eit != eend; ++eit, ++num_edges)
        {
            edge_type::edge_snapshot e = stub_edge_type::get_snapshot(*eit);

            LSSCA_(info) << num_edges << ": (" << e.source_ << ", " << e.target_
                         << ", " << e.label_ << ")";
        }
    }

    LSSCA_(info) << "Found total of " << num_edges << " edges";

    // Kernel 3: graph extraction
    // Input:
    //     G - the graph read in from Kernel 1
    //     edge_set - the list of maximal edges
    //     d - the SubGraphPathLength
    // Output:
    //     subgraphs - the list of subgraphs extracted from G
    //
    // subgraphs = map(edges, extract_subgraph)

    LSSCA_(info) << "Starting Kernel 3";

    typedef components::server::graph graph_type;

    /* Begin: timed execution of Kernel 3 */
    hpx::util::high_resolution_timer k3_t;
    client_dist_graph_set_type subgraphs;
    subgraphs.create(here);
    lcos::eager_future<kernel3_action>
        k3(here, edge_set.get_gid(), subgraphs.get_gid(), depth);
    k3.get();
    total_time = k3_t.elapsed();
    /* End: timed execution of Kernel 3 */

    LSSCA_(info) << "Completed Kernel 3 in " << total_time << " sec";
    std::cout << "Completed Kernel 3 in " << total_time << " sec" << std::endl;

    LSSCA_(info) << "Subgraphs:";

    int num_graphs = 0;

    gids_type slocals = subgraphs.locals();

    gids_type::const_iterator slend = slocals.end();
    for (gids_type::const_iterator slit = slocals.begin();
         slit != slend; ++slit)
    {
        gids_type subgraphs = stub_local_graph_set_type::get(*slit);

        gids_type::const_iterator gend = subgraphs.end();
        for (gids_type::const_iterator git = subgraphs.begin();
             git != gend; ++git, ++num_graphs)
        {
            int order = stub_graph_type::order(*git);
            int size = stub_graph_type::size(*git);

            LSSCA_(info) << num_graphs << ": order" << order << ", size=" << size;

        }
    }

    // Free components
    subgraphs.free();
    edge_set.free();

    // Kernel 4: graph analysis algorithm
    // Input:
    //     G - the graph read in from Kernel 1
    //     scale - the scale for this run of the benchmark
    //     k4_approx - the log of the number of vertices to consider
    // Output:
    //     bc_scores - the collection of betweenness centrality scores

    LSSCA_(info) << "Starting Kernel 4";

    LSSCA_(info) << "K4 approx. is " << k4_approx;

    gid_type V = G.vertices();

    if (k4_approx == 0)
        k4_approx = scale;

    gid_type VS;
    if (k4_approx < scale && k4_approx > 0)
    {
         gid_type here = find_here();

         // Create new VS, a subset of V(G)
         LSSCA_(info) << "Creating new VS";
         client_dist_vertex_set_type vs;
         vs.create(here);
         VS = vs.get_gid();

         // Select random items
         gids_type v_locals =
             lcos::eager_future<
                 dist_vertex_set_type::locals_action
             >(V).get();
         select_random_vertices(v_locals, k4_approx, VS);
     }
     else if (k4_approx == scale)
     {
         LSSCA_(info) << "Using V as VS";
         VS = V;
     }
     else
     {
         LSSCA_(info) << "Error: k4_approx not in (0,scale]; using V as VS";
         k4_approx = scale;
         VS = V;
     }

    /* Begin: timed execution of Kernel 4 */
    hpx::util::high_resolution_timer k4_t;
    client_dist_gids_map_type bc_scores;
    bc_scores.create(here);
    lcos::eager_future<kernel4_action>
        k4(here, V, VS, k4_approx, bc_scores.get_gid());
    k4.get();
    total_time = k4_t.elapsed();
    /* End: timed execution of Kernel 4 */

    LSSCA_(info) << "Completed Kernel 4 in " << total_time << " sec";
    std::cout << "Completed Kernel 4 in " << total_time << " sec" << std::endl;

    double teps;
    if (k4_approx == scale)
        teps = calculate_teps(V, G.order(), total_time, true);
    else
        teps = calculate_teps(V, 1<<k4_approx, total_time, false);
    std::cout << "TEPS = " << teps << std::endl;

    // Free components
    if (k4_approx < scale && k4_approx > 0)
    {
        stub_dist_vertex_set_type::free(VS);
    }
    bc_scores.free();

    // Free components
    G.free();

    // Shut down runtime services
    components::stubs::runtime_support::shutdown_all();

    return 0;
}

///////////////////////////////////////////////////////////////////////////////
bool parse_commandline(int argc, char *argv[], po::variables_map& vm)
{
    try {
        po::options_description desc_cmdline ("Usage: hpx_runtime [options]");
        desc_cmdline.add_options()
            ("help,h", "print out program usage (this message)")
            ("run_agas_server,r", "run AGAS server as part of this runtime instance")
            ("worker,w", "run this instance in worker (non-console) mode")
            ("agas,a", po::value<std::string>(), 
                "the IP address the AGAS server is running on (default taken "
                "from hpx.ini), expected format: 192.168.1.1:7912")
            ("hpx,x", po::value<std::string>(), 
                "the IP address the HPX parcelport is listening on (default "
                "is localhost:7910), expected format: 192.168.1.1:7913")
            ("threads,t", po::value<int>(), 
                "the number of operating system threads to spawn for this"
                "HPX locality")
            ("file,f", po::value<std::string>(),
                "the file containing the graph data")
            ("depth,d", po::value<int>(),
                "the subgraph path length for Kernel 3 (default is 3)")
            ("k4_approx,k", po::value<int>(),
                "the approximate scale for Kernel 4")
        ;

        po::store(po::command_line_parser(argc, argv)
            .options(desc_cmdline).run(), vm);
        po::notify(vm);

        // print help screen
        if (vm.count("help")) {
            std::cout << desc_cmdline;
            return false;
        }
    }
    catch (std::exception const& e) {
        std::cerr << "rmat: exception caught: " << e.what() << std::endl;
        return false;
    }
    return true;
}

///////////////////////////////////////////////////////////////////////////////
inline void 
split_ip_address(std::string const& v, std::string& addr, boost::uint16_t& port)
{
    std::string::size_type p = v.find_first_of(":");
    try {
        if (p != std::string::npos) {
            addr = v.substr(0, p);
            port = boost::lexical_cast<boost::uint16_t>(v.substr(p+1));
        }
        else {
            addr = v;
        }
    }
    catch (boost::bad_lexical_cast const& /*e*/) {
        std::cerr << "rmat: illegal port number given: " << v.substr(p+1) << std::endl;
        std::cerr << "      using default value instead: " << port << std::endl;
    }
}

///////////////////////////////////////////////////////////////////////////////
// helper class for AGAS server initialization
class agas_server_helper
{
public:
    agas_server_helper(std::string host, boost::uint16_t port)
      : agas_pool_(), agas_(agas_pool_, host, port)
    {
        agas_.run(false);
    }

private:
    hpx::util::io_service_pool agas_pool_; 
    hpx::naming::resolver_server agas_;
};

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    try {
        // analyze the command line
        po::variables_map vm;
        if (!parse_commandline(argc, argv, vm))
            return -1;

        // Check command line arguments.
        std::string hpx_host("localhost"), agas_host;
        boost::uint16_t hpx_port = HPX_PORT, agas_port = 0;
        int num_threads = 1;
        std::string filename;
        int depth = 3;
        int k4_approx = 0;

        hpx::runtime::mode mode = hpx::runtime::console;    // default is console mode

        // extract IP address/port arguments
        if (vm.count("agas")) 
            split_ip_address(vm["agas"].as<std::string>(), agas_host, agas_port);

        if (vm.count("hpx")) 
            split_ip_address(vm["hpx"].as<std::string>(), hpx_host, hpx_port);

        if (vm.count("threads"))
            num_threads = vm["threads"].as<int>();

        if (vm.count("worker"))
            mode = hpx::runtime::worker;
            
        if (vm.count("file"))
            filename = vm["file"].as<std::string>();

        if (vm.count("depth"))
            depth = vm["depth"].as<int>();

        if (vm.count("k4_approx"))
            k4_approx = vm["k4_approx"].as<int>();

        // initialize and run the AGAS service, if appropriate
        std::auto_ptr<agas_server_helper> agas_server;
        if (vm.count("run_agas_server"))  // run the AGAS server instance here
            agas_server.reset(new agas_server_helper(agas_host, agas_port));

        // initialize and start the HPX runtime
        typedef hpx::runtime_impl<hpx::threads::policies::global_queue_scheduler> runtime_type;
        runtime_type rt(hpx_host, hpx_port, agas_host, agas_port, mode);
        rt.run(boost::bind(hpx_main, depth, filename, k4_approx), num_threads);

    }
    catch (std::exception& e) {
        std::cerr << "std::exception caught: " << e.what() << "\n";
        return -1;
    }
    catch (...) {
        std::cerr << "unexpected exception caught\n";
        return -2;
    }
    return 0;
}

