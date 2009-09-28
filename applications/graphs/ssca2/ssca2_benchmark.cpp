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
#include <hpx/components/vertex/vertex.hpp>
#include <hpx/components/distributed_set/distributed_set.hpp>
#include <hpx/components/distributed_set/local_set.hpp>

#include <hpx/components/distributed_map/distributed_map.hpp>
#include <hpx/components/distributed_map/local_map.hpp>

#include "ssca2/ssca2.hpp"

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

typedef std::vector<naming::id_type> gids_type;

int nrand(int);

///////////////////////////////////////////////////////////////////////////////

int rmat(naming::id_type, int, int, int);

typedef
    actions::plain_result_action4<int, naming::id_type, int, int, int, rmat>
rmat_action;

///////////////////////////////////////////////////////////////////////////////

HPX_REGISTER_ACTION(rmat_action);

int rmat(naming::id_type G, int scale, int edge_factor, int type)
{
    LSSCA_(info) << "event: action(ssca2_benchmark::rmat) status(begin)";
    LSSCA_(info) << "parent(" << threads::get_parent_id() << ")";

    std::size_t num_edges_added;
    double a_, b_, c_, d_;
    std::size_t order, size, type_max;
    boost::unordered_map<int64_t, naming::id_type> known_vertices;
    boost::unordered_map<int64_t, int64_t> known_edges;
    int status = -1;

    typedef hpx::components::server::graph::add_vertex_action graph_add_vertex_action;
    typedef hpx::components::server::graph::init_action     graph_init_action;
    typedef hpx::components::server::graph::order_action    graph_order_action;
    typedef hpx::components::server::graph::size_action     graph_size_action;
    typedef hpx::components::server::graph::add_edge_action graph_add_edge_action;
    typedef hpx::components::server::graph::vertex_name_action graph_vertex_name_action;

    typedef naming::id_type gid_type;

    // Setup
    order = 1 << scale;
    size = edge_factor * order;
    type_max = 1 << scale;

    if (type == 0)
    {   // nice
        a_ = 0.57;
        b_ = 0.19;
        c_ = 0.19;
        d_ = 0.05;
    }
    else if (type == 1)
    {
        // SSCA2 v2.2
        a_ = 0.55;
        b_ = 0.1;
        c_ = 0.1;
        d_ = 0.25;
    }
    else
    {   // Erdos-Renyi
        a_ = 0.25;
        b_ = 0.25;
        c_ = 0.25;
        d_ = 0.25;
    }

    // Print info message
    LSSCA_(info) << "R-MAT Scalable Graph Generator";
    LSSCA_(info) << "Scale: " << scale;
    LSSCA_(info) << "Edge-factor:" << edge_factor;
    LSSCA_(info) << "(A,B,C,D) = " << setprecision(2)
              << "(" << a_ << ", " << b_ << ", " << c_ << ", " << d_ << ")";

    naming::id_type here = applier::get_applier().get_runtime_support_gid();

    srand(time(0));
       
    typedef hpx::components::server::vertex vertex_type;
    typedef hpx::components::server::distributed_set<vertex_type> dist_vertex_set_type;
    typedef std::vector<naming::id_type> gids_type;

    // Start adding edges in phases
    num_edges_added = 0;

    int x, y;
    double p;
    std::size_t step;
    std::vector<lcos::future_value<int> > results;
    while (num_edges_added < size)
    {
        // Choose edge
        x = 1;
        y = 1;
        step = order/2;
        for (int i=0; i<scale; ++i)
        {
            p = rand()*1.0 / RAND_MAX;

            // Pick the partition
            if (p < a_)
            {   
            // Do nothing
            }   
            else if ((p >= a_) && (p < a_+b_))
            {
                y += step;
            } 
            else if ((p >= a_+b_) && (p < a_+b_+c_))
            {
                x += step;
            }
            else if ((p >= a_+b_+c_) && (p < a_+b_+c_+d_))
            {
                x += step;
                y += step;
            }
            step = step/2;
        }

        // Use hash to catch duplicate vertices
        // Note: Update this to do the two actions in parallel
        if (known_vertices.find(x-1) == known_vertices.end())
        {
            known_vertices[x-1] = lcos::eager_future<
                graph_add_vertex_action
            >(G, naming::invalid_id).get();
        }
        if (known_vertices.find(y-1) == known_vertices.end())
        {
            known_vertices[y-1] = lcos::eager_future<
                graph_add_vertex_action
            >(G, naming::invalid_id).get();
        }

        // Use hash table to catch duplicate edges
        int64_t key = (x-1)*order + (y-1);
        if (x-1 != y-1 && known_edges.find(key) == known_edges.end())
        {
           known_edges[key] = key;

           results.push_back(
               lcos::eager_future<
                   graph_add_edge_action
               >(G, known_vertices[x-1], known_vertices[y-1], nrand(type_max)));

           LSSCA_(info) << "Adding edge ("
                        << known_vertices[x-1] << ", "
                        << known_vertices[y-1] << ")";

           num_edges_added += 1;
        }
    }

    // Check that all in flight actions have completed
    while (results.size() > 0)
    {
        results.back().get();
        results.pop_back();
    }

    return 0;
}

///////////////////////////////////////////////////////////////////////////////

int hpx_main(int depth, std::string input_file, int scale, int edge_factor, int type)
{
    LSSCA_(info) << "event: action(ssca2_benchmark::hpx_main) status(begin)";
    LSSCA_(info) << "parent(" << threads::get_parent_id() << ")";
    std::cout.setf(std::ios::dec);

    typedef std::vector<naming::id_type> gids_type;

    // Find out where here is
    naming::id_type here = applier::get_applier().get_runtime_support_gid();

    // SSCA#2 Graph Analysis Benchmark

    using hpx::components::ssca2;
    using hpx::components::distributed_set;
    using hpx::components::local_set;
    using hpx::components::distributed_map;
    using hpx::components::server::edge;

    typedef naming::id_type gid_type;

    double total_time;

    // Instantiate the SSCA2 component which implements the kernels
    // This is not strictly necessary, and the component actions
    // could have been implemented directly in this file
    ssca2 SSCA2 (ssca2::create(here));

    // Create the graph used for with all kernels
    using hpx::components::graph;
    graph G(graph::create(here));

    // Generate the R-MAT graph if no file is given,
    // otherwise, execute Kernel 1 to read in graph data
    if (input_file.length() == 0)
    {
        LSSCA_(info) << "Skipping Kernel 1";

        lcos::eager_future<rmat_action> r(
            here, G.get_gid(), scale, edge_factor, type);
        r.get();
    }
    else
    {
        // Kernel 1: graph construction (see above)
        // Input:
        //    infile - the file containing the graph data
        // Output:
        //    G = the graph containing the graph data
        //
        // for_each(infile.lines(), add_edge_from_line)

        LSSCA_(info) << "Starting Kernel 1";

        /* Begin: timed execution of Kernel 1 */
        hpx::util::high_resolution_timer k1_t;
        SSCA2.read_graph(G.get_gid(), input_file);
        total_time = k1_t.elapsed();
        /* End: timed execution of Kernel 1 */
        LSSCA_(info) << "Completed Kernel 1 in " << total_time << " ms";
        std::cout << "Completed Kernel 1 in " << total_time << " ms" << std::endl;
    }

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
    distributed_set<edge> edge_set(distributed_set<edge>::create(here));
    SSCA2.large_set(G.get_gid(), edge_set.get_gid());
    total_time = k2_t.elapsed();
    /* End: timed execution of Kernel 2 */
    LSSCA_(info) << "Completed Kernel 2 in " << total_time << " ms";
    std::cout << "Completed Kernel 2 in " << total_time << " ms" << std::endl;

    LSSCA_(info) << "Large set:";

    int num_edges = 0;

    gids_type locals = edge_set.locals();

    gids_type::const_iterator lend = locals.end();
    for (gids_type::const_iterator lit = locals.begin();
         lit != lend; ++lit)
    {
        gids_type edges = components::stubs::local_set<edge>::get(*lit);

        gids_type::const_iterator eend = edges.end();
        for (gids_type::const_iterator eit = edges.begin();
             eit != eend; ++eit, ++num_edges)
        {
            edge::edge_snapshot e = components::stubs::edge::get_snapshot(*eit);

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
    distributed_set<graph_type> subgraphs(distributed_set<graph_type>::create(here));
    SSCA2.extract(edge_set.get_gid(), subgraphs.get_gid());
    total_time = k3_t.elapsed();
    /* End: timed execution of Kernel 3 */

    LSSCA_(info) << "Completed Kernel 3 in " << total_time << " ms";
    std::cout << "Completed Kernel 3 in " << total_time << " ms" << std::endl;

    LSSCA_(info) << "Subgraphs:";

    int num_graphs = 0;

    gids_type slocals = subgraphs.locals();

    gids_type::const_iterator slend = slocals.end();
    for (gids_type::const_iterator slit = slocals.begin();
         slit != slend; ++slit)
    {
        gids_type subgraphs = components::stubs::local_set<graph_type>::get(*slit);

        gids_type::const_iterator gend = subgraphs.end();
        for (gids_type::const_iterator git = subgraphs.begin();
             git != gend; ++git, ++num_graphs)
        {
            int order = components::stubs::graph::order(*git);
            int size = components::stubs::graph::size(*git);

            LSSCA_(info) << num_graphs << ": order" << order << ", size=" << size;

        }
    }

    // Kernel 4: graph analysis algorithm
    // Input:
    //     G - the graph read in from Kernel 1
    //     scale - the scale for this run of the benchmark
    //     k4_approx - the log of the number of vertices to consider
    // Output:
    //     bc_scores - the collection of betweenness centrality scores

    LSSCA_(info) << "Starting Kernel 4";

    typedef std::map<naming::id_type,naming::id_type> gids_map_type;
    typedef components::server::vertex vertex_type;

    int k4_approx = scale;

    LSSCA_(info) << "K4 approx. is " << k4_approx;

    gid_type V = G.vertices();

    gid_type VS;
    if (k4_approx < scale && k4_approx > 0)
    {
         gid_type here = applier::get_applier().get_prefix();

         // Create new VS, a subset of V(G)
         LSSCA_(info) << "Creating new VS";
         VS = hpx::components::distributed_set<
                 hpx::components::server::vertex
              >::create(here).get_gid();

         // Select random items
         gids_type v_locals =
             lcos::eager_future<
                 components::server::distributed_set<vertex_type>::locals_action
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
         VS = V;
     }

    /* Begin: timed execution of Kernel 4 */
    hpx::util::high_resolution_timer k4_t;
    distributed_map<gids_map_type>
        bc_scores(distributed_map<gids_map_type>::create(here));
    lcos::eager_future<kernel4_action>
        k4(here, V, VS, k4_approx, bc_scores.get_gid());
    k4.get();
    total_time = k4_t.elapsed();
    /* End: timed execution of Kernel 4 */

    LSSCA_(info) << "Completed Kernel 4 in " << total_time << " ms";
    std::cout << "Completed Kernel 4 in " << total_time << " ms" << std::endl;

    double teps;
    if (k4_approx == scale)
        teps = calculate_teps(V, G.order(), total_time, true);
    else
        teps = calculate_teps(V, 1<<k4_approx, total_time, false);
    std::cout << "TEPS = " << teps << std::endl;

    // Free the graph component
    subgraphs.free();
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
            ("scale,s", po::value<int>(),
                "the scale of the graph, default is 4")
            ("depth,d", po::value<int>(),
                "the subgraph path length for Kernel 3")
            ("edge_factor,e", po::value<int>(),
                "the edge factor of the R-MAT graph")
            ("type,y", po::value<int>(),
                "the type of R-MAT, controls the (a,b,c,d) parameters")
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
// From Accelerated C++, p 135

int nrand(int n)
{
    if (n <= 0 || n > RAND_MAX)
        std::cerr << "Argument to nrand is out of range" << std::endl;

    const int bucket_size = RAND_MAX / n;
    int r;

    do r = rand() / bucket_size;
    while (r >= n);

    return r;
}

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
        int scale = 4;
        int depth = 3;
        int edge_factor = 8;
        int type = 0;

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

        if (vm.count("scale"))
            scale = vm["scale"].as<int>();

        if (vm.count("depth"))
            depth = vm["depth"].as<int>();

        if (vm.count("edge_factor"))
            edge_factor = vm["edge_factor"].as<int>();

        if (vm.count("type"))
            type = vm["type"].as<int>();

        // initialize and run the AGAS service, if appropriate
        std::auto_ptr<agas_server_helper> agas_server;
        if (vm.count("run_agas_server"))  // run the AGAS server instance here
            agas_server.reset(new agas_server_helper(agas_host, agas_port));

        // initialize and start the HPX runtime
        typedef hpx::runtime_impl<hpx::threads::policies::global_queue_scheduler> runtime_type;
        runtime_type rt(hpx_host, hpx_port, agas_host, agas_port, mode);

        rt.run(boost::bind(hpx_main, depth, filename, scale, edge_factor, type), num_threads);
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

