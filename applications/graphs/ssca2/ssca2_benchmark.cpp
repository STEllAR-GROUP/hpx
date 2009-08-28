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
#include <hpx/components/vertex/vertex.hpp>
#include <hpx/components/distributed_set/distributed_set.hpp>

#include <hpx/components/distributed_map/distributed_map.hpp>

#include "ssca2/ssca2.hpp"

#include <hpx/lcos/eager_future.hpp>
#include <hpx/lcos/future_wait.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/export.hpp>

#include <boost/unordered_map.hpp>

#define LSSCA_(lvl) LAPP_(lvl) << " [SSCA] "

using namespace hpx;
using namespace std;
namespace po = boost::program_options;

int nrand(int);

///////////////////////////////////////////////////////////////////////////////

int rmat(naming::id_type, int, int, int);

typedef
    actions::plain_result_action4<int, naming::id_type, int, int, int, rmat>
rmat_action;

HPX_REGISTER_ACTION(rmat_action);

int rmat(naming::id_type G, int scale, int edge_factor, int type)
{
    std::size_t num_edges_added;
    double a_, b_, c_, d_;
    std::size_t order, size;
    boost::unordered_map<int64_t, int64_t> known_edges;
    int type_max = 16;
    int status = -1;

    typedef hpx::components::server::graph::init_action     graph_init_action;
    typedef hpx::components::server::graph::order_action    graph_order_action;
    typedef hpx::components::server::graph::size_action     graph_size_action;
    typedef hpx::components::server::graph::add_edge_action graph_add_edge_action;
    typedef hpx::components::server::graph::vertex_name_action graph_vertex_name_action;

    // Setup
    order = 1 << scale;
    size = edge_factor * order;

    if (type == 0)
    {   // nice
        a_ = 0.57;
        b_ = 0.19;
        c_ = 0.19;
        d_ = 0.05;
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
       
    // Initialize the graph.
    // Assumes we are initializing size-many vertices
    // with labels ranging from [0,size).
    lcos::future_value<int> result = lcos::eager_future<graph_init_action>(G, order);
    result.get();

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

        // Use hash table to catch duplicate edges
        int64_t key = (x-1)*order + (y-1);
        if (x-1 != y-1 && known_edges.find(key) == known_edges.end())
        {
           known_edges[key] = key;

           lcos::future_value<naming::id_type> u = lcos::eager_future<graph_vertex_name_action>(G, x-1);
           lcos::future_value<naming::id_type> v = lcos::eager_future<graph_vertex_name_action>(G, y-1);

           results.push_back(
               lcos::eager_future<
                   graph_add_edge_action
               >(G, u.get(), v.get(), nrand(type_max)));

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

int hpx_main(int depth, int scale, int edge_factor, int type)
{
    naming::id_type here = applier::get_applier().get_runtime_support_gid();

    // Create an R-MAT graph.
    using hpx::components::graph;
    graph G (graph::create(here));

    lcos::eager_future<rmat_action> r(
        here, G.get_gid(), scale, edge_factor, type);
    r.get();

    // SSCA#2 Graph Analysis Benchmark

    using hpx::components::ssca2;
    using hpx::components::distributed_set;

    ssca2 SSCA2 (ssca2::create(here));

    // Kernel 1: graph construction (see above)
    // Input:
    //    infile - the file containing the graph data
    // Output:
    //    G = the graph containing the graph data
    //
    // G = graph()
    // for_each(infile.lines(), add_edge_from_line)

    // Kernel 2: classify large sets
    // Input:
    //    G - the graph read in from Kernel 1
    // Output:
    //    edge_set - the list of maximal edges
    //
    // edges = filter(G.edges(), max_edge)

    typedef hpx::components::server::ssca2::edge_set_type edge_set_type;
    typedef distributed_set<edge_set_type> dist_edge_set_type;
    dist_edge_set_type edge_set (dist_edge_set_type::create(here));

    SSCA2.large_set(G.get_gid(), edge_set.get_gid());

    // Kernel 3: graph extraction
    // Input:
    //     G - the graph read in from Kernel 1
    //     edge_set - the list of maximal edges
    //     d - the SubGraphPathLength
    // Output:
    //     subgraphs - the list of subgraphs extracted from G
    //
    // subgraphs = map(edges, extract_subgraph)

    typedef hpx::components::server::ssca2::graph_set_type graph_set_type;
    typedef distributed_set<graph_set_type> dist_graph_set_type;
    dist_graph_set_type subgraphs(dist_graph_set_type::create(here));

    SSCA2.extract(edge_set.get_gid(), subgraphs.get_gid());

    LSSCA_(info) << "Completed Kernel 3";

    // Kernel 4: ...

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
            ("scale,s", po::value<int>(),
                "the scale of the graph, default is 4")
            ("depth,d", po::value<int>(),
                "the subgraph path length for Kernel 3")
            ("edge_factor,f", po::value<int>(),
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

// From Accelerated C++, p 135
int nrand(int n)
{
    if (n <= 0 || n > RAND_MAX)
        throw domain_error("Argument to nrand is out of range");

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

        rt.run(boost::bind(hpx_main, depth, scale, edge_factor, type), num_threads);
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

