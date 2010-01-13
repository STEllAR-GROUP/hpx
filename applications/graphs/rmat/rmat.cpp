//  Copyright (c) 2008-2010 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include <iostream>

#include <stdlib.h>

#include <hpx/hpx.hpp>
#include <hpx/util/logging.hpp>

#include <hpx/runtime/actions/plain_action.hpp>

#include <hpx/components/graph/graph.hpp>

#include <hpx/lcos/eager_future.hpp>
#include <hpx/lcos/future_wait.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/export.hpp>

#include <boost/unordered_map.hpp>

#define LRMAT_(lvl) LAPP_(lvl) << " [RMAT] "

using namespace hpx;
using namespace std;
namespace po = boost::program_options;

int nrand(int);

///////////////////////////////////////////////////////////////////////////////
// HPX types

typedef hpx::naming::id_type gid_type;
typedef std::vector<gid_type> gids_type;
typedef std::map<gid_type,gid_type> gids_map_type;

typedef hpx::lcos::future_value<int> future_int_type;
typedef std::vector<future_int_type> future_ints_type;

///////////////////////////////////////////////////////////////////////////////
// Misc. component types

typedef hpx::components::server::graph graph_type;
typedef hpx::components::graph client_graph_type;

///////////////////////////////////////////////////////////////////////////////
// Definitions for R-MAT generator

int rmat(naming::id_type, int, int, int);

typedef
    actions::plain_result_action4<int, naming::id_type, int, int, int, rmat>
rmat_action;

///////////////////////////////////////////////////////////////////////////////

HPX_REGISTER_ACTION(rmat_action);

int rmat(naming::id_type G, int scale, int edge_factor, int type)
{
    std::cout.setf(std::ios::dec);

    std::size_t num_edges_added;
    double a_, b_, c_, d_;
    std::size_t order, size, type_max;
    boost::unordered_map<int64_t, naming::id_type> known_vertices;
    boost::unordered_map<int64_t, int64_t> known_edges;
    int status = -1;
    std::string type_name;

    typedef graph_type::add_vertex_action graph_add_vertex_action;
    typedef graph_type::init_action     graph_init_action;
    typedef graph_type::order_action    graph_order_action;
    typedef graph_type::size_action     graph_size_action;
    typedef graph_type::add_edge_action graph_add_edge_action;
    typedef graph_type::vertex_name_action graph_vertex_name_action;

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

        type_name = "nice";
    }
    else if (type == 1)
    {
        // SSCA2 v2.2
        a_ = 0.55;
        b_ = 0.1;
        c_ = 0.1;
        d_ = 0.25;

        type_name = "nasty";
    }
    else
    {   // Erdos-Renyi
        a_ = 0.25;
        b_ = 0.25;
        c_ = 0.25;
        d_ = 0.25;

        type_name = "erdos";
    }

    std::string out_filename = "rmat_" + boost::lexical_cast<std::string>(scale)
                                       + "_"
                                       + boost::lexical_cast<std::string>(edge_factor)
                                       + "_"
                                       + type_name
                                       + ".tuples";
    ofstream out_file;
    out_file.open (out_filename.c_str());


    // Print info message
    LRMAT_(info) << "R-MAT Scalable Graph Generator";
    LRMAT_(info) << "Scale: " << scale;
    LRMAT_(info) << "Edge-factor:" << edge_factor;
    LRMAT_(info) << "(A,B,C,D) = " << setprecision(2)
              << "(" << a_ << ", " << b_ << ", " << c_ << ", " << d_ << ")";

    naming::id_type here = hpx::applier::get_applier().get_runtime_support_gid();

    srand(time(0));
       
    // Start adding edges in phases
    num_edges_added = 0;

    int x, y;
    double p;
    std::size_t step;
    future_ints_type results;
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
        int type_val;
        int64_t key = (x-1)*order + (y-1);
        if (x-1 != y-1 && known_edges.find(key) == known_edges.end())
        {
           known_edges[key] = key;
           type_val = nrand(type_max);

           results.push_back(
               lcos::eager_future<
                   graph_add_edge_action
               >(G, known_vertices[x-1], known_vertices[y-1], type_val));

           LRMAT_(info) << "Adding edge ("
                        << known_vertices[x-1] << ", "
                        << known_vertices[y-1] << ")";

           out_file << known_vertices[x-1].get_lsb() << " "
                    << known_vertices[y-1].get_lsb() << " "
                    << type_val << std::endl;

           num_edges_added += 1;
        }
    }

    // Check that all in flight actions have completed
    while (results.size() > 0)
    {
        results.back().get();
        results.pop_back();
    }

    out_file.close();

    return 0;
}

///////////////////////////////////////////////////////////////////////////////

int hpx_main(int scale, int edge_factor, int type)
{
    // SSCA#2 Graph Analysis Benchmark
    LRMAT_(info) << "Starting R-MAT graph generator";

    std::cout.setf(std::ios::dec);

    gid_type here = hpx::applier::get_applier().get_runtime_support_gid();;

    // Create the graph used for with all kernels
    client_graph_type G;
    G.create(here);

    // Generate the R-MAT graph if no file is given,
    // otherwise, execute Kernel 1 to read in graph data
    lcos::eager_future<rmat_action> r(
        here, G.get_gid(), scale, edge_factor, type);
    r.get();

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
            ("scale,s", po::value<int>(),
                "the scale of the graph, default is 4")
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
        int scale = 4;
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
        if (mode == hpx::runtime::worker) {
            rt.run(num_threads);
        }
        else
        {
            rt.run(boost::bind(hpx_main, scale, edge_factor, type), num_threads);
        }
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

