//  Copyright (c) 2007-2009 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/lcos/counting_semaphore.hpp>

#include <boost/program_options.hpp>
#include <boost/lexical_cast.hpp>

#include "mandelbrot_component/mandelbrot.hpp"
#include "mandelbrot_component/mandelbrot_callback.hpp"

using namespace hpx;
namespace po = boost::program_options;

///////////////////////////////////////////////////////////////////////////////
void mandelbrot_callback(lcos::counting_semaphore& sem,
    mandelbrot::result const& result)
{
//     std::cout << result.x_ << "," << result.y_ << "," << result.iterations_ 
//               << std::endl;
    sem.signal();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(int sizex, int sizey, int iterations)
{
    // get list of all known localities
    applier::applier& appl = applier::get_applier();

    // get prefixes of all remote localities (if any)
    std::vector<naming::id_type> prefixes;
    appl.get_remote_prefixes(prefixes);

    // execute the mandelbrot() functions remotely only, if any, otherwise
    // locally
    if (prefixes.empty())
        prefixes.push_back(appl.get_runtime_support_gid());

    std::size_t prefix_count = prefixes.size();
    util::high_resolution_timer t;

    // initialize the worker threads, one for each of the pixels
    lcos::counting_semaphore sem;

    boost::scoped_ptr<mandelbrot::server::callback> cb(
        new mandelbrot::server::callback(
            boost::bind(mandelbrot_callback, boost::ref(sem), _1)));
    naming::id_type callback_gid = cb->get_gid();

    for (int x = 0, i = 0; x < sizex; ++x) {
        for (int y = 0; y < sizey; ++y, ++i) {
            mandelbrot::data data(x, y, sizex, sizey, iterations);
            applier::apply_c<mandelbrot_action>(callback_gid, 
                prefixes[i % prefix_count], data);
        }
    }

    // wait for the calculation to finish
    int waitfor = sizex*sizey;
    while (--waitfor >= 0)
        sem.wait();

    double elapsed = t.elapsed();
    std::cout << "elapsed: " << elapsed << std::endl;

    // initiate shutdown of the runtime systems on all localities
    components::stubs::runtime_support::shutdown_all();

    return 0;
}

///////////////////////////////////////////////////////////////////////////////
bool parse_commandline(int argc, char *argv[], po::variables_map& vm)
{
    try {
        po::options_description desc_cmdline ("Usage: mandelbrot [options]");
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
            ("sizex,X", po::value<int>(), 
                "the horizontal (X) size of the generated image (default is 20)")
            ("sizey,Y", po::value<int>(), 
                "the vertical (Y) size of the generated image (default is 20)")
            ("iterations,i", po::value<int>(), 
                "the nmber of iterations to use for the mandelbrot set calculations"
                " (default is 100")
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
        std::cerr << "mandelbrot: exception caught: " << e.what() << std::endl;
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
        std::cerr << "mandelbrot: illegal port number given: " << v.substr(p+1) << std::endl;
        std::cerr << "           using default value instead: " << port << std::endl;
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
// this is the runtime type we use in this application
typedef hpx::runtime_impl<hpx::threads::policies::global_queue_scheduler> runtime_type;

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
        int size_x = 20;
        int size_y = 20;
        int iterations = 100;

        // extract IP address/port arguments
        if (vm.count("agas")) 
            split_ip_address(vm["agas"].as<std::string>(), agas_host, agas_port);

        if (vm.count("hpx")) 
            split_ip_address(vm["hpx"].as<std::string>(), hpx_host, hpx_port);

        if (vm.count("threads"))
            num_threads = vm["threads"].as<int>();

        if (vm.count("sizex"))
            size_x = vm["sizex"].as<int>();
        if (vm.count("sizey"))
            size_y = vm["sizey"].as<int>();
        if (vm.count("iterations"))
            iterations = vm["iterations"].as<int>();

        // initialize and run the AGAS service, if appropriate
        std::auto_ptr<agas_server_helper> agas_server;
        if (vm.count("run_agas_server"))  // run the AGAS server instance here
            agas_server.reset(new agas_server_helper(agas_host, agas_port));

        // initialize and start the HPX runtime
        if (vm.count("worker")) {
            runtime_type rt(hpx_host, hpx_port, agas_host, agas_port, hpx::runtime::worker);
            rt.run(num_threads);
        }
        else {
            runtime_type rt(hpx_host, hpx_port, agas_host, agas_port, hpx::runtime::console);
            rt.run(boost::bind(hpx_main, size_x, size_y, iterations), num_threads);
        }
    }
    catch (std::exception& e) {
        std::cerr << "mandelbrot: std::exception caught: " << e.what() << "\n";
        return -1;
    }
    catch (...) {
        std::cerr << "mandelbrot: unexpected exception caught\n";
        return -2;
    }
    return 0;
}

