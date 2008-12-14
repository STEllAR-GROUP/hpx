//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>

#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/lcos/future_callback.hpp>
#include <hpx/lcos/counting_semaphore.hpp>

#include <boost/program_options.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/export.hpp>

using namespace hpx;
namespace po = boost::program_options;

///////////////////////////////////////////////////////////////////////////////
int mandelbrot(double x, double y, int iterations);

typedef 
    actions::plain_result_action3<int, double, double, int, mandelbrot> 
mandelbrot_action;

HPX_REGISTER_ACTION(mandelbrot_action);

///////////////////////////////////////////////////////////////////////////////
inline long double sqr(long double x)
{
    return x * x;
}

///////////////////////////////////////////////////////////////////////////////
int mandelbrot(double xpt, double ypt, int iterations)
{
    long double x = 0;
    long double y = 0;      //converting from pixels to points

    int k = 0;
    for(/**/; k <= iterations; ++k)
    {
        // The Mandelbrot Function Z = Z*Z+c into x and y parts
        long double xnew = sqr(x) - sqr(y) + xpt;
        long double ynew = 2 * x*y - ypt;
        if (sqr(xnew) + sqr(ynew) > 4) 
            break;
        x = xnew;
        y = ynew;
    }

    return (k >= iterations) ? 0 : k;
}

///////////////////////////////////////////////////////////////////////////////
void mandelbrot_callback(lcos::counting_semaphore& sem,
    int x, int y, int iterations)
{
//     std::cout << x << "," << y << "," << iterations << std::endl;
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

    // execute the mandelbrot() function locally
    prefixes.push_back(appl.get_runtime_support_gid());
    std::size_t prefix_count = prefixes.size();

    typedef lcos::eager_future<mandelbrot_action> future_type;
    typedef lcos::future_callback<future_type> future_callback_type;

    util::high_resolution_timer t;

    // initialize the worker threads, one for each of the pixels
    lcos::counting_semaphore sem;
    std::vector<future_callback_type> futures;
    futures.reserve(sizex*sizey);   // preallocate vector

    double deltax = 1.0 / sizex;
    double deltay = 1.0 / sizey;

    for (int x = 0, i = 0; x < sizex; ++x) {
        for (int y = 0; y < sizey; ++y, ++i) {
            futures.push_back(future_callback_type(
                    future_type(prefixes[i % prefix_count], x * deltax, 
                        y * deltay, iterations), 
                    boost::bind(mandelbrot_callback, boost::ref(sem), x, y, _1)
                ));
        }
    }

    // wait for the calculation to finish
    sem.wait(sizex*sizey);

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
        hpx::runtime rt(hpx_host, hpx_port, agas_host, agas_port);
        rt.run(boost::bind(hpx_main, size_x, size_y, iterations), num_threads);
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

