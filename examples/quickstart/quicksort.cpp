//  Copyright (c) 2007-2010 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <algorithm>

#include <hpx/hpx.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/lcos/future_wait.hpp>

#include <boost/program_options.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/export.hpp>

using namespace hpx;
namespace po = boost::program_options;

///////////////////////////////////////////////////////////////////////////////
// template <typename Iterator>
// void qsort(Iterator begin, Iterator end)
// {
//     if (begin != end) {
//         typedef typename iterator_traits<Iterator>::value_type value_type;
//         Iterator middle = std::partition(
//             begin, end, bind2nd(std::less<value_type>(), *begin));
//         qsort(begin, middle);
//         qsort(std::max(begin+1, middle), end);
//     }
// }
// 
// int main()
// {
//     util::high_resolution_timer t;
// 
//     // randomly fill the vector
//     std::vector<int> data(1000);
//     std::generate(data.begin(), data.end(), std::rand);
// 
//     // sort the vector
//     qsort(data.begin(), data.end());
//     std::cout << "elapsed: " << elapsed << ", result: " << result << std::endl;
// }

///////////////////////////////////////////////////////////////////////////////
template <typename T>
struct quicksort
{
    static void call(naming::id_type prefix, naming::id_type data,
        std::size_t begin, std::size_t end);

    typedef actions::plain_action4<
        naming::id_type, naming::id_type, std::size_t, std::size_t, &quicksort::call
    > quicksort_action;
};

template <typename T>
inline std::size_t partition(T* data, std::size_t begin, std::size_t end)
{
    T *first = data + begin;
    T *last = first + (end - begin);

    T* middle = std::partition(
        first, last, std::bind2nd(std::less<T>(), *first));

    return middle - data;
}

int sort_count = 0;

template <typename T>
void quicksort<T>::call(naming::id_type prefix, naming::id_type d, 
    std::size_t begin, std::size_t end)
{
    if (begin != end) {
        static util::high_resolution_timer t;
        double elapsed = t.elapsed();

        components::memory_block mb(d);
        components::access_memory_block<T> data(mb.get());

//        std::cout << "elapsed: " << (t.elapsed() - elapsed) << std::endl;

        std::size_t middle_idx = partition(data.get_ptr(), begin, end);

        ++sort_count;
//         std::cout << "middle: " << middle_idx << std::endl;

        // always spawn the larger part in a new thread
        if (2 * middle_idx < end - begin) {
            lcos::eager_future<quicksort_action> n(prefix, prefix, d, 
                (std::max)(begin+1, middle_idx), end);

            call(prefix, d, begin, middle_idx);
//            call(prefix, d, (std::max)(begin+1, middle_idx), end);
            components::wait(n);
        }
        else {
            lcos::eager_future<quicksort_action> n(prefix, prefix, d, 
                begin, middle_idx);

//            call(prefix, d, begin, middle_idx);
            call(prefix, d, (std::max)(begin+1, middle_idx), end);
            components::wait(n);
        }
    }
}

hpx::actions::manage_object_action<boost::uint8_t> const raw_memory =
    hpx::actions::manage_object_action<boost::uint8_t>();

///////////////////////////////////////////////////////////////////////////////
int hpx_main(int argument)
{
    // get list of all known localities
    std::vector<naming::id_type> prefixes;
    naming::id_type prefix;
    applier::applier& appl = applier::get_applier();
    if (appl.get_remote_prefixes(prefixes)) {
        // execute the qsort() function on any of the remote localities
        prefix = prefixes[0];
    }
    else {
        // execute the qsort() function locally
        prefix = appl.get_runtime_support_gid();
    }

    {
        // create (remote) memory block
        components::memory_block mb;
        mb.create(prefix, sizeof(int)*argument, raw_memory);
        components::access_memory_block<int> data(mb.get());

        // randomly fill the vector
        std::generate(data.get_ptr(), data.get_ptr() + argument, std::rand);

        util::high_resolution_timer t;
        std::sort(data.get_ptr(), data.get_ptr() + argument);

        double elapsed = t.elapsed();
        std::cout << "elapsed: " << elapsed << std::endl;

        std::generate(data.get_ptr(), data.get_ptr() + argument, std::rand);
        t.restart();

        lcos::eager_future<quicksort<int>::quicksort_action> n(
            prefix, prefix, mb.get_gid(), 0, argument);
        components::wait(n);

        elapsed = t.elapsed();
        std::cout << "elapsed: " << elapsed << std::endl;
        std::cout << "count: " << sort_count << std::endl;

        mb.free();
    }

    // initiate shutdown of the runtime systems on all localities
    components::stubs::runtime_support::shutdown_all();

    return 0;
}

HPX_REGISTER_ACTION_EX(quicksort<int>::quicksort_action, quicksort_action);

///////////////////////////////////////////////////////////////////////////////
bool parse_commandline(int argc, char *argv[], po::variables_map& vm)
{
    try {
        po::options_description desc_cmdline ("Usage: quicksort [options]");
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
            ("value,v", po::value<int>(), 
                "the number of random elements to sort (default is 1000)")
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
        std::cerr << "quicksort: exception caught: " << e.what() << std::endl;
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
        std::cerr << "quicksort: illegal port number given: " << v.substr(p+1) << std::endl;
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
    ~agas_server_helper()
    {
        agas_.stop();
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
        int argument = 1000;
        hpx::runtime::mode mode = hpx::runtime::console;    // default is console mode

        // extract IP address/port arguments
        if (vm.count("agas")) 
            split_ip_address(vm["agas"].as<std::string>(), agas_host, agas_port);

        if (vm.count("hpx")) 
            split_ip_address(vm["hpx"].as<std::string>(), hpx_host, hpx_port);

        if (vm.count("threads"))
            num_threads = vm["threads"].as<int>();

        if (vm.count("value"))
            argument = vm["value"].as<int>();

        if (vm.count("worker"))
            mode = hpx::runtime::worker;

        // initialize and run the AGAS service, if appropriate
        std::auto_ptr<agas_server_helper> agas_server;
        if (vm.count("run_agas_server"))  // run the AGAS server instance here
            agas_server.reset(new agas_server_helper(agas_host, agas_port));

        // initialize and start the HPX runtime
        runtime_type rt(hpx_host, hpx_port, agas_host, agas_port, mode);
        rt.run(boost::bind(hpx_main, argument), num_threads);
    }
    catch (std::exception& e) {
        std::cerr << "quicksort: std::exception caught: " << e.what() << "\n";
        return -1;
    }
    catch (...) {
        std::cerr << "quicksort: unexpected exception caught\n";
        return -2;
    }
    return 0;
}

