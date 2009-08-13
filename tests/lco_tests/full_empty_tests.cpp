//  Copyright (c) 2007-2009 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>

#include <boost/bind.hpp>
#include <boost/detail/lightweight_test.hpp>
#include <boost/program_options.hpp>

using namespace hpx;
namespace po = boost::program_options;

///////////////////////////////////////////////////////////////////////////////
void test1_helper(hpx::util::full_empty<int>& data)
{
    // retrieve gid for this thread
    naming::id_type gid = applier::get_applier().get_thread_manager().
            get_thread_gid(threads::get_self().get_thread_id());
    BOOST_TEST(gid);

    data.set(1);
    BOOST_TEST(!data.is_empty());
}

void test1(threads::thread_state_ex)
{
    // retrieve gid for this thread
    naming::id_type gid = applier::get_applier().get_thread_manager().
            get_thread_gid(threads::get_self().get_thread_id());
    BOOST_TEST(gid);

    // create a full_empty data item
    hpx::util::full_empty<int> data;
    BOOST_TEST(data.is_empty());

    // schedule the helper thread
    applier::register_work(boost::bind(&test1_helper, boost::ref(data)));

    // wait for the other thread to set 'data' to full
    int value = 0;
    data.read(value);   // this blocks for test1_helper to set value

    BOOST_TEST(!data.is_empty());
    BOOST_TEST(value == 1);

    value = 0;
    data.read(value);   // this should not block anymore

    BOOST_TEST(!data.is_empty());
    BOOST_TEST(value == 1);
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    // retrieve gid for this thread
    naming::id_type gid = applier::get_applier().get_thread_manager().
            get_thread_gid(threads::get_self().get_thread_id());
    BOOST_TEST(gid);

    // schedule test threads: test1
    applier::register_work(test1);

    // initiate shutdown of the runtime system
    components::stubs::runtime_support::shutdown_all();

    return 0;
}

///////////////////////////////////////////////////////////////////////////////
bool parse_commandline(char const* name, int argc, char *argv[], 
    po::variables_map& vm)
{
    try {
        std::string usage("Usage: ");
        usage += name;
        usage += " [options]";

        po::options_description desc_cmdline (usage);
        desc_cmdline.add_options()
            ("help,h", "print out program usage (this message)")
            ("run_agas_server,r", "run AGAS server as part of this runtime instance")
            ("worker,w", "run this instance in worker (non-console) mode")
            ("agas", po::value<std::string>(), 
                "the IP address the AGAS server is running on (default taken "
                "from hpx.ini), expected format: 192.168.1.1:7912")
            ("hpx", po::value<std::string>(), 
                "the IP address the HPX parcelport is listening on (default "
                "is localhost:7910), expected format: 192.168.1.1:7913")
            ("threads,t", po::value<int>(), 
                "the number of operating system threads to spawn for this"
                "HPX locality")
            ("num_tests,n", po::value<int>(), 
                "the number of times to repeat the test (default: 1)")
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
        std::cerr << "hpx_runtime: exception caught: " << e.what() << std::endl;
        return false;
    }
    return true;
}

///////////////////////////////////////////////////////////////////////////////
// helper class for DGAS server initialization
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
inline void 
split_ip_address(std::string const& v, std::string& addr, boost::uint16_t& port)
{
    try {
        std::string::size_type p = v.find_first_of(":");
        if (p != std::string::npos) {
            addr = v.substr(0, p);
            port = boost::lexical_cast<boost::uint16_t>(v.substr(p+1));
        }
        else {
            addr = v;
        }
    }
    catch (boost::bad_lexical_cast const& /*e*/) {
        ;   // ignore bad_cast exceptions
    }
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    try {
        // analyze the command line
        po::variables_map vm;
        if (!parse_commandline("hpx_runtime", argc, argv, vm))
            return -1;

        // Check command line arguments.
        std::string hpx_host("localhost"), dgas_host;
        boost::uint16_t hpx_port = HPX_PORT, dgas_port = 0;
        hpx::runtime::mode mode = hpx::runtime::console;    // default is console mode
        int num_threads = 0;
        int num_tests = 1;

        // extract IP address/port arguments
        if (vm.count("agas")) 
            split_ip_address(vm["agas"].as<std::string>(), dgas_host, dgas_port);

        if (vm.count("hpx")) 
            split_ip_address(vm["hpx"].as<std::string>(), hpx_host, hpx_port);

        if (vm.count("worker"))
            mode = hpx::runtime::worker;

        if (vm.count("threads"))
            num_threads = vm["threads"].as<int>();

        if (vm.count("num_tests"))
            num_tests = vm["num_tests"].as<int>();

        // initialize and run the DGAS service, if appropriate
        std::auto_ptr<agas_server_helper> dgas_server;
        if (vm.count("run_agas_server"))  // run the AGAS server instance here
            dgas_server.reset(new agas_server_helper(dgas_host, dgas_port));

        // start the HPX runtime using different numbers of threads
        if (0 == num_threads) {
            int num_of_cores = boost::thread::hardware_concurrency();
            hpx::runtime rt(hpx_host, hpx_port, dgas_host, dgas_port, mode);
            for (int t = 0; t < num_tests; ++t) {
                for (int i = 1; i <= 2*num_of_cores; ++i) { 
                    rt.run(hpx_main, i);
                    std::cerr << ".";
                }
            }
            std::cerr << "\n";
        }
        else {
            hpx::runtime rt(hpx_host, hpx_port, dgas_host, dgas_port, mode);
            for (int t = 0; t < num_tests; ++t) {
                rt.run(hpx_main, num_threads);
                std::cerr << ".";
            }
            std::cerr << "\n";
        }
    }
    catch (std::exception& e) {
        BOOST_TEST(false);
        std::cerr << "std::exception caught: " << e.what() << "\n";
    }
    catch (...) {
        BOOST_TEST(false);
        std::cerr << "unexpected exception caught\n";
    }
    return boost::report_errors();
}


