//  Copyright (c) 2007-2008 Hartmut Kaiser, Richard D Guidry Jr
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <cstring>
#include <iostream>

#include <hpx/hpx.hpp>
#include <hpx/components/distributing_factory/distributing_factory.hpp>
#include <hpx/components/amr_test/stencil.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

using namespace hpx;
using namespace std;

///////////////////////////////////////////////////////////////////////////////
// Create functional components, one for each data point, use those to 
// initialize stencil instances
void init_stencils(
    components::distributing_factory::result_type const& stencils)
{
    typedef components::distributing_factory::result_type result_type;

    result_type::const_iterator end = stencils.end();
    for (result_type::const_iterator it = stencils.begin(); it != end; ++it) 
    {
        
    }
}

///////////////////////////////////////////////////////////////////////////////
// Get gids of output ports of all stencils
void get_output_ports(
    components::distributing_factory::result_type const& stencils,
    std::vector<std::vector<naming::id_type> >& outputs)
{
    typedef components::distributing_factory::result_type result_type;

    result_type::const_iterator end = stencils.end();
    for (result_type::const_iterator it = stencils.begin(); it != end; ++it) 
    {
        
    }
}

///////////////////////////////////////////////////////////////////////////////
// connectthe given output ports with the correct input ports, creating the 
// required static dataflow structure
void connect_input_ports(
    components::distributing_factory::result_type const& stencils,
    std::vector<std::vector<naming::id_type> > const& outputs)
{
    typedef components::distributing_factory::result_type result_type;

    result_type::const_iterator end = stencils.end();
    for (result_type::const_iterator it = stencils.begin(); it != end; ++it) 
    {
        
    }
}

///////////////////////////////////////////////////////////////////////////////
threads::thread_state 
hpx_main(threads::thread_self& self, applier::applier& appl)
{
    {
        // locally create a distributing factory
        components::distributing_factory dist_factory(
            components::distributing_factory::create(self, appl, 
                appl.get_runtime_support_gid(), true));

        // create a couple of stencil components
        typedef components::distributing_factory::result_type result_type;

        components::component_type stencil_type = 
            components::get_component_type<components::amr::stencil>();
        result_type stencils = dist_factory.create_components(self, stencil_type, 3);

        // create functional components and and use those to initialize stencils
        init_stencils(stencils);

        // ask stencil instances for their output gids
        std::vector<std::vector<naming::id_type> > outputs;
        get_output_ports(stencils, outputs);

        // connect output gids with corresponding stencil inputs
        connect_input_ports(stencils, outputs);

        // free all stencil components
        dist_factory.free_components(stencils);

    }   // distributing_factory needs to go out of scope before shutdown

    // initiate shutdown of the runtime systems on all localities
    components::stubs::runtime_support::shutdown_all(appl);

    return threads::terminated;
}

///////////////////////////////////////////////////////////////////////////////
bool parse_commandline(int argc, char *argv[], po::variables_map& vm)
{
    try {
        po::options_description desc_cmdline ("Usage: hpx_runtime [options]");
        desc_cmdline.add_options()
            ("help,h", "print out program usage (this message)")
            ("run_dgas_server,r", "run DGAS server as part of this runtime instance")
            ("dgas,d", po::value<std::string>(), 
                "the IP address the DGAS server is running on (default taken "
                "from hpx.ini), expected format: 192.168.1.1:7912")
            ("hpx,x", po::value<std::string>(), 
                "the IP address the HPX parcelport is listening on (default "
                "is localhost:7910), expected format: 192.168.1.1:7913")
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
        std::cerr << "generic_client: exception caught: " << e.what() << std::endl;
        return false;
    }
    return true;
}

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
// helper class for DGAS server initialization
class dgas_server_helper
{
public:
    dgas_server_helper(std::string host, boost::uint16_t port)
      : dgas_pool_(), dgas_(dgas_pool_, host, port)
    {}

    void run (bool blocking)
    {
        dgas_.run(blocking);
    }

private:
    hpx::util::io_service_pool dgas_pool_; 
    hpx::naming::resolver_server dgas_;
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
        std::string hpx_host("localhost"), dgas_host;
        boost::uint16_t hpx_port = HPX_PORT, dgas_port = 0;

        // extract IP address/port arguments
        if (vm.count("dgas")) 
            split_ip_address(vm["dgas"].as<std::string>(), dgas_host, dgas_port);

        if (vm.count("hpx")) 
            split_ip_address(vm["hpx"].as<std::string>(), hpx_host, hpx_port);

        // initialize and run the DGAS service, if appropriate
        std::auto_ptr<dgas_server_helper> dgas_server;
        if (vm.count("run_dgas_server"))  // run the DGAS server instance here
            dgas_server.reset(new dgas_server_helper(dgas_host, dgas_port));

        // initialize and start the HPX runtime
        hpx::runtime rt(hpx_host, hpx_port, dgas_host, dgas_port);
        rt.run(hpx_main);
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

