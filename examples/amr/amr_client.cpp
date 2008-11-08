//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <cstring>
#include <iostream>

#include <hpx/hpx.hpp>
#include <hpx/lcos/future_wait.hpp>
#include <hpx/components/distributing_factory/distributing_factory.hpp>
#include <hpx/components/amr/stencil_value.hpp>
#include <hpx/components/amr/functional_component.hpp>
#include <hpx/components/amr_test/stencil.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>
#include <boost/assign/std/vector.hpp>

namespace po = boost::program_options;

using namespace hpx;
using namespace std;

///////////////////////////////////////////////////////////////////////////////
// Create functional components, one for each data point, use those to 
// initialize the stencil value instances
void init_stencils(applier::applier& appl,
    components::distributing_factory::iterator_range_type const& stencils,
    components::distributing_factory::iterator_range_type const& functions)
{
    components::distributing_factory::iterator_type stencil = stencils.first;
    components::distributing_factory::iterator_type function = functions.first;

    for (/**/; stencil != stencils.second; ++stencil, ++function)
    {
        BOOST_ASSERT(function != functions.second);
        components::amr::stubs::stencil_value<3>::set_functional_component(
            appl, *stencil, *function);
    }
    BOOST_ASSERT(function == functions.second);
}

///////////////////////////////////////////////////////////////////////////////
// Get gids of output ports of all functions
void get_output_ports(threads::thread_self& self, applier::applier& appl,
    components::distributing_factory::iterator_range_type const& stencils,
    std::vector<std::vector<naming::id_type> >& outputs)
{
    typedef components::distributing_factory::result_type result_type;
    typedef 
        std::vector<lcos::future_value<std::vector<naming::id_type> > >
    lazyvals_type;

    // start an asynchronous operation for each of the stencil value instances
    lazyvals_type lazyvals;
    components::distributing_factory::iterator_type stencil = stencils.first;
    for (/**/; stencil != stencils.second; ++stencil)
    {
        lazyvals.push_back(components::amr::stubs::stencil_value<3>::
            get_output_ports_async(appl, *stencil));
    }

    // now wait for the results
    lazyvals_type::iterator lend = lazyvals.end();
    for (lazyvals_type::iterator lit = lazyvals.begin(); lit != lend; ++lit) 
    {
        outputs.push_back((*lit).get(self));
    }
}

///////////////////////////////////////////////////////////////////////////////
// Connect the given output ports with the correct input ports, creating the 
// required static data-flow structure.
//
// Currently we have exactly one stencil_value instance per data point, where 
// the output ports of a stencil_value are connected to the input ports of the 
// direct neighbors of itself.
inline std::size_t mod(int idx, std::size_t maxidx)
{
    return (idx < 0) ? (idx + maxidx) % maxidx : idx % maxidx;
}

void connect_input_ports(applier::applier& appl,
    components::distributing_factory::result_type const* stencils,
    std::vector<std::vector<std::vector<naming::id_type> > > const& outputs)
{
    typedef components::distributing_factory::result_type result_type;

    int steps = (int)outputs.size();
    for (int step = 0; step < steps; ++step) 
    {
        std::size_t numvals = outputs[0].size();
        components::distributing_factory::iterator_range_type r = 
            locality_results(stencils[step]);
        components::distributing_factory::iterator_type stencil = r.first;
        for (int i = 0; stencil != r.second; ++stencil, ++i)
        {
            using namespace boost::assign;
            int outstep = mod(step-1, steps);

            std::vector<naming::id_type> output_ports;
            output_ports += 
                    outputs[outstep][mod(i-1, numvals)][2],    // sw input is connected to the ne output of the sw element
                    outputs[outstep][mod(i  , numvals)][1],    // s input is connected to the n output of the s element 
                    outputs[outstep][mod(i+1, numvals)][0]     // se input is connected to the nw output of the se element
                ;

            components::amr::stubs::stencil_value<3>::
                connect_input_ports(appl, *stencil, output_ports);
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
void prepare_initial_data(threads::thread_self& self, applier::applier& appl, 
    components::distributing_factory::iterator_range_type const& functions, 
    std::vector<naming::id_type>& initial_data)
{
    typedef std::vector<lcos::future_value<naming::id_type> > lazyvals_type;

    // create a data item value type for each of the functions
    lazyvals_type lazyvals;
    components::distributing_factory::iterator_type function = functions.first;

    for (std::size_t i = 0; function != functions.second; ++function, ++i)
    {
        lazyvals.push_back(components::amr::stubs::functional_component::
            alloc_data_async(appl, *function, i));
    }

    // now wait for the results
    lazyvals_type::iterator lend = lazyvals.end();
    for (lazyvals_type::iterator lit = lazyvals.begin(); lit != lend; ++lit) 
    {
        initial_data.push_back((*lit).get(self));
    }
}

///////////////////////////////////////////////////////////////////////////////
// do actual work
void execute(threads::thread_self& self, applier::applier& appl, 
    components::distributing_factory::iterator_range_type const& stencils, 
    std::vector<naming::id_type> const& initial_data, 
    std::vector<naming::id_type>& result_data)
{
    // start the execution of all stencil stencils (data items)
    typedef std::vector<lcos::future_value<naming::id_type> > lazyvals_type;

    lazyvals_type lazyvals;
    components::distributing_factory::iterator_type stencil = stencils.first;
    for (std::size_t i = 0; stencil != stencils.second; ++stencil, ++i)
    {
        BOOST_ASSERT(i < initial_data.size());
        lazyvals.push_back(components::amr::stubs::stencil_value<3>::
            call_async(appl, *stencil, initial_data[i]));
    }

    // now wait for the results
    lazyvals_type::iterator lend = lazyvals.end();
    for (lazyvals_type::iterator lit = lazyvals.begin(); lit != lend; ++lit) 
    {
        result_data.push_back((*lit).get(self));
    }
}

///////////////////////////////////////////////////////////////////////////////
struct timestep_data
{
    int index_;       // sequential number of this datapoint
    int timestep_;    // current time step
    double value_;    // current value
};

///////////////////////////////////////////////////////////////////////////////
threads::thread_state 
hpx_main(threads::thread_self& self, applier::applier& appl, std::size_t numvals)
{
    // get component types needed below
    components::component_type function_type = 
        components::get_component_type<components::amr::stencil>();
    components::component_type stencil_type = 
        components::get_component_type<components::amr::server::stencil_value<3> >();

    {
        typedef components::distributing_factory::result_type result_type;

        // create a distributing factory locally
        components::distributing_factory factory(
            components::distributing_factory::create(self, appl, 
                appl.get_runtime_support_gid(), true));

        // create a couple of stencil (functional) components and the same 
        // amount of stencil_value components
        result_type functions = factory.create_components(self, function_type, numvals);
        result_type stencils[2] = 
        {
            factory.create_components(self, stencil_type, numvals),
            factory.create_components(self, stencil_type, numvals)
        };

        // initialize stencil_values using the stencil (functional) components
        init_stencils(appl, locality_results(stencils[0]), locality_results(functions));
        init_stencils(appl, locality_results(stencils[1]), locality_results(functions));

        // ask stencil instances for their output gids
        std::vector<std::vector<std::vector<naming::id_type> > > outputs(2);
        get_output_ports(self, appl, locality_results(stencils[0]), outputs[0]);
        get_output_ports(self, appl, locality_results(stencils[1]), outputs[1]);

        // connect output gids with corresponding stencil inputs
        connect_input_ports(appl, stencils, outputs);

        // prepare initial data
        std::vector<naming::id_type> initial_data;
        prepare_initial_data(self, appl, locality_results(functions), initial_data);

        // do actual work
        std::vector<naming::id_type> result_data;
        execute(self, appl, locality_results(stencils[0]), initial_data, result_data);

        // start asynchronous get operations
        components::stubs::memory_block stub(appl);

        // get some output memory_block_data instances
        components::access_memory_block<timestep_data> val1, val2, val3;
        boost::tie(val1, val2, val3) = 
            components::wait(self, stub.get_async(result_data[0]), 
                stub.get_async(result_data[1]), stub.get_async(result_data[2]));

        std::cout << "Result: " 
                  << val1->value_ << ", " 
                  << val2->value_ << ", " 
                  << val3->value_ << std::endl;

        // free all allocated components
        for (std::size_t i = 0; i < functions.size(); ++i) 
        {
            components::amr::stubs::functional_component::
                free_data(appl, functions[i].first_gid_, result_data[i]);
        }
        factory.free_components(stencils[1]);
        factory.free_components(stencils[0]);
        factory.free_components(functions);

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
            ("numvals,n", po::value<std::size_t>(), 
                "the number of data points to use for the computation")
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
        std::cerr << "amr_client: exception caught: " << e.what() << std::endl;
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
        std::cerr << "amr_client: illegal port number given: " << v.substr(p+1) << std::endl;
        std::cerr << "            using default value instead: " << port << std::endl;
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

        // extract IP address/port arguments
        if (vm.count("agas")) 
            split_ip_address(vm["agas"].as<std::string>(), agas_host, agas_port);

        if (vm.count("hpx")) 
            split_ip_address(vm["hpx"].as<std::string>(), hpx_host, hpx_port);

        if (vm.count("threads"))
            num_threads = vm["threads"].as<int>();

        // initialize and run the AGAS service, if appropriate
        std::auto_ptr<agas_server_helper> agas_server;
        if (vm.count("run_agas_server"))  // run the AGAS server instance here
            agas_server.reset(new agas_server_helper(agas_host, agas_port));

        std::size_t numvals = 3;
        if (vm.count("numvals"))
            numvals = vm["numvals"].as<std::size_t>();

        // initialize and start the HPX runtime
        hpx::runtime rt(hpx_host, hpx_port, agas_host, agas_port);
        rt.run(boost::bind (hpx_main, _1, _2, numvals), num_threads);
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

