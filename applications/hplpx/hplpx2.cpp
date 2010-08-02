#include "hplmatrex2/hplmatrex2.hpp"

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/program_options.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/export.hpp>

using namespace boost::posix_time;
using namespace hpx;
using namespace std;
namespace po=boost::program_options;

//the size of the matrix
unsigned int SIZE = 1024;
unsigned int ABSIZE = 512;
unsigned int BSIZE = 250;
double ERROR = 0;

/*This small program is used to perform LU decomposition on a randomly
generated matrix.  Unlike the standard HPL algorithm, partial pivoting
is not implemented.  This is for simplicity and takes into account
the fact that partial pivoting is rarely needed for randomly generated
matrices.  Upon completion of the computations, the error of the
generated solution is calculated and displayed as both cumulative
error and average error per part of solution, as well as the total
execution time from just before starting the AGAS server to the
statistical printout.*/

//////////////////////////////////////////////////////////////////
//this is where the data structure is created and the computation
//function is called
int hpx_main(){
	int i,j,t=0;

	std::vector<naming::gid_type> prefixes;
	applier::applier& appl = applier::get_applier();
	naming::gid_type prefix;
	if(appl.get_remote_prefixes(prefixes)){
		prefix = prefixes[0];
	}
	else{
		prefix = appl.get_runtime_support_raw_gid();
	}

	using hpx::components::HPLMatrex2;
	HPLMatrex2 dat;

	dat.create(naming::id_type(prefix,naming::id_type::unmanaged));

	dat.construct(SIZE,SIZE+1,ABSIZE,BSIZE);

	ERROR = dat.LUsolve();
	dat.destruct();
	dat.free();
	components::stubs::runtime_support::shutdown_all();
	return 0;
}

///////////////////////////////////////////////////////////////////////
bool parse_commandline(int argc, char *argv[], po::variables_map& vm)
{
    try {
        po::options_description desc_cmdline ("Usage: fibonacci [options]");
        desc_cmdline.add_options()
            ("help,h", "print out program usage (this message)")

            ("run_agas_server,r", "run AGAS server as part of this runtime instance")
            ("worker,w", "run this instance in worker (non-console) mode")
            ("config", po::value<std::string>(),
                "load the specified file as an application configuration file")
            ("agas,a", po::value<std::string>(),
                "the IP address the AGAS server is running on (default taken "
                "from hpx.ini), expected format: 192.168.1.1:7912")
            ("hpx,x", po::value<std::string>(),
                "the IP address the HPX parcelport is listening on (default "
                "is localhost:7910), expected format: 192.168.1.1:7913")
            ("localities,l", po::value<int>(),
                "the number of localities to wait for at application startup"
                "(default is 1)")
            ("threads,t", po::value<int>(),
                "the number of operating system threads to spawn for this"
                "HPX locality")
            ("queueing,q", po::value<std::string>(),
                "the queue scheduling policy to use, options are 'global' "
                "and 'local' (default is 'global')")
            ("size,S", po::value<int>(),
                "the height of the n x n+1 matrix generated, where n is the height "
		"and the minimum size value is 3(default is 1024)")
            ("csv,s", "generate statistics of the run in comma separated format")
            ("blocksize,b", po::value<int>(),
                "blocksize correlates to the amount of work performed by each "
		"thread during gaussian elimination (default is 256)")
	    ("allocblock,A", po::value<int>(),
		"allocblock effects the amount of work each thread performs "
		"during memory allocation, initialization, and during the "
		"final correctness check.  Only powers of 2 are accepted, "
		"all other values will be rounded down to the nearest power of 2"
		"(default is 512)");

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
        std::cerr << "error: exception caught: " << e.what() << std::endl;
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
        std::cerr << "warning: illegal port number given: " << v.substr(p+1) << std::endl;
        std::cerr << "           using default value instead: " << port << std::endl;
    }
}

///////////////////////////////////////////////////////////////////////////////

//AGAS server helper class, just like in the examples
class agas_server_helper{
public:
	agas_server_helper(std::string host, boost::uint16_t port)
		: agas_pool_(), agas_(agas_pool_, host, port)
	{agas_.run(false);}

private:
	hpx::util::io_service_pool agas_pool_;
	hpx::naming::resolver_server agas_;
};

/////////////////////////////////////////
//down here is the main function
int main(int argc, char* argv[]){
	try{
		//read in command line
		po::variables_map vm;
		if(!parse_commandline(argc, argv, vm)){
			return -1;
		}

		//initialize information
		std::string hpx_host("localhost"), agas_host;
		boost::uint16_t hpx_port = HPX_PORT, agas_port = 0;
		int num_threads = 1;
		int num_localities = 1;
		std::string queueing = "global";
		hpx::runtime::mode mode = hpx::runtime::console;

		// extract IP address/port arguments
        	if(vm.count("agas"))
            	    split_ip_address(vm["agas"].as<std::string>(), agas_host, agas_port);

	        if(vm.count("hpx"))
        	    split_ip_address(vm["hpx"].as<std::string>(), hpx_host, hpx_port);

	        if(vm.count("localities"))
        	    num_localities = vm["localities"].as<int>();

	        if(vm.count("threads"))
        	    num_threads = vm["threads"].as<int>();

	        if(vm.count("queueing"))
        	    queueing = vm["queueing"].as<std::string>();

	        if(vm.count("size")){
		    if(vm["size"].as<int>() > 0){
        	    	SIZE = vm["size"].as<int>();
		    }
		    else{
			std::cerr<<"warning: --size option ignored, input value is invalid"
				", using default\n";
		    }
		}

	        if(vm.count("worker")){
        	    mode = hpx::runtime::worker;
	            if(vm.count("config")){
        	        std::cerr<<"warning: --config option ignored, used for console "
                        	"instance only\n";
	            }
        	}

		if(vm.count("blocksize")){
		    if((vm["blocksize"].as<int>() > 0) && (vm["blocksize"].as<int>() <= SIZE)){
			BSIZE = vm["blocksize"].as<int>();
		    }
		    else{
			std::cerr<<"warning: --blocksize option ignored, input value is "
				"invalid, using default\n";
		    }
		}

                if(vm.count("allocblock")){
                    if((vm["allocblock"].as<int>() > 0) && (vm["allocblock"].as<int>() <= SIZE)){
                        ABSIZE = vm["allocblock"].as<int>();
                    }
                    else{
                        std::cerr<<"warning: --allocblock option ignored, input value is "
                                "invalid, using default\n";
                    }
                }

		//now that the parameters have been set, we get a start time for the timing
		//of the application
		ptime start(microsec_clock::local_time());

		//initialize AGAS and start the HPX runtime
		std::auto_ptr<agas_server_helper> agas_server;
		if(vm.count("run_agas_server"))
			agas_server.reset(new agas_server_helper(agas_host, agas_port));
		//define the runtime type
		if(queueing == "global"){
			typedef hpx::runtime_impl<hpx::threads::policies::global_queue_scheduler>
			    runtime_type;
			runtime_type rt(hpx_host, hpx_port, agas_host, agas_port, mode);
                	if(mode == hpx::runtime::worker){
                	    rt.run(num_threads, num_localities);
                	}
                	else{
                	    // if we've got a configuration file (as console) we read it in,
                	    // otherwise this information will be automatically pulled from
                	    // the console
                	    if(vm.count("config")){
                	        std::string config(vm["config"].as<std::string>());
                	        rt.get_config().load_application_configuration(config.c_str());
                	    }

                	    rt.run(hpx_main, num_threads, num_localities);
                	}
		}
		else if(queueing == "local"){
			typedef hpx::runtime_impl<hpx::threads::policies::local_queue_scheduler>
			    runtime_type;
			hpx::threads::policies::local_queue_scheduler::init_parameter_type
			    init(num_threads, 1000);

                        runtime_type rt(hpx_host, hpx_port, agas_host, agas_port, mode, init);
                        if(mode == hpx::runtime::worker){
                            rt.run(num_threads, num_localities);
                        }
                        else{
                            // if we've got a configuration file (as console) we read it in,
                            // otherwise this information will be automatically pulled from
                            // the console
                            if(vm.count("config")){
                                std::string config(vm["config"].as<std::string>());
                                rt.get_config().load_application_configuration(config.c_str());
                            }

                            rt.run(hpx_main, num_threads, num_localities);
                        }
		}
		else{BOOST_ASSERT(false);}

		//output the results
               	if(vm.count("csv")){
            	    std::cout<<num_threads<<","<<SIZE<<","<<BSIZE<<","<<ERROR<<","
			<<ERROR/SIZE<<","<<ptime(microsec_clock::local_time())-start
			<<std::endl;
	        }
                else{
                    std::cout<<"total error: "
			<<ERROR<<std::endl<<"average error: "<<ERROR/SIZE<<std::endl
			<<"time elapsed: "<<ptime(microsec_clock::local_time())-start
			<<std::endl;
                }
	}
	catch(std::exception& e){
		std::cerr<<"std::exception caught: "<<e.what()<<std::endl;
		return -1;
	}
	catch(...){
		std::cerr<<"unexpected exception caught\n";
		return -2;
	}
	return 0;
}
