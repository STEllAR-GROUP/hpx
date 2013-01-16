#include <hpx/hpx_init.hpp>
//#include <hpx/include/actions.hpp>
//#include "fft/fft.hpp"
#include <boost/assign/std.hpp>

#include <iostream>
#include <fstream>

#include "fft/fft_distrib.hpp"
#include "fft/fft.hpp"

#define default_szie 1024
using boost::program_options::variables_map;
using boost::program_options::options_description;

char const* const fft_symbolic_name = "/fft/fft_test";
////////////////////////////////////////////////////////////////////////////////
    typedef std::vector<fft::complex_type> complex_vector;
////////////////////////////////////////////////////////////////////////////////

static void print_vector_contents(complex_vector const& x)
{
    typedef std::vector<fft::complex_type> complex_vector;
    complex_vector::const_iterator itr = x.begin();
    while(itr != x.end())
    {
        std::cout << itr->re << " " << itr->im << std::endl;
        ++itr;
    }
}
////////////////////////////////////////////////////////////////////////////////

static void write_to_file(std::ofstream& filename, const complex_vector& variable)
{                                                                                
    complex_vector::const_iterator itr = variable.begin();                       
    while(itr != variable.end())                                                 
    {                                                                            
        filename << itr->re << " " << itr->im << std::endl;                      
        ++itr;                                                                   
    }                                                                            
}                                                                                
                                                                                 
static void write_to_file_real(std::ofstream& filehandler, const complex_vector& variable)
{                                                                                
    complex_vector::const_iterator itr = variable.begin();                       
    while(itr != variable.end())                                                 
    {                                                                            
        filehandler << std::sqrt((itr->re*itr->re)+(itr->im*itr->im)) << std::endl; 
        ++itr;                                                                   
    }                                                                            
}
////////////////////////////////////////////////////////////////////////////////

int hpx_main(boost::program_options::variables_map& vm)
{
    {
        typedef std::vector<fft::complex_type> complex_vec;

        hpx::util::high_resolution_timer timer;
        std::ofstream fout("output_r2ditfft_real.txt");

        //hpx::lcos::future<complex_vec> result_vec;
        complex_vec final;
        std::string const datafilename = vm["file"].as<std::string>();
		//std::cout<<"Input filename at fft_client.cpp:" << datafilename << std::endl; 
        std::size_t num_workers = vm["num-workers"].as<std::size_t>();
        bool use_dataflow = vm["use-dataflow"].as<bool>();

        fft::fft_distrib distrib_obj(datafilename, fft_symbolic_name, num_workers
            , use_dataflow);
        
        /// instantiate and initialize. 
        distrib_obj.instantiate();
        distrib_obj.read_split_data();
		
		std::cout << "Component Creation and File read-split:" << timer.elapsed()
			<< " [s]" << std::endl;
		timer.restart();
        final = distrib_obj.transform();
        //complex_vec final = result_vec.get();
		std::cout << "FFT transform:" << timer.elapsed()
			<< " [s]" << std::endl;
		timer.restart();

        write_to_file_real(fout, final);
		std::cout << "Final file output:" << timer.elapsed()
			<< " [s]" << std::endl;
		

        std::cout << "Finished fft." << std::endl;
        
    }
    return hpx::finalize();
}
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    using boost::program_options::options_description;
    using boost::program_options::value;

    // Application specific options
    options_description cmdline("Usage: " HPX_APPLICATION_STRING " [options]");

    cmdline.add_options()
        ("file", value<std::string>()->default_value("fft_input.txt"), 
            "name of FFT data file")
        ("num-workers", value<std::size_t>()->default_value(1),
            "number of workers/threads to create per locality")
        ("use-dataflow", value<bool>()->default_value(false),
            "choose dataflow version of fft")
    ;
    // Initialize and run HPX
    return hpx::init(cmdline, argc, argv);
}
