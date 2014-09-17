//  (C) Copyright 2013 Damond Howard
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/util.hpp>
#include <hpx/include/lcos.hpp>
#include <vector>
#include <random>
#include <boost/foreach.hpp>

double calculate_pi(boost::uint64_t num_of_iterations);
double  check_if_hit();

HPX_PLAIN_ACTION(calculate_pi,calculate_pi_action);
HPX_PLAIN_ACTION(check_if_hit,check_if_hit_action);

double calculate_pi(boost::uint64_t num_of_iterations)
{
    boost::atomic<uint64_t> hits(0);

    std::vector<hpx::naming::id_type> localities = hpx::find_all_localities();
    std::vector<hpx::lcos::future<double> > futures;
    futures.reserve(num_of_iterations);
    for(uint64_t i=0;i<num_of_iterations;i++)
    {
        BOOST_FOREACH(hpx::naming::id_type const& node, localities)
        {
            futures.push_back(hpx::async<check_if_hit_action>(node));
        }
    }

    hpx::wait(futures,
        [&](std::size_t ,double d)
        {
            if (d <= 1)
                hits++;
        });

    std::cout<<"localities "<<localities.size()<<std::endl;
    std::cout<<"hits "<<hits<<std::endl;
    std::cout<<"iterations "<<num_of_iterations<<std::endl;
    return (double)hits/num_of_iterations*4.0;
}

double check_if_hit()
{
    static std::default_random_engine generator(std::time(0));
    static std::uniform_real_distribution<double> distribution(0.0,0.999999999999999);

    double x = distribution(generator);
    double y = distribution(generator);
    double z = x*x+y*y;
    return z;
}

int main(int argc,char* argv[])
{
    boost::program_options::options_description
        desc_commandline("Usage: " HPX_APPLICATION_STRING "[options]");

    desc_commandline.add_options()
        ( "iterations",
        boost::program_options::value<boost::uint64_t>()->default_value(100),
        "number of iterations");
    return hpx::init(desc_commandline,argc,argv);
}

int hpx_main(boost::program_options::variables_map& vm)
{
    boost::uint64_t num_of_iterations = vm["iterations"].as<boost::uint64_t>();
    hpx::util::high_resolution_timer t;

    double pi = calculate_pi(num_of_iterations);
    std::cout<<pi<<std::endl;
    std::cout<<t.elapsed()<<std::endl;
    return hpx::finalize();
}
