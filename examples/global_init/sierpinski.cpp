////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Satyaki Upadhyay
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////


// http://www.wikiwand.com/en/Sierpinski_triangle
// At each iteration, a white triangle in deleted from the middle
// of every black triangle,
// thereby splitting it into three equal triangles.

// Takes two program options, the number of iterations (n-value, with default 5)
// and the side length of the original triangle (side-length, with default 100)

#include <hpx/hpx.hpp>
#include "hpx_initialize.hpp"
#include <hpx/include/actions.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/util.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/serialization.hpp>

#include <cstdint>
#include <iostream>


// Class representing a Sierpinski triangle
class sierpinski{
    public:
    std::uint64_t black_triangles, white_triangles;
    double area;

    sierpinski(){}
    sierpinski(std::uint64_t black, std::uint64_t white, double area){
        this->black_triangles = black;
        this->white_triangles = white;
        this->area = area;
    }

    inline sierpinski operator+(const sierpinski& other) const {
        sierpinski res = sierpinski (
            black_triangles+other.black_triangles,
            white_triangles+other.white_triangles,
            area+other.area);
        return res;
    }


    private:
    //Serialization is necessary to transmit objects from one locality to another
    friend class hpx::serialization::access;
    template <class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & black_triangles;
        ar & white_triangles;
        ar & area;
    }
};


///////////////////////////////////////////////////////////////////////////////
// forward declaration of the get_sierpinski function
sierpinski get_sierpinski(std::uint64_t n, double len);


// This is to generate the required boilerplate we need for the remote
// invocation to work.
HPX_PLAIN_ACTION(get_sierpinski, get_sierpinski_action);


///////////////////////////////////////////////////////////////////////////////
sierpinski get_sierpinski(std::uint64_t n, double len)
{
    if (n == 0)
        return sierpinski(1, 0, sqrt(3)/4.0*len*len);

    hpx::naming::id_type const locality_id = hpx::find_here();

    // The problem for iteration n is broken down into three sub-problems,
    // each of size n-1 (and side length gets halved). It is very inefficient
    // as it does the same calculations
    // three times but this is the method used for the sake of using HPX.

    get_sierpinski_action s;
    hpx::future<sierpinski> n1 = hpx::async(s, locality_id, n - 1, len/2);
    hpx::future<sierpinski> n2 = hpx::async(s, locality_id, n - 1, len/2);
    hpx::future<sierpinski> n3 = hpx::async(s, locality_id, n - 1, len/2);

    sierpinski ans = n1.get() + n2.get() + n3.get();
    ans.white_triangles++;
    return ans;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    using boost::program_options::options_description;
		using boost::program_options::value;
		using boost::program_options::variables_map;

    // Configure application-specific options
    options_description
        desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ( "n-value",
          boost::program_options::value<std::uint64_t>()->default_value(5),
          "n value for the Sierpinski function")
        ;
    desc_commandline.add_options()
        ( "side-length",
          boost::program_options::value<std::uint64_t>()->default_value(100),
          "side-length of the original triangle")
        ;

    // Add application-specific options into variables_map
		variables_map vm;
		store(parse_command_line(argc, argv, desc_commandline), vm);
		notify(vm);

		std::uint64_t n = vm["n-value"].as<std::uint64_t>();
		std::uint64_t len = vm["side-length"].as<std::uint64_t>();

		{
		    get_sierpinski_action s;
		    sierpinski ans = s(hpx::find_here(), n, (double)len);

		    std::cout << "After iteration: " << n << std::endl;
		    std::cout << "Black triangles: " << ans.black_triangles << std::endl;
		    std::cout << "White triangles: " << ans.white_triangles << std::endl;
		    std::cout << "Area: " << ans.area << std::endl;
		}

    return hpx::finalize();
}
