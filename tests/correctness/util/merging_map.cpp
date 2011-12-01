////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/util/merging_map.hpp>

using hpx::util::merging_map;
using hpx::util::incrementer;
using hpx::util::decrementer;

///////////////////////////////////////////////////////////////////////////////
struct test_data
{
    boost::uint32_t x;
    boost::uint32_t y;
    char const* data;
};

template <
    std::size_t N
>
void run_test(
    test_data (&td) [N]
    )
{
    typedef merging_map<boost::uint32_t, std::string> map_type;

    std::cout << "\n";

    map_type map;

    for (std::size_t i = 0; i < N; ++i)
    {
        if (td[i].data)
            map.bind(td[i].x, td[i].y, td[i].data); 
    }

    BOOST_FOREACH(map_type::const_reference e, map)
    {
        std::cout << e.key_ << " -> " << e.data_ << "\n";
    }
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    test_data tests [][4] =
    {
        // Bind tests.
        {
            { 0,  1,  "a" }
          , { 1,  0,  "b" }
        }
      , {
            { 0,  7,  "a" }
          , { 0,  7,  "b" } 
        }
      , {
            { 0,  7,  "a" }
          , { 0,  0,  "b" } 
        }
      , {
            { 0,  7,  "a" }
          , { 7,  7,  "b" } 
        }
      , {
            { 0,  7,  "a" }
          , { 1,  6,  "b" } 
        }
      , {
            { 0,  7,  "a" }
          , { 5,  10, "b" } 
        }
      , {
            { 5,  10, "a" }
          , { 0,  7,  "b" } 
        }
      , {
            { 3,  3,  "a" }
          , { 4,  4,  "b" } 
          , { 5,  5,  "c" } 
          , { 3,  5,  "d" } 
        }
      , {
            { 0,  3,  "a" }
          , { 4,  4,  "b" } 
          , { 5,  8,  "c" } 
          , { 3,  5,  "d" } 
        }
      , {
            { 0,  3,  "a" }
          , { 5,  8,  "b" } 
          , { 3,  5,  "c" } 
        }
      , {
            { 0,  10, "a" }
          , { 13, 13, "b" } 
          , { 15, 20, "c" } 
          , { 9,  18, "d" } 
        }
      , {
            { 0,  1,  "a" }
          , { 1,  1,  "b" } 
        }

        // Merge tests.
      , {
            { 0,  1,  "z" }
          , { 1,  0,  "z" }
        }
      , {
            { 0,  7,  "z" }
          , { 0,  7,  "z" } 
        }
      , {
            { 0,  7,  "z" }
          , { 0,  0,  "z" } 
        }
      , {
            { 0,  7,  "z" }
          , { 7,  7,  "z" } 
        }
      , {
            { 0,  7,  "z" }
          , { 1,  6,  "z" } 
        }
      , {
            { 0,  7,  "z" }
          , { 5,  10, "z" } 
        }
      , {
            { 5,  10, "z" }
          , { 0,  7,  "z" } 
        }
      , {
            { 3,  3,  "z" }
          , { 4,  4,  "z" } 
          , { 5,  5,  "z" } 
          , { 3,  5,  "z" } 
        }
      , {
            { 0,  3,  "z" }
          , { 4,  4,  "z" } 
          , { 5,  8,  "z" } 
          , { 3,  5,  "z" } 
        }
      , {
            { 0,  3,  "z" }
          , { 5,  8,  "z" } 
          , { 3,  5,  "z" } 
        }
      , {
            { 0,  10, "z" }
          , { 13, 13, "z" } 
          , { 15, 20, "z" } 
          , { 9,  18, "z" } 
        }
      , {
            { 0,  1,  "z" }
          , { 1,  1,  "z" } 
        }
      , {
            { 0,  3,  "z" }
          , { 5,  8,  "z" } 
        }
    };

    std::size_t const num_tests = sizeof(tests) / sizeof(tests[0]);
    
    for (std::size_t i = 0; i < num_tests; ++i)
        run_test(tests[i]);

    // TODO: Add more apply tests.
    {
        typedef merging_map<boost::uint32_t, boost::uint32_t>
            map_type;
    
        map_type table;
    
        table.bind(2, 3,  50U); // [2, 3] 
        table.bind(5, 6,  70U); // [5, 6]
        table.bind(8, 11, 60U); // [8, 11]
    
        table.apply(3, 10, incrementer<boost::uint32_t>(35)); // [3, 10]

        std::cout << "\n";
    
        BOOST_FOREACH(map_type::const_reference e, table)
        {
            std::cout << e.key_ << " -> " << e.data_ << "\n";
        }

        table.apply(0, 11, decrementer<boost::uint32_t>(300)); // [0, 11]

        std::cout << "\n";

        BOOST_FOREACH(map_type::const_reference e, table)
        {
            std::cout << e.key_ << " -> " << e.data_ << "\n";
        }
    }
}

