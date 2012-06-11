////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/util/merging_map.hpp>

#include <boost/format.hpp>

using hpx::util::merging_map;

template <
    typename T
>
struct incrementer
{
  private:
    T const amount_;

  public:
    explicit incrementer(
        T const& amount
        )
      : amount_(amount)
    {
        BOOST_ASSERT(amount);
    }

    incrementer(
        incrementer const& other
        )
      : amount_(other.amount_)
    {}

    void operator()(
        T& v
        ) const
    {
        std::cout << "incrementing " << v << " by " << amount_ << "\n";
        v += amount_;
    }
};

template <
    typename T
>
struct decrementer
{
  private:
    T const amount_;

  public:
    decrementer(
        T const& amount
        )
      : amount_(amount)
    {
        BOOST_ASSERT(amount);
    }

    decrementer(
        decrementer const& other
        )
      : amount_(other.amount_)
    {}

    void operator()(
        T& v
        ) const
    {
        // We don't worry about removing entries when they're at 0. The AGAS
        // server code handles this after all the counts have been updated.
        if (amount_ >= v)
        {
            std::cout << "decrementing " << v << " by " << v << "\n";
            v = 0;
        }
        else
        {
            std::cout << "decrementing " << v << " by " << amount_ << "\n";
            v -= amount_;
        }
    }
};

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

    std::cout << "\n";

    // TODO: Add more find tests.
    {
        typedef merging_map<boost::uint32_t, boost::uint32_t> map_type;

        typedef map_type::iterator iterator;

        map_type table;

        table.bind(2, 2, 50U); // [2, 2]

        std::pair<iterator, iterator> r = table.find(2);

        for (; r.first != r.second; ++r.first)
        {
            std::cout << r.first->key_  << " -> " << r.first->data_ << "\n";
        }
    }

    std::cout << "\n";

    // TODO: Add more apply tests.
    {
        typedef merging_map<boost::uint32_t, boost::uint32_t> map_type;

        map_type table;

        table.bind(2, 3,  50U); // [2, 3]
        table.bind(5, 6,  70U); // [5, 6]
        table.bind(8, 11, 60U); // [8, 11]

        table.apply(3, 10, incrementer<boost::uint32_t>(35)); // [3, 10]

        BOOST_FOREACH(map_type::const_reference e, table)
        {
            std::cout << e.key_ << " -> " << e.data_ << "\n";
        }

        table.apply(0, 11, decrementer<boost::uint32_t>(300)); // [0, 11]

        std::cout << "--\n";

        BOOST_FOREACH(map_type::const_reference e, table)
        {
            std::cout << e.key_ << " -> " << e.data_ << "\n";
        }
    }

    std::cout << "\n";

    // TODO: Add more apply tests.
    {
        typedef merging_map<boost::uint32_t, boost::uint32_t> map_type;

        map_type table;

        table.bind(2, 3,  125U); // [2, 3]
        table.bind(5, 6,  110U); // [5, 6]
        table.bind(6, 7,  75U);  // [7, 7]
        table.bind(8, 15, 85U);  // [8, 15]

        std::cout << "-- initial\n";

        BOOST_FOREACH(map_type::const_reference e, table)
        {
            std::cout << e.key_ << " -> " << e.data_ << "\n";
        }

        table.apply(3, 11, decrementer<boost::uint32_t>(50), 100U); // [3, 11]

        std::cout << "-- [3, 11] -= 50 (default 100)\n";

        BOOST_FOREACH(map_type::const_reference e, table)
        {
            std::cout << e.key_ << " -> " << e.data_ << "\n";
        }

        table.apply(3, 11, decrementer<boost::uint32_t>(60), 100U); // [3, 11]

        std::cout << "-- [3, 11] -= 60 (default 100)\n";

        BOOST_FOREACH(map_type::const_reference e, table)
        {
            std::cout << e.key_ << " -> " << e.data_ << "\n";
        }
    }

    std::cout << "\n";

    // TODO: Add more apply tests.
    {
        typedef merging_map<boost::uint32_t, boost::uint32_t> map_type;

        map_type table;

        table.bind(2, 3,  125U); // [2, 3]
        table.bind(5, 6,  110U); // [5, 6]
        table.bind(6, 7,  75U);  // [7, 7]
        table.bind(8, 15, 85U);  // [8, 15]

        std::cout << "-- initial\n";

        BOOST_FOREACH(map_type::const_reference e, table)
        {
            std::cout << e.key_ << " -> " << e.data_ << "\n";
        }

        table.apply(5, 5, decrementer<boost::uint32_t>(50), 100U); // [5, 5]

        std::cout << "-- [5, 5] -= 50 (default 100)\n";

        BOOST_FOREACH(map_type::const_reference e, table)
        {
            std::cout << e.key_ << " -> " << e.data_ << "\n";
        }

        table.apply(5, 5, decrementer<boost::uint32_t>(60), 100U); // [5, 5]

        std::cout << "-- [5, 5] -= 60 (default 100)\n";

        BOOST_FOREACH(map_type::const_reference e, table)
        {
            std::cout << e.key_ << " -> " << e.data_ << "\n";
        }
    }

    std::cout << "\n";

    // TODO: Add more apply tests.
    {
        typedef merging_map<boost::uint32_t, boost::uint32_t> map_type;
        typedef map_type::key_type key_type;

        boost::format fmt("[0x%04X, 0x%04X] -> %u\n");

        key_type const key(0x00FF, 0x00FF); // [0x00FF, 0x00FF]

        map_type table;

        table.bind(0x0000, 0xFFFF, 255U); // [0x0000, 0xFFFF]

        std::cout << "-- initial\n";

        BOOST_FOREACH(map_type::const_reference e, table)
        {
            std::cout << (fmt % boost::icl::lower(e.key_)
                              % boost::icl::upper(e.key_)
                              % e.data_);
        }

        table.apply(key, decrementer<boost::uint32_t>(127), 255U);

        std::cout << "-- [0x00FF, 0x00FF] -= 127 (default 255)\n";

        BOOST_FOREACH(map_type::const_reference e, table)
        {
            std::cout << (fmt % boost::icl::lower(e.key_)
                              % boost::icl::upper(e.key_)
                              % e.data_);
        }

        table.apply(key, decrementer<boost::uint32_t>(128), 255U);

        std::cout << "-- [0x00FF, 0x00FF] -= 128 (default 255)\n";

        BOOST_FOREACH(map_type::const_reference e, table)
        {
            std::cout << (fmt % boost::icl::lower(e.key_)
                              % boost::icl::upper(e.key_)
                              % e.data_);
        }
    }

    std::cout << "\n";

    // TODO: Add more apply tests.
    {
        typedef merging_map<boost::uint32_t, boost::uint32_t> map_type;

        map_type table;

        table.bind(2, 3, 50U); // [2, 3]

        table.apply(3, 3, incrementer<boost::uint32_t>(35)); // [3, 3]

        BOOST_FOREACH(map_type::const_reference e, table)
        {
            std::cout << e.key_ << " -> " << e.data_ << "\n";
        }
    }
}

