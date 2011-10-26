////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/hpx_init.hpp>
#include <hpx/util/function.hpp>

#include <boost/serialization/access.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;

using hpx::util::function;

using hpx::init;
using hpx::finalize;

///////////////////////////////////////////////////////////////////////////////
struct foo
{
  private:
    boost::uint64_t x_;
    boost::uint64_t y_;

    friend class boost::serialization::access;

    template <
        typename Archive
    >
    void serialize(
        Archive& ar
      , unsigned const
        )
    {
        ar & x_;
        ar & y_;
    }

  public:
    foo(
        boost::uint64_t x = 0
      , boost::uint64_t y = 0
        )
      : x_(x)
      , y_(y)
    {}

    void operator()()
    {
        std::cout << "x: " << x_ << "\n"
                  << "y: " << y_ << "\n"
                  ;
    } 
};

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    {
        foo f(17, 19);

        function<void()> f0(f);

        function<void()> f1(f0), f2;

        f2 = f0;

        f0();
        f1();
        f2();
    }

    finalize();

    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description cmdline("Usage: " HPX_APPLICATION_STRING " [options]");

    // Initialize and run HPX
    return init(cmdline, argc, argv);
}

