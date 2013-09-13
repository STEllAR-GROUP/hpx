////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>

///////////////////////////////////////////////////////////////////////////////
struct cout_continuation
{
    typedef void result_type;
    typedef HPX_STD_TUPLE<
            hpx::lcos::future<int>
          , hpx::lcos::future<int>
          , hpx::lcos::future<int> > data_type;

    result_type operator()(
        hpx::lcos::future<data_type> data
    ) const
    {
        data_type v = data.move();
        std::cout << HPX_STD_GET(0, v).get() << "\n";
        std::cout << HPX_STD_GET(1, v).get() << "\n";
        std::cout << HPX_STD_GET(2, v).get() << "\n";
    }
};
///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    {
        hpx::future<int> a = hpx::lcos::make_ready_future<int>(17);
        hpx::future<int> b = hpx::lcos::make_ready_future<int>(42);
        hpx::future<int> c = hpx::lcos::make_ready_future<int>(-1);

        hpx::when_all(a, b, c).then(cout_continuation());
    }

    return hpx::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    return hpx::init(argc, argv); // Initialize and run HPX.
}
