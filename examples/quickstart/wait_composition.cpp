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

    result_type operator()(hpx::future<std::vector<hpx::future<int> > > f) const
    {
        for (std::size_t i = 0; i < f.get().size(); ++i)
            std::cout << f.get()[i].get() << "\n";
    }
};

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    {
        hpx::future<int> a = hpx::lcos::make_future<int>(17);
        hpx::future<int> b = hpx::lcos::make_future<int>(42);
        hpx::future<int> c = hpx::lcos::make_future<int>(-1);

        hpx::when_all(a, b, c).then(cout_continuation());
    }

    return hpx::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    return hpx::init(argc, argv); // Initialize and run HPX.
}
