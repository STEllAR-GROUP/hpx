//  Copyright (c) 2013 Vinay C Amatya
//
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/util.hpp>
#include <hpx/include/lcos.hpp>

#include <boost/cstdint.hpp>
#include <boost/format.hpp>

#include <iostream>
#include <iterator>
#include <vector>
#include <utility>

#include <hpx/for_each.hpp>

template <typename T>
struct print_obj
{
public:
    typedef void result_type;

    template <typename F>
    struct result;

    template <typename F, typename A1>
    struct result<F(A1)>
    {
        typedef void type;
    };

    typename result<print_obj(T)>::type
        operator()(T const& input) const
    {
        std::cout << input << std::endl;
    }
};

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    typedef std::vector<std::size_t> vector_type;
    typedef std::vector<std::size_t>::iterator iterator_type;

    vector_type v1, v2;

    std::size_t count = 0;
    while (count < 2000)
    {
        v1.push_back(count);
        ++count;
    }

    iterator_type itr_b = v1.begin();
    iterator_type itr_e = v1.end();

    typedef boost::iterator_range<iterator_type> range_type;

    range_type my_range(itr_b, itr_e);

    v2.reserve(v1.size());
    iterator_type itr_o = v2.begin();

    hpx::for_each(my_range, print_obj<std::size_t>());

    return hpx::finalize(); // Handles HPX shutdown
}

int main(int argc, char* argv[])
{
    return hpx::init(argc, argv);
}
