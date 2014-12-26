//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define HPX_LIMIT 7

#include <hpx/hpx_main.hpp>
#include <hpx/components/vector/vector.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <hpx/include/parallel_for_each.hpp>
#include <hpx/parallel/segmented_algorithms/transform_reduce.hpp>

///////////////////////////////////////////////////////////////////////////////
// Define the vector types to be used.
HPX_REGISTER_VECTOR(double);
HPX_REGISTER_VECTOR(int);

struct multiply
{
    template <typename T>
    T operator()(hpx::util::tuple<T, T> const& r) const
    {
        using hpx::util::get;
        return get<0>(r) * get<1>(r);
    }
};

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename T>
T test_transform_reduce(ExPolicy && policy, hpx::vector<T> const& xvalues,
    hpx::vector<T> const& yvalues)
{
    using hpx::util::make_zip_iterator;
    return
        hpx::parallel::transform_reduce(policy,
            make_zip_iterator(boost::begin(xvalues), boost::begin(yvalues)),
            make_zip_iterator(boost::end(xvalues), boost::end(yvalues)),
            multiply(), T(0), std::plus<T>()
        );
}

template <typename T>
void transform_reduce_tests()
{
    std::size_t const num = 10007;

    {
        hpx::vector<T> xvalues(num, T(1));
        hpx::vector<T> yvalues(num, T(1));

        HPX_TEST_EQ(
            test_transform_reduce(hpx::parallel::seq, xvalues, yvalues),
            T(num));
        HPX_TEST_EQ(
            test_transform_reduce(hpx::parallel::par, xvalues, yvalues),
            T(num));
    }
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
//     transform_reduce_tests<int>();
    transform_reduce_tests<double>();

    return 0;
}

