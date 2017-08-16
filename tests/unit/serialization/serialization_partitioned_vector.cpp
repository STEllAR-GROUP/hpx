//  Copyright (c) 2017 Christopher Taylor
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>

#include <hpx/include/parallel_fill.hpp>
#include <hpx/include/partitioned_vector.hpp>
#include <hpx/include/serialization.hpp>

#include <hpx/util/lightweight_test.hpp>

#include <cstddef>
#include <numeric>
#include <vector>

HPX_REGISTER_PARTITIONED_VECTOR(int);
HPX_REGISTER_PARTITIONED_VECTOR(unsigned);

typedef unsigned long ulong;

HPX_REGISTER_PARTITIONED_VECTOR(ulong);
HPX_REGISTER_PARTITIONED_VECTOR(long);
HPX_REGISTER_PARTITIONED_VECTOR(double);

template <typename T>
void test(T minval, T maxval)
{
    {
        std::vector<char> buffer;

        hpx::serialization::output_archive oarchive(buffer,
            hpx::serialization::disable_data_chunking);

        std::size_t sz = static_cast<std::size_t>(maxval-minval);

        hpx::partitioned_vector<T> os(sz);
        os.register_as("test_vector");
        hpx::parallel::fill(
            hpx::parallel::execution::par, std::begin(os), std::end(os), 42);

        oarchive << os;

        hpx::serialization::input_archive iarchive(buffer);

        hpx::partitioned_vector<T> is(os.size());
        hpx::parallel::fill(
            hpx::parallel::execution::par, std::begin(is), std::end(is), 0);

        iarchive >> is;

        HPX_TEST_EQ(os.size(), is.size());
        for (std::size_t i = 0; i != os.size(); ++i)
        {
            HPX_TEST_EQ(os[i], is[i]);
        }
    }
}

int main()
{
    test<int>((std::numeric_limits<int>::min)(),
        (std::numeric_limits<int>::min)() + 100);
    test<int>((std::numeric_limits<int>::max)() - 100,
        (std::numeric_limits<int>::max)());
    test<int>(-100, 100);
    test<unsigned>((std::numeric_limits<unsigned>::min)(),
        (std::numeric_limits<unsigned>::min)() + 100);
    test<unsigned>((std::numeric_limits<unsigned>::max)() - 100,
        (std::numeric_limits<unsigned>::max)());
    test<long>((std::numeric_limits<long>::min)(),
        (std::numeric_limits<long>::min)() + 100);
    test<long>((std::numeric_limits<long>::max)() - 100,
        (std::numeric_limits<long>::max)());
    test<long>(-100, 100);
    test<unsigned long>((std::numeric_limits<unsigned long>::min)(),
        (std::numeric_limits<unsigned long>::min)() + 100);
    test<unsigned long>((std::numeric_limits<unsigned long>::max)() - 100,
        (std::numeric_limits<unsigned long>::max)());
    test<double>((std::numeric_limits<double>::min)(),
        (std::numeric_limits<double>::min)() + 100);
    test<double>(-100, 100);

    return hpx::util::report_errors();
}

