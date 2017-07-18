//  Copyright (c) 2017 Christopher Taylor
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/serialization/partitioned_vector.hpp>
#include <hpx/runtime/serialization/input_archive.hpp>
#include <hpx/runtime/serialization/output_archive.hpp>

#include <hpx/include/partitioned_vector.hpp>

#include <hpx/util/lightweight_test.hpp>

#include <cstddef>
#include <numeric>

template <typename T>
void test(T minval, T maxval)
{
    {
        std::vector<char> buffer;
        hpx::serialization::output_archive oarchive(buffer,
            hpx::serialization::disable_data_chunking);

        std::size_t sz = static_cast<std::size_t>(maxval-minval);
        hpx::partitioned_vector<T> os(sz);

        T n = minval;
        hpx::parallel::generate(std::begin(os), std::end(os),
            [&n] { return ++n; });

        oarchive << os;

        hpx::serialization::input_archive iarchive(buffer);
        hpx::partitioned_vector<T> is(os.size());
        iarchive >> is;
        HPX_TEST_EQ(os.size(), is.size());
        for(std::size_t i = 0; i < os.size(); ++i)
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

