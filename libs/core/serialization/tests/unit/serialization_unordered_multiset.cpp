//  Copyright (c) 2026 Ujjwal Shekhar
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/modules/serialization.hpp>
#include <hpx/modules/testing.hpp>

#include <algorithm>
#include <iterator>
#include <numeric>
#include <unordered_set>
#include <utility>
#include <vector>

template <typename T>
struct A
{
    A() {}

    explicit A(T t)
      : t_(t)
    {
    }
    T t_;

    A& operator=(T t)
    {
        t_ = t;
        return *this;
    }
    bool operator==(A const& a) const
    {
        return t_ == a.t_;
    }

    template <typename Archive>
    void serialize(Archive& ar, unsigned)
    {
        ar & t_;
    }
};

// Hash function for A
template <typename T>
struct std::hash<A<T>>
{
    std::size_t operator()(A<T> const& a) const noexcept
    {
        return std::hash<T>()(a.t_);
    }
};

// Hash function for std::vector<int>
template <>
struct std::hash<std::vector<int>>
{
    std::size_t operator()(std::vector<int> const& v) const noexcept
    {
        return std::hash<std::size_t>()(v.size());
    }
};

template <class T>
std::ostream& operator<<(std::ostream& os, A<T> const& a)
{
    return os << a.t_;
}

template <typename T>
void test(T min, T max)
{
    {
        std::vector<char> buffer;
        hpx::serialization::output_archive oarchive(buffer);
        std::unordered_multiset<T, std::hash<T>, std::equal_to<T>,
            std::allocator<T>>
            os;
        for (T c = min; c < max; ++c)
        {
            os.insert(c);

            // inserting duplicates to test multiset
            os.insert(c);
            os.insert(c);
        }
        oarchive << os;
        hpx::serialization::input_archive iarchive(buffer);
        std::unordered_multiset<T, std::hash<T>, std::equal_to<T>,
            std::allocator<T>>
            is;
        iarchive >> is;
        HPX_TEST_EQ(os.size(), is.size());
        for (auto i = os.begin(), j = is.begin();
            i != os.end() && j != is.end(); ++i, ++j)
        {
            HPX_TEST_EQ(os.count(*i), is.count(*i));
            HPX_TEST_EQ(os.count(*j), is.count(*j));
        }
    }
    {
        std::vector<char> buffer;
        hpx::serialization::output_archive oarchive(buffer);
        std::unordered_multiset<A<T>, std::hash<A<T>>, std::equal_to<A<T>>,
            std::allocator<A<T>>>
            os;
        for (T c = min; c < max; ++c)
        {
            os.insert(A<T>(c));

            // inserting duplicates to test multiset
            os.insert(A<T>(c));
            os.insert(A<T>(c));
        }
        oarchive << os;
        hpx::serialization::input_archive iarchive(buffer);
        std::unordered_multiset<A<T>, std::hash<A<T>>, std::equal_to<A<T>>,
            std::allocator<A<T>>>
            is;
        iarchive >> is;
        HPX_TEST_EQ(os.size(), is.size());
        for (auto i = os.begin(), j = is.begin();
            i != os.end() && j != is.end(); ++i, ++j)
        {
            HPX_TEST_EQ(os.count(*i), is.count(*i));
            HPX_TEST_EQ(os.count(*j), is.count(*j));
        }
    }
}

template <typename T>
void test_fp(T min, T max)
{
    {
        std::vector<char> buffer;
        hpx::serialization::output_archive oarchive(buffer);
        std::unordered_multiset<T, std::hash<T>, std::equal_to<T>,
            std::allocator<T>>
            os;
        for (T c = min; c < max; c += static_cast<T>(0.5))
        {
            os.insert(c);

            // inserting duplicates to test multiset
            os.insert(c);
            os.insert(c);
        }
        oarchive << os;
        hpx::serialization::input_archive iarchive(buffer);
        std::unordered_multiset<T, std::hash<T>, std::equal_to<T>,
            std::allocator<T>>
            is;
        iarchive >> is;
        HPX_TEST_EQ(os.size(), is.size());
        for (auto i = os.begin(), j = is.begin();
            i != os.end() && j != is.end(); ++i, ++j)
        {
            HPX_TEST_EQ(os.count(*i), is.count(*i));
            HPX_TEST_EQ(os.count(*j), is.count(*j));
        }
    }
    {
        std::vector<char> buffer;
        hpx::serialization::output_archive oarchive(buffer);
        std::unordered_multiset<A<T>, std::hash<A<T>>, std::equal_to<A<T>>,
            std::allocator<A<T>>>
            os;
        for (T c = min; c < max; c += static_cast<T>(0.5))
        {
            os.insert(A<T>(c));

            // inserting duplicates to test multiset
            os.insert(A<T>(c));
            os.insert(A<T>(c));
        }
        oarchive << os;
        hpx::serialization::input_archive iarchive(buffer);
        std::unordered_multiset<A<T>, std::hash<A<T>>, std::equal_to<A<T>>,
            std::allocator<A<T>>>
            is;
        iarchive >> is;
        HPX_TEST_EQ(os.size(), is.size());
        for (auto i = os.begin(), j = is.begin();
            i != os.end() && j != is.end(); ++i, ++j)
        {
            HPX_TEST_EQ(os.count(*i), is.count(*i));
            HPX_TEST_EQ(os.count(*j), is.count(*j));
        }
    }
}

// prohibited, but for adl
namespace std {
    std::ostream& operator<<(std::ostream& os, std::vector<int> const& vec)
    {
        std::copy(vec.begin(), vec.end(), std::ostream_iterator<int>(os, " "));
        return os;
    }
}    // namespace std

void test_vector_as_value()
{
    std::vector<char> buffer;
    hpx::serialization::output_archive oarchive(buffer);
    std::unordered_multiset<std::vector<int>, std::hash<std::vector<int>>,
        std::equal_to<std::vector<int>>, std::allocator<std::vector<int>>>
        os;
    for (int k = 0; k < 10; ++k)
    {
        std::vector<int> vec(10);
        std::iota(vec.begin(), vec.end(), k);
        os.insert(vec);

        // inserting duplicates to test multiset
        os.insert(vec);
        os.insert(vec);
    }
    oarchive << os;
    hpx::serialization::input_archive iarchive(buffer);
    std::unordered_multiset<std::vector<int>, std::hash<std::vector<int>>,
        std::equal_to<std::vector<int>>, std::allocator<std::vector<int>>>
        is;
    iarchive >> is;
    HPX_TEST_EQ(os.size(), is.size());
    for (auto i = os.begin(), j = is.begin(); i != os.end() && j != is.end();
        ++i, ++j)
    {
        HPX_TEST_EQ(os.count(*i), is.count(*i));
        HPX_TEST_EQ(os.count(*j), is.count(*j));
    }
}

int main()
{
    test<char>(
        (std::numeric_limits<char>::min)(), (std::numeric_limits<char>::max)());
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
    test_fp<float>((std::numeric_limits<float>::min)(),
        (std::numeric_limits<float>::min)() + 100);
    test_fp<float>(-100, 100);
    test<double>((std::numeric_limits<double>::min)(),
        (std::numeric_limits<double>::min)() + 100);
    test<double>(-100, 100);

    test_vector_as_value();

    return hpx::util::report_errors();
}
