//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2015 Andreas Schaefer
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/modules/serialization.hpp>
#include <hpx/modules/testing.hpp>

#include <set>
#include <vector>

template <typename Cargo>
struct DummyContainer
{
    DummyContainer() = default;

    explicit DummyContainer(Cargo cargo)
      : cargo(cargo)
    {
    }

    template <typename Archive>
    void serialize(Archive& archive, unsigned)
    {
        archive & cargo;
    }

    bool operator<(DummyContainer<Cargo> const other) const
    {
        return cargo < other.cargo;
    }

    Cargo cargo;
};

void test_int()
{
    {
        std::vector<char> buffer;
        hpx::serialization::output_archive oarchive(buffer);

        std::multiset<int> os;
        os.insert(-1000);
        os.insert(12345);
        os.insert(34567);
        os.insert(-2000);
        oarchive << os;

        hpx::serialization::input_archive iarchive(buffer);
        std::multiset<int> is;
        iarchive >> is;
        HPX_TEST_EQ(os.size(), is.size());

        std::multiset<int>::iterator j = is.begin();
        for (std::multiset<int>::iterator i = os.begin(); i != os.end(); ++i, ++j)
        {
            HPX_TEST_EQ(*i, *j);
        }
    }
}

template <typename T>
void test(T min, T max)
{
    {
        std::vector<char> buffer;
        hpx::serialization::output_archive oarchive(buffer);
        std::multiset<T> os;
        for (T c = min; c < max; ++c)
        {
            os.insert(c);

            // inserting duplicates to test multiset
            os.insert(c);
            os.insert(c);
        }
        oarchive << os;
        hpx::serialization::input_archive iarchive(buffer);
        std::multiset<T> is;
        iarchive >> is;
        HPX_TEST_EQ(os.size(), is.size());

        typename std::multiset<T>::iterator j = is.begin();
        for (typename std::multiset<T>::iterator i = os.begin(); i != os.end(); ++i)
        {
            HPX_TEST_EQ(*i, *j);
            ++j;
        }
    }
    {
        std::vector<char> buffer;
        hpx::serialization::output_archive oarchive(buffer);
        std::multiset<DummyContainer<T>> os;
        for (T c = min; c < max; ++c)
        {
            os.emplace(c);

            // inserting duplicates to test multiset
            os.emplace(c);
            os.emplace(c);
        }
        oarchive << os;
        hpx::serialization::input_archive iarchive(buffer);
        std::multiset<DummyContainer<T>> is;
        iarchive >> is;
        HPX_TEST_EQ(os.size(), is.size());

        typename std::multiset<DummyContainer<T>>::iterator j = is.begin();
        for (typename std::multiset<DummyContainer<T>>::iterator i = os.begin();
            i != os.end(); ++i)
        {
            HPX_TEST_EQ(i->cargo, j->cargo);
            ++j;
        }
    }
}

template <typename T>
void test_fp(T min, T max)
{
    {
        std::vector<char> buffer;
        hpx::serialization::output_archive oarchive(buffer);
        std::multiset<T> os;
        for (T c = min; c < max; c += static_cast<T>(0.5))
        {
            os.insert(c);

            // inserting duplicates to test multiset
            os.insert(c);
            os.insert(c);
        }
        oarchive << os;
        hpx::serialization::input_archive iarchive(buffer);
        std::multiset<T> is;
        iarchive >> is;
        HPX_TEST_EQ(os.size(), is.size());

        typename std::multiset<T>::iterator j = is.begin();
        for (typename std::multiset<T>::iterator i = os.begin(); i != os.end(); ++i)
        {
            HPX_TEST_EQ(*i, *j);
            ++j;
        }
    }
    {
        std::vector<char> buffer;
        hpx::serialization::output_archive oarchive(buffer);
        std::multiset<DummyContainer<T>> os;
        for (T c = min; c < max; c += static_cast<T>(0.5))
        {
            os.emplace(c);

            // inserting duplicates to test multiset
            os.emplace(c);
            os.emplace(c);
        }
        oarchive << os;
        hpx::serialization::input_archive iarchive(buffer);
        std::multiset<DummyContainer<T>> is;
        iarchive >> is;
        HPX_TEST_EQ(os.size(), is.size());

        typename std::multiset<DummyContainer<T>>::iterator j = is.begin();
        for (typename std::multiset<DummyContainer<T>>::iterator i = os.begin();
            i != os.end(); ++i)
        {
            HPX_TEST_EQ(i->cargo, j->cargo);
            ++j;
        }
    }
}

int main()
{
    test_int();
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
    test_fp<float>((std::numeric_limits<float>::max)() - 100,
        (std::numeric_limits<float>::max)());    //it's incorrect
    // because floatmax() - 100 causes cancellations error, digits are not affected
    test_fp<float>(-100, 100);
    test<double>((std::numeric_limits<double>::min)(),
        (std::numeric_limits<double>::min)() + 100);
    test<double>((std::numeric_limits<double>::max)() - 100,
        (std::numeric_limits<double>::max)());    //it's the same
    test<double>(-100, 100);

    return hpx::util::report_errors();
}
