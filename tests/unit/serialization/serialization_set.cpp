//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2015 Andreas Schaefer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/serialization/set.hpp>

#include <hpx/runtime/serialization/input_archive.hpp>
#include <hpx/runtime/serialization/output_archive.hpp>

#include <hpx/util/lightweight_test.hpp>

template <typename CARGO>
struct DummyContainer
{
    DummyContainer()
    {}

    DummyContainer(CARGO cargo) :
        cargo(cargo)
    {}

    template <typename Archive>
    void serialize(Archive & archive, unsigned)
    {
        archive & cargo;
    }

    bool operator<(const DummyContainer<CARGO> other) const
    {
        return cargo < other.cargo;
    }

    CARGO cargo;
};

void test_int()
{
    {
        std::vector<char> buffer;
        hpx::serialization::output_archive oarchive(buffer);

        std::set<int> os;
        os.insert(-1000);
        os.insert(12345);
        os.insert(34567);
        os.insert(-2000);
        oarchive << os;

        hpx::serialization::input_archive iarchive(buffer);
        std::set<int> is;
        iarchive >> is;
        HPX_TEST_EQ(os.size(), is.size());

        std::set<int>::iterator j = is.begin();
        for(std::set<int>::iterator i = os.begin();
            i != os.end();
            ++i, ++j)
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
        std::set<T> os;
        for(T c = min; c < max; ++c)
        {
            os.insert(c);
        }
        oarchive << os;
        hpx::serialization::input_archive iarchive(buffer);
        std::set<T> is;
        iarchive >> is;
        HPX_TEST_EQ(os.size(), is.size());

        typename std::set<T>::iterator j = is.begin();
        for(typename std::set<T>::iterator i = os.begin(); i != os.end(); ++i)
        {
            HPX_TEST_EQ(*i, *j);
            ++j;
        }
    }
    {
        std::vector<char> buffer;
        hpx::serialization::output_archive oarchive(buffer);
        std::set<DummyContainer<T> > os;
        for(T c = min; c < max; ++c)
        {
            os.insert(c);
        }
        oarchive << os;
        hpx::serialization::input_archive iarchive(buffer);
        std::set<DummyContainer<T> > is;
        iarchive >> is;
        HPX_TEST_EQ(os.size(), is.size());

        typename std::set<DummyContainer<T> >::iterator j = is.begin();
        for(typename std::set<DummyContainer<T> >::iterator i = os.begin(); i != os.end(); ++i)
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
        std::set<T> os;
        for(T c = min; c < max; c += static_cast<T>(0.5))
        {
            os.insert(c);
        }
        oarchive << os;
        hpx::serialization::input_archive iarchive(buffer);
        std::set<T> is;
        iarchive >> is;
        HPX_TEST_EQ(os.size(), is.size());

        typename std::set<T>::iterator j = is.begin();
        for(typename std::set<T>::iterator i = os.begin(); i != os.end(); ++i)
        {
            HPX_TEST_EQ(*i, *j);
            ++j;
        }
    }
    {
        std::vector<char> buffer;
        hpx::serialization::output_archive oarchive(buffer);
        std::set<DummyContainer<T> > os;
        for(T c = min; c < max; c += static_cast<T>(0.5))
        {
            os.insert(c);
        }
        oarchive << os;
        hpx::serialization::input_archive iarchive(buffer);
        std::set<DummyContainer<T> > is;
        iarchive >> is;
        HPX_TEST_EQ(os.size(), is.size());

        typename std::set<DummyContainer<T> >::iterator j = is.begin();
        for(typename std::set<DummyContainer<T> >::iterator i = os.begin(); i != os.end(); ++i)
        {
            HPX_TEST_EQ(i->cargo, j->cargo);
            ++j;
        }
    }
}

int main()
{
    test_int();
    test<char>((std::numeric_limits<char>::min)(), (std::numeric_limits<char>::max)());
    test<int>((std::numeric_limits<int>::min)(), (std::numeric_limits<int>::min)() + 100);
    test<int>((std::numeric_limits<int>::max)() - 100, (std::numeric_limits<int>::max)());
    test<int>(-100, 100);
    test<unsigned>((std::numeric_limits<unsigned>::min)(), (std::numeric_limits<unsigned>::min)() + 100);
    test<unsigned>((std::numeric_limits<unsigned>::max)() - 100, (std::numeric_limits<unsigned>::max)());
    test<long>((std::numeric_limits<long>::min)(), (std::numeric_limits<long>::min)() + 100);
    test<long>((std::numeric_limits<long>::max)() - 100, (std::numeric_limits<long>::max)());
    test<long>(-100, 100);
    test<unsigned long>((std::numeric_limits<unsigned long>::min)(), (std::numeric_limits<unsigned long>::min)() + 100);
    test<unsigned long>((std::numeric_limits<unsigned long>::max)() - 100, (std::numeric_limits<unsigned long>::max)());
    test_fp<float>((std::numeric_limits<float>::min)(), (std::numeric_limits<float>::min)() + 100);
    test_fp<float>((std::numeric_limits<float>::max)() - 100, (std::numeric_limits<float>::max)()); //it's incorrect
    // because floatmax() - 100 causes cancellations error, digits are not affected
    test_fp<float>(-100, 100);
    test<double>((std::numeric_limits<double>::min)(), (std::numeric_limits<double>::min)() + 100);
    test<double>((std::numeric_limits<double>::max)() - 100, (std::numeric_limits<double>::max)()); //it's the same
    test<double>(-100, 100);

    return hpx::util::report_errors();
}
