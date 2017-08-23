//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/serialization/list.hpp>

#include <hpx/runtime/serialization/input_archive.hpp>
#include <hpx/runtime/serialization/output_archive.hpp>

#include <hpx/util/lightweight_test.hpp>

#include <cstddef>
#include <list>
#include <vector>

template <typename T>
struct A
{
    A() {}

    A(T t) : t_(t) {}
    T t_;

    A & operator=(T t) { t_ = t; return *this; }

    template <typename Archive>
    void serialize(Archive & ar, unsigned)
    {
        ar & t_;
    }
};

// non-default constructible
struct B
{
    const int a;
    double b;

public:
    B() = delete;
    B(int a): a(a) {}

    template <class Archive>
    void serialize(Archive& ar, unsigned)
    {
        ar & b;
    }

    int get_a() const
    {
        return a;
    }

    void set_b(short b)
    {
        this->b = b;
    }

    short get_b() const
    {
        return b;
    }
};

template <class Archive>
void save_construct_data(Archive& ar, const B* b, unsigned)
{
    ar << b->get_a();
}

template <class Archive>
void load_construct_data(Archive& ar, B* b, unsigned)
{
    int a = 0;
    ar >> a;
    ::new (b) B(a);
}

void test_bool()
{
    {
        std::vector<char> buffer;
        hpx::serialization::output_archive oarchive(buffer);

        std::list<bool> os;
        os.push_back(true);
        os.push_back(false);
        os.push_back(false);
        os.push_back(true);
        oarchive << os;

        hpx::serialization::input_archive iarchive(buffer);
        std::list<bool> is;
        iarchive >> is;
        HPX_TEST_EQ(os.size(), is.size());
        for (auto ot = os.begin(), it = is.begin();
             ot != os.end(); ++ot, ++it)
        {
            HPX_TEST_EQ(*ot, *it);
        }
    }
    {
        std::vector<char> buffer;
        hpx::serialization::output_archive oarchive(buffer);

        std::list<A<bool> > os;
        os.push_back(true);
        os.push_back(false);
        os.push_back(false);
        os.push_back(true);
        oarchive << os;

        hpx::serialization::input_archive iarchive(buffer);
        std::list<A<bool> > is;
        iarchive >> is;
        HPX_TEST_EQ(os.size(), is.size());
        for (auto ot = os.begin(), it = is.begin();
             ot != os.end(); ++ot, ++it)
        {
            HPX_TEST_EQ(ot->t_, it->t_);
        }
    }
}

template <typename T>
void test(T min, T max)
{
    {
        std::vector<char> buffer;
        hpx::serialization::output_archive oarchive(buffer);
        std::list<T> os;
        for (T c = min; c < max; ++c)
        {
            os.push_back(c);
        }
        oarchive << os;
        hpx::serialization::input_archive iarchive(buffer);
        std::list<T> is;
        iarchive >> is;
        HPX_TEST_EQ(os.size(), is.size());
        for (auto ot = os.begin(), it = is.begin();
             ot != os.end(); ++ot, ++it)
        {
            HPX_TEST_EQ(*ot, *it);
        }
    }
    {
        std::vector<char> buffer;
        hpx::serialization::output_archive oarchive(buffer);
        std::list<A<T> > os;
        for (T c = min; c < max; ++c)
        {
            os.push_back(c);
        }
        oarchive << os;
        hpx::serialization::input_archive iarchive(buffer);
        std::list<A<T> > is;
        iarchive >> is;
        HPX_TEST_EQ(os.size(), is.size());
        for (auto ot = os.begin(), it = is.begin();
             ot != os.end(); ++ot, ++it)
        {
            HPX_TEST_EQ(ot->t_, it->t_);
        }
    }
}

template <typename T>
void test_fp(T min, T max)
{
    {
        std::vector<char> buffer;
        hpx::serialization::output_archive oarchive(buffer);
        std::list<T> os;
        for (T c = min; c < max; c += static_cast<T>(0.5))
        {
            os.push_back(c);
        }
        oarchive << os;
        hpx::serialization::input_archive iarchive(buffer);
        std::list<T> is;
        iarchive >> is;
        HPX_TEST_EQ(os.size(), is.size());
        for (auto ot = os.begin(), it = is.begin();
             ot != os.end(); ++ot, ++it)
        {
            HPX_TEST_EQ(*ot++, *it++);
        }
    }
    {
        std::vector<char> buffer;
        hpx::serialization::output_archive oarchive(buffer);
        std::list<A<T> > os;
        for (T c = min; c < max; c += static_cast<T>(0.5))
        {
            os.push_back(c);
        }
        oarchive << os;
        hpx::serialization::input_archive iarchive(buffer);
        std::list<A<T> > is;
        iarchive >> is;
        HPX_TEST_EQ(os.size(), is.size());
        for (auto ot = os.begin(), it = is.begin();
             ot != os.end(); ++ot, ++it)
        {
            HPX_TEST_EQ(ot->t_, it->t_);
        }
    }
}

void test_non_default_constructible()
{
    std::vector<char> buffer;
    hpx::serialization::output_archive oarchive(buffer);

    std::list<B> os;
    os.push_back(1);
    os.push_back(2);
    os.push_back(3);
    os.push_back(4);

    short b = 1;
    for (auto& i: os) {
        i.set_b(b);
        ++b;
    }

    oarchive << os;

    hpx::serialization::input_archive iarchive(buffer);
    std::list<B> is;
    iarchive >> is;
    HPX_TEST_EQ(os.size(), is.size());
    for (auto ot = os.begin(), it = is.begin();
         ot != os.end(); ++ot, ++it)
    {
        HPX_TEST_EQ(ot->get_a(), it->get_a());
        HPX_TEST_EQ(ot->get_b(), it->get_b());
    }
}

int main()
{
    test_bool();
    test<char>((std::numeric_limits<char>::min)(),
        (std::numeric_limits<char>::max)());
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
        (std::numeric_limits<float>::max)()); //it's incorrect
    // because floatmax() - 100 causes cancellations error,
    // digits are not affected
    test_fp<float>(-100, 100);
    test<double>((std::numeric_limits<double>::min)(),
        (std::numeric_limits<double>::min)() + 100);
    test<double>((std::numeric_limits<double>::max)() - 100,
        (std::numeric_limits<double>::max)()); //it's the same
    test<double>(-100, 100);

    test_non_default_constructible();

    return hpx::util::report_errors();
}
