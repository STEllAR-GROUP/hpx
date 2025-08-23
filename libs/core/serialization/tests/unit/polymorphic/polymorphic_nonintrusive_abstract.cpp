//  Copyright (c) 2015 Thomas Heller
//  Copyright (c) 2015 Anton Bikineev
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/serialization/base_object.hpp>
#include <hpx/serialization/input_archive.hpp>
#include <hpx/serialization/output_archive.hpp>
#include <hpx/serialization/serialize.hpp>

#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <string>
#include <vector>

template <typename T>
struct Base
{
    Base() = default;

    explicit Base(std::string const& prefix)
      : prefix_(prefix)
    {
    }

    virtual ~Base() {}

    virtual std::size_t size() = 0;
    virtual std::string print() = 0;

    std::string prefix_;
};

HPX_TRAITS_NONINTRUSIVE_POLYMORPHIC_TEMPLATE((template <class T>), Base<T>)

template <typename Archive, typename T>
void serialize(Archive& ar, Base<T>& b, unsigned)
{
    ar & b.prefix_;
}

template <typename T>
struct Derived1 : Base<T>
{
    Derived1()
      : size_(0)
    {
    }

    Derived1(std::string const& prefix, std::size_t size)
      : Base<T>(prefix)
      , size_(size)
    {
    }

    std::size_t size() override
    {
        return size_;
    }

    std::size_t size_;
};

HPX_TRAITS_NONINTRUSIVE_POLYMORPHIC_TEMPLATE((template <class T>), Derived1<T>)

template <typename Archive, typename T>
void serialize(Archive& ar, Derived1<T>& d1, unsigned)
{
    ar& hpx::serialization::base_object<Base<T>>(d1);
    ar & d1.size_;
}

struct Derived2 : Derived1<double>
{
    Derived2() {}

    Derived2(
        std::string const& message, std::string const& prefix, std::size_t size)
      : Derived1<double>(prefix, size)
      , message_(message)
    {
    }

    std::string print() override
    {
        return message_;
    }

    std::string message_;
};

HPX_SERIALIZATION_REGISTER_CLASS(Derived2)

template <typename Archive>
void serialize(Archive& ar, Derived2& d2, unsigned)
{
    ar& hpx::serialization::base_object<Derived1<double>>(d2);
    ar & d2.message_;
}

int main()
{
    std::vector<char> buffer;
    Derived2 d("/tmp", "fancy", 10);
    Base<double>& b = d;
    Base<double>* b_ptr = &d;
    HPX_TEST_EQ(d.print(), b.print());
    HPX_TEST_EQ(d.size(), b.size());
    HPX_TEST_EQ(d.prefix_, b.prefix_);
    HPX_TEST_EQ(d.print(), b_ptr->print());
    HPX_TEST_EQ(d.size(), b_ptr->size());
    HPX_TEST_EQ(d.prefix_, b_ptr->prefix_);

    {
        hpx::serialization::output_archive oarchive(buffer);
        oarchive << b;
        oarchive << hpx::serialization::detail::raw_ptr(b_ptr);
    }

    {
        Derived2 d1;
        Derived2 d2;
        Base<double>& b1 = d1;
        Base<double>* b2 = nullptr;

        hpx::serialization::input_archive iarchive(buffer);
        iarchive >> b1;
        iarchive >> hpx::serialization::detail::raw_ptr(b2);

        HPX_TEST(b2 != nullptr);

        HPX_TEST_EQ(d.print(), b1.print());
        HPX_TEST_EQ(d.size(), b1.size());
        HPX_TEST_EQ(d.prefix_, b1.prefix_);
        HPX_TEST_EQ(d.print(), b2->print());
        HPX_TEST_EQ(d.size(), b2->size());
        HPX_TEST_EQ(d.prefix_, b2->prefix_);

        delete b2;
    }
    return hpx::util::report_errors();
}
