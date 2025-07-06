//  Copyright (c) 2014-2015 Anton Bikineev
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/modules/testing.hpp>
#include <hpx/serialization/base_object.hpp>
#include <hpx/serialization/serialize.hpp>
#include <hpx/serialization/shared_ptr.hpp>

#include <memory>
#include <vector>

template <class T>
struct A
{
    int a;

    explicit A(int a)
      : a(a)
    {
    }
    A() = default;

    virtual ~A() {}

    virtual void foo() const = 0;

    template <typename Ar>
    void serialize(Ar& ar, unsigned)
    {
        ar & a;
    }
    HPX_SERIALIZATION_POLYMORPHIC_ABSTRACT(A);
};

template <class T>
struct B : A<T>
{
    int b;

    explicit B(int b)
      : A<T>(b - 1)
      , b(b)
    {
    }
    B() = default;

    void foo() const override {}

    template <typename Ar>
    void serialize(Ar& ar, unsigned)
    {
        ar& hpx::serialization::base_object<A<T>>(*this);
        ar & b;
    }
    HPX_SERIALIZATION_POLYMORPHIC_TEMPLATE(B, override);
};

int main()
{
    std::vector<char> vector;
    {
        hpx::serialization::output_archive archive{vector};
        std::shared_ptr<A<int>> b = std::make_shared<B<int>>(2);
        std::shared_ptr<A<char>> c = std::make_shared<B<char>>(8);
        archive << b << c;
    }

    {
        hpx::serialization::input_archive archive{vector};
        std::shared_ptr<A<int>> b;
        std::shared_ptr<A<char>> c;
        archive >> b >> c;

        HPX_TEST_EQ(std::static_pointer_cast<B<int>>(b)->a, 1);
        HPX_TEST_EQ(std::static_pointer_cast<B<int>>(b)->b, 2);

        HPX_TEST_EQ(std::static_pointer_cast<B<char>>(c)->a, 7);
        HPX_TEST_EQ(std::static_pointer_cast<B<char>>(c)->b, 8);
    }

    return hpx::util::report_errors();
}
