//  Copyright (c) 2014-2015 Anton Bikineev
//  Copyright (c) 2015 Andreas Schaefer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <vector>

#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/serialization/base_object.hpp>
#include <hpx/runtime/serialization/shared_ptr.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

template <class T>
struct A
{
    int a = 0;

    explicit A(int a):
        a(a)
    {}

    A() = default;

    virtual void foo() const = 0;

};

HPX_TRAITS_NONINTRUSIVE_POLYMORPHIC_TEMPLATE(template<class T>, A<T>)

template <class T>
struct B: A<T>
{
    int b = 0;

    explicit B(int b):
        A<T>(b-1),
        b(b)
    {}

    B() = default;

    virtual void foo() const{}

    HPX_SERIALIZATION_POLYMORPHIC_TEMPLATE_SEMIINTRUSIVE(B);
};

HPX_TRAITS_NONINTRUSIVE_POLYMORPHIC_TEMPLATE(template<class T>, B<T>)
HPX_SERIALIZATION_REGISTER_CLASS_TEMPLATE(template<class T>, B<T>)

namespace hpx { namespace serialization {

    template <class Archive, class T>
    void serialize(Archive& archive, A<T>& s, unsigned)
    {
      archive & s.a;
    }

    template <class Archive, class T>
    void serialize(Archive& archive, B<T>& s, unsigned)
    {
      archive & hpx::serialization::base_object<A<T> >(s);
      archive & s.b;
    }

} }

int main()
{
  std::vector<char> vector;
  {
    hpx::serialization::output_archive archive{vector};
    boost::shared_ptr<A<int> > b = boost::make_shared<B<int> >(-4);
    boost::shared_ptr<A<char> > c = boost::make_shared<B<char> >(44);
    boost::shared_ptr<A<double> > d = boost::make_shared<B<double> >(99);
    archive << b << c << d;
  }

  {
    hpx::serialization::input_archive archive{vector};
    boost::shared_ptr<A<int> > b;
    boost::shared_ptr<A<char> > c;
    boost::shared_ptr<A<double> > d;
    archive >> b >> c >> d;
    HPX_TEST_EQ(b->a, -5);
    HPX_TEST_EQ(boost::static_pointer_cast<B<int> >(b)->b, -4);

    HPX_TEST_EQ(c->a, 43);
    HPX_TEST_EQ(boost::static_pointer_cast<B<char> >(c)->b, 44);

    HPX_TEST_EQ(d->a, 98);
    HPX_TEST_EQ(boost::static_pointer_cast<B<double> >(d)->b, 99);
  }

}
