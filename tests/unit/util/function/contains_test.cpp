//  Taken from the Boost.Function library

//  Copyright Douglas Gregor 2001-2003.
//  Copyright 2013 Hartmut Kaiser
//
//  Use, modification and
//  distribution is subject to the Boost Software License, Version
//  1.0. (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/util.hpp>
#include <hpx/util/lightweight_test.hpp>

static int forty_two() { return 42; }

struct Seventeen
{
    int operator()() const { return 17; }
};

//struct ReturnInt
//{
//    explicit ReturnInt(int value) : value(value) {}
//
//    int operator()() const { return value; }
//
//    int value;
//};
//
//bool operator==(const ReturnInt& x, const ReturnInt& y)
//{ return x.value == y.value; }
//
//bool operator!=(const ReturnInt& x, const ReturnInt& y)
//{ return x.value != y.value; }
//
//namespace contain_test {
//
//    struct ReturnIntFE
//    {
//        explicit ReturnIntFE(int value) : value(value) {}
//
//        int operator()() const { return value; }
//
//        int value;
//    };
//}
//
//#ifndef BOOST_NO_ARGUMENT_DEPENDENT_LOOKUP
//
//namespace contain_test {
//# ifndef BOOST_NO_FUNCTION_TEMPLATE_ORDERING
//    bool function_equal(const ReturnIntFE& x, const ReturnIntFE& y)
//    { return x.value == y.value; }
//# else
//    bool function_equal_impl(const ReturnIntFE& x, const ReturnIntFE& y, int)
//    { return x.value == y.value; }
//# endif // #ifndef BOOST_NO_FUNCTION_TEMPLATE_ORDERING
//}
//#else // BOOST_NO_ARGUMENT_DEPENDENT_LOOKUP
//namespace boost {
//# ifndef BOOST_NO_FUNCTION_TEMPLATE_ORDERING
//    bool
//        function_equal(const contain_test::ReturnIntFE& x,
//        const contain_test::ReturnIntFE& y)
//    { return x.value == y.value; }
//# else
//    bool
//        function_equal_impl(const contain_test::ReturnIntFE& x,
//        const contain_test::ReturnIntFE& y, int)
//    { return x.value == y.value; }
//# endif
//}
//#endif

static void target_test()
{
    hpx::util::function_nonser<int()> f;

    f = &forty_two;
    HPX_TEST(*f.target<int (*)()>() == &forty_two);
    HPX_TEST(!f.target<Seventeen>());

    f = Seventeen();
    HPX_TEST(!f.target<int (*)()>());
    HPX_TEST(f.target<Seventeen>());

    Seventeen this_seventeen;
    f = this_seventeen;
    HPX_TEST(!f.target<int (*)()>());
    HPX_TEST(f.target<Seventeen>());
}

//static void equal_test()
//{
//    hpx::util::function_nonser<int()> f;
//
//    f = &forty_two;
//    HPX_TEST(f == &forty_two);
//    HPX_TEST(f != ReturnInt(17));
//#if !(defined(__GNUC__) && __GNUC__ == 3 && __GNUC_MINOR__ <= 3)
//    HPX_TEST(&forty_two == f);
//    HPX_TEST(ReturnInt(17) != f);
//#endif
//
//    HPX_TEST(f.contains(&forty_two));
//
//    f = ReturnInt(17);
//    HPX_TEST(f != &forty_two);
//    HPX_TEST(f == ReturnInt(17));
//    HPX_TEST(f != ReturnInt(16));
//#if !(defined(__GNUC__) && __GNUC__ == 3 && __GNUC_MINOR__ <= 3)
//    HPX_TEST(&forty_two != f);
//    HPX_TEST(ReturnInt(17) == f);
//    HPX_TEST(ReturnInt(16) != f);
//#endif
//
//    HPX_TEST(f.contains(ReturnInt(17)));
//
//    f = contain_test::ReturnIntFE(17);
//    HPX_TEST(f != &forty_two);
//    HPX_TEST(f == contain_test::ReturnIntFE(17));
//    HPX_TEST(f != contain_test::ReturnIntFE(16));
//#if !(defined(__GNUC__) && __GNUC__ == 3 && __GNUC_MINOR__ <= 3)
//    HPX_TEST(&forty_two != f);
//    HPX_TEST(contain_test::ReturnIntFE(17) == f);
//    HPX_TEST(contain_test::ReturnIntFE(16) != f);
//#endif
//
//    HPX_TEST(f.contains(contain_test::ReturnIntFE(17)));
//
//#if !defined(BOOST_FUNCTION_NO_FUNCTION_TYPE_SYNTAX)
//    hpx::util::function_nonser<int(void)> g;
//
//    g = &forty_two;
//    HPX_TEST(g == &forty_two);
//    HPX_TEST(g != ReturnInt(17));
//#  if !(defined(__GNUC__) && __GNUC__ == 3 && __GNUC_MINOR__ <= 3)
//    HPX_TEST(&forty_two == g);
//    HPX_TEST(ReturnInt(17) != g);
//#  endif
//
//    g = ReturnInt(17);
//    HPX_TEST(g != &forty_two);
//    HPX_TEST(g == ReturnInt(17));
//    HPX_TEST(g != ReturnInt(16));
//#  if !(defined(__GNUC__) && __GNUC__ == 3 && __GNUC_MINOR__ <= 3)
//    HPX_TEST(&forty_two != g);
//    HPX_TEST(ReturnInt(17) == g);
//    HPX_TEST(ReturnInt(16) != g);
//#  endif
//#endif
//}
//
//static void ref_equal_test()
//{
//    {
//        ReturnInt ri(17);
//        hpx::util::function_nonser0<int> f = boost::ref(ri);
//
//        // References and values are equal
//        HPX_TEST(f == boost::ref(ri));
//        HPX_TEST(f == ri);
//        HPX_TEST(boost::ref(ri) == f);
//        HPX_TEST(!(f != boost::ref(ri)));
//        HPX_TEST(!(f != ri));
//        HPX_TEST(!(boost::ref(ri) != f));
//#if !(defined(__GNUC__) && __GNUC__ == 3 && __GNUC_MINOR__ <= 3)
//        HPX_TEST(ri == f);
//        HPX_TEST(!(ri != f));
//#endif
//
//        // Values equal, references inequal
//        ReturnInt ri2(17);
//        HPX_TEST(f == ri2);
//        HPX_TEST(f != boost::ref(ri2));
//        HPX_TEST(boost::ref(ri2) != f);
//        HPX_TEST(!(f != ri2));
//        HPX_TEST(!(f == boost::ref(ri2)));
//        HPX_TEST(!(boost::ref(ri2) == f));
//#if !(defined(__GNUC__) && __GNUC__ == 3 && __GNUC_MINOR__ <= 3)
//        HPX_TEST(ri2 == f);
//        HPX_TEST(!(ri2 != f));
//#endif
//    }
//
//#if !defined(BOOST_FUNCTION_NO_FUNCTION_TYPE_SYNTAX)
//    {
//        ReturnInt ri(17);
//        hpx::util::function_nonser<int(void)> f = boost::ref(ri);
//
//        // References and values are equal
//        HPX_TEST(f == boost::ref(ri));
//        HPX_TEST(f == ri);
//        HPX_TEST(boost::ref(ri) == f);
//        HPX_TEST(!(f != boost::ref(ri)));
//        HPX_TEST(!(f != ri));
//        HPX_TEST(!(boost::ref(ri) != f));
//#  if !(defined(__GNUC__) && __GNUC__ == 3 && __GNUC_MINOR__ <= 3)
//        HPX_TEST(ri == f);
//        HPX_TEST(!(ri != f));
//#  endif
//
//        // Values equal, references inequal
//        ReturnInt ri2(17);
//        HPX_TEST(f == ri2);
//        HPX_TEST(f != boost::ref(ri2));
//        HPX_TEST(boost::ref(ri2) != f);
//        HPX_TEST(!(f != ri2));
//        HPX_TEST(!(f == boost::ref(ri2)));
//        HPX_TEST(!(boost::ref(ri2) == f));
//#  if !(defined(__GNUC__) && __GNUC__ == 3 && __GNUC_MINOR__ <= 3)
//        HPX_TEST(ri2 == f);
//        HPX_TEST(!(ri2 != f));
//#  endif
//    }
//#endif
//}

int main(int, char*[])
{
    target_test();
//    equal_test();
//    ref_equal_test();

    return hpx::util::report_errors();
}
