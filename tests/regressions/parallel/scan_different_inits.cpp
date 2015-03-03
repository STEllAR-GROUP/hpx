//  Copyright (c) 2015 Daniel Bourgeois
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_scan.hpp>
#include <hpx/util/lightweight_test.hpp>

void test_zero()
{
    using namespace hpx::parallel;
    typedef typename std::vector<int>::iterator Iter;
    std::vector<int> a;
    std::vector<int> b, c, d, e;

    Iter i_inc_add =
        inclusive_scan(par, a.begin(), a.end(), b.begin(), 100,
        [](int bar, int baz){ return bar+baz; });
    Iter i_inc_mult =
        inclusive_scan(par, a.begin(), a.end(), c.begin(), 10,
        [](int bar, int baz){ return bar*baz; });
    Iter i_exc_add =
        exclusive_scan(par, a.begin(), a.end(), d.begin(), 100,
        [](int bar, int baz){ return bar+baz; });
    Iter i_exc_mult =
        exclusive_scan(par, a.begin(), a.end(), e.begin(), 10,
        [](int bar, int baz){ return bar*baz; });

    HPX_TEST(i_inc_add == b.begin());
    HPX_TEST(i_inc_mult == c.begin());
    HPX_TEST(i_exc_add == d.begin());
    HPX_TEST(i_exc_mult == e.begin());
}
void test_async_zero()
{
    using namespace hpx::parallel;
    typedef typename std::vector<int>::iterator Iter;
    typedef typename hpx::future<Iter> Fut_Iter;
    std::vector<int> a;
    std::vector<int> b, c, d, e;

    Fut_Iter f_inc_add =
        inclusive_scan(par(task), a.begin(), a.end(), b.begin(), 100,
        [](int bar, int baz){ return bar+baz; });
    Fut_Iter f_inc_mult =
        inclusive_scan(par(task), a.begin(), a.end(), c.begin(), 10,
        [](int bar, int baz){ return bar*baz; });
    Fut_Iter f_exc_add =
        exclusive_scan(par(task), a.begin(), a.end(), d.begin(), 100,
        [](int bar, int baz){ return bar+baz; });
    Fut_Iter f_exc_mult =
        exclusive_scan(par(task), a.begin(), a.end(), e.begin(), 10,
        [](int bar, int baz){ return bar*baz; });

    HPX_TEST(f_inc_add.get() == b.begin());
    HPX_TEST(f_inc_mult.get() == c.begin());
    HPX_TEST(f_exc_add.get() == d.begin());
    HPX_TEST(f_exc_mult.get() == e.begin());
}
void test_one(std::vector<int> a)
{
    using namespace hpx::parallel;
    typedef typename std::vector<int>::iterator Iter;
    std::size_t n = a.size();
    std::vector<int> b(n), c(n), d(n), e(n);
    std::vector<int> b_ans(n), c_ans(n), d_ans(n), e_ans(n);

    inclusive_scan(par, a.begin(), a.end(), b.begin(), 100,
        [](int bar, int baz){ return bar+baz; });
    inclusive_scan(par, a.begin(), a.end(), c.begin(), 10,
        [](int bar, int baz){ return bar*baz; });
    exclusive_scan(par, a.begin(), a.end(), d.begin(), 100,
        [](int bar, int baz){ return bar+baz; });
    exclusive_scan(par, a.begin(), a.end(), e.begin(), 10,
        [](int bar, int baz){ return bar*baz; });

    detail::sequential_inclusive_scan(a.begin(), a.end(), b_ans.begin(), 100,
        [](int bar, int baz){ return bar+baz; });
    detail::sequential_inclusive_scan(a.begin(), a.end(), c_ans.begin(), 10,
        [](int bar, int baz){ return bar*baz; });
    detail::sequential_exclusive_scan(a.begin(), a.end(), d_ans.begin(), 100,
        [](int bar, int baz){ return bar+baz; });
    detail::sequential_exclusive_scan(a.begin(), a.end(), e_ans.begin(), 10,
        [](int bar, int baz){ return bar*baz; });

    HPX_TEST(std::equal(b.begin(), b.end(), b_ans.begin()));
    HPX_TEST(std::equal(c.begin(), c.end(), c_ans.begin()));
    HPX_TEST(std::equal(d.begin(), d.end(), d_ans.begin()));
    HPX_TEST(std::equal(e.begin(), e.end(), e_ans.begin()));
}

void test_async_one(std::vector<int> a)
{
    using namespace hpx::parallel;
    typedef typename std::vector<int>::iterator Iter;
    typedef typename hpx::future<Iter> Fut_Iter;
    std::size_t n = a.size();
    std::vector<int> b(n), c(n), d(n), e(n);
    std::vector<int> b_ans(n), c_ans(n), d_ans(n), e_ans(n);

    Fut_Iter f_inc_add =
        inclusive_scan(par(task), a.begin(), a.end(), b.begin(), 100,
        [](int bar, int baz){ return bar+baz; });
    Fut_Iter f_inc_mult =
        inclusive_scan(par(task), a.begin(), a.end(), c.begin(), 10,
        [](int bar, int baz){ return bar*baz; });
    Fut_Iter f_exc_add =
        exclusive_scan(par(task), a.begin(), a.end(), d.begin(), 100,
        [](int bar, int baz){ return bar+baz; });
    Fut_Iter f_exc_mult =
        exclusive_scan(par(task), a.begin(), a.end(), e.begin(), 10,
        [](int bar, int baz){ return bar*baz; });

    detail::sequential_inclusive_scan(a.begin(), a.end(), b_ans.begin(), 100,
        [](int bar, int baz){ return bar+baz; });
    detail::sequential_inclusive_scan(a.begin(), a.end(), c_ans.begin(), 10,
        [](int bar, int baz){ return bar*baz; });
    detail::sequential_exclusive_scan(a.begin(), a.end(), d_ans.begin(), 100,
        [](int bar, int baz){ return bar+baz; });
    detail::sequential_exclusive_scan(a.begin(), a.end(), e_ans.begin(), 10,
        [](int bar, int baz){ return bar*baz; });

    f_inc_add.wait();
    f_inc_mult.wait();
    f_exc_add.wait();
    f_exc_mult.wait();

    HPX_TEST(std::equal(b.begin(), b.end(), b_ans.begin()));
    HPX_TEST(std::equal(c.begin(), c.end(), c_ans.begin()));
    HPX_TEST(std::equal(d.begin(), d.end(), d_ans.begin()));
    HPX_TEST(std::equal(e.begin(), e.end(), e_ans.begin()));
}

int hpx_main(boost::program_options::variables_map& vm)
{
    std::vector<int> a1{1,1,2,2,3,3,4,4};
    test_one(a1);
    test_async_one(a1);

    std::vector<int> a2{1,1};
    test_one(a2);
    test_async_one(a2);

    std::vector<int> a3{1};
    test_one(a3);
    test_async_one(a3);

    test_zero();
    test_async_zero();

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    std::vector<std::string> cfg;
    cfg.push_back("hpx.os_threads=" + boost::lexical_cast<std::string>
                  (hpx::threads::hardware_concurrency()));

    HPX_TEST_EQ_MSG(hpx::init(argc, argv, cfg), 0,
        "HPX main exted with non-zero status");

    return hpx::util::report_errors();
}
