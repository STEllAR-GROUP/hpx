//  Copyright (c) 2015 Daniel Bourgeois
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_is_partitioned.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/range/functions.hpp>

#include "test_utils.hpp"

#include <iostream>

////////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_partitioned1(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    //fill first half of array with even numbers and second half
    //with odd numbers
    std::fill(boost::begin(c), boost::begin(c) + c.size()/2,
        2*(std::rand() % 100));
    std::fill(boost::begin(c) + c.size()/2, boost::end(c),
        2*(std::rand() % 100) + 1);

    bool parted = hpx::parallel::is_partitioned(policy,
        iterator(boost::begin(c)), iterator(boost::end(c)),
        [](std::size_t n){ return n % 2 == 0; });

    HPX_TEST(parted);
}

template <typename ExPolicy, typename IteratorTag>
void test_partitioned1_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    //fill first half of array with even numbers and second half
    //with odd numbers
    std::fill(boost::begin(c), boost::begin(c) + c.size()/2,
        2*(std::rand() % 100));
    std::fill(boost::begin(c) + c.size()/2, boost::end(c),
        2*(std::rand() % 100) + 1);

    hpx::future<bool> f =
        hpx::parallel::is_partitioned(p,
        iterator(boost::begin(c)), iterator(boost::end(c)),
        [](std::size_t n){ return n % 2 == 0; });
    f.wait();

    HPX_TEST(f.get());
}

template <typename IteratorTag>
void test_partitioned1()
{
    using namespace hpx::parallel;
    test_partitioned1(seq, IteratorTag());
    test_partitioned1(par, IteratorTag());
    test_partitioned1(par_vec, IteratorTag());

    test_partitioned1_async(seq(task), IteratorTag());
    test_partitioned1_async(par(task), IteratorTag());


    test_partitioned1(execution_policy(seq), IteratorTag());
    test_partitioned1(execution_policy(par), IteratorTag());
    test_partitioned1(execution_policy(par_vec), IteratorTag());
    test_partitioned1(execution_policy(seq(task)), IteratorTag());
    test_partitioned1(execution_policy(par(task)), IteratorTag());
}

void partitioned_test1()
{
    test_partitioned1<std::random_access_iterator_tag>();
    test_partitioned1<std::forward_iterator_tag>();
    test_partitioned1<std::input_iterator_tag>();
}

////////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_partitioned2(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c_odd(10007);
    //fill all of array with odds
    std::fill(boost::begin(c_odd), boost::end(c_odd),
        2*(std::rand() % 100) + 1);
    std::vector<std::size_t> c_even(10007);
    //fill all of array with evens
    std::fill(boost::begin(c_odd), boost::end(c_odd),
        2*(std::rand() % 100));

    bool parted_odd = hpx::parallel::is_partitioned(policy,
        iterator(boost::begin(c_odd)), iterator(boost::end(c_odd)),
        [](std::size_t n){ return n % 2 == 0; });
    bool parted_even = hpx::parallel::is_partitioned(policy,
        iterator(boost::begin(c_even)), iterator(boost::end(c_even)),
        [](std::size_t n){ return n % 2 == 0; });

    HPX_TEST(parted_odd);
    HPX_TEST(parted_even);
}

template <typename ExPolicy, typename IteratorTag>
void test_partitioned2_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c_odd(10007);
    //fill all of array with odds
    std::fill(boost::begin(c_odd), boost::end(c_odd),
        2*(std::rand() % 100) + 1);
    std::vector<std::size_t> c_even(10007);
    //fill all of array with evens
    std::fill(boost::begin(c_odd), boost::end(c_odd),
        2*(std::rand() % 100));

    hpx::future<bool> f_odd = hpx::parallel::is_partitioned(p,
        iterator(boost::begin(c_odd)), iterator(boost::end(c_odd)),
        [](std::size_t n){ return n % 2 == 0; });
    hpx::future<bool> f_even = hpx::parallel::is_partitioned(p,
        iterator(boost::begin(c_even)), iterator(boost::end(c_even)),
        [](std::size_t n){ return n % 2 == 0; });

    f_odd.wait();
    HPX_TEST(f_odd.get());
    f_even.wait();
    HPX_TEST(f_even.get());
}

template <typename IteratorTag>
void test_partitioned2()
{
    using namespace hpx::parallel;
    test_partitioned2(seq, IteratorTag());
    test_partitioned2(par, IteratorTag());
    test_partitioned2(par_vec, IteratorTag());

    test_partitioned2_async(seq(task), IteratorTag());
    test_partitioned2_async(par(task), IteratorTag());


    test_partitioned2(execution_policy(seq), IteratorTag());
    test_partitioned2(execution_policy(par), IteratorTag());
    test_partitioned2(execution_policy(par_vec), IteratorTag());
    test_partitioned2(execution_policy(seq(task)), IteratorTag());
    test_partitioned2(execution_policy(par(task)), IteratorTag());
}

void partitioned_test2()
{
    test_partitioned2<std::random_access_iterator_tag>();
    test_partitioned2<std::forward_iterator_tag>();
    test_partitioned2<std::input_iterator_tag>();
}

////////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_partitioned3(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c_beg(10007);
    //fill first half of array with even numbers and second half
    //with odd numbers
    std::fill(boost::begin(c_beg), boost::begin(c_beg) + c_beg.size()/2,
        2*(std::rand() % 100));
    std::fill(boost::begin(c_beg) + c_beg.size()/2, boost::end(c_beg),
        2*(std::rand() % 100) + 1);
    std::vector<size_t> c_end = c_beg;
    //add odd number to the beginning
    c_beg[0] -= 1;
    //add even number to end
    c_end[c_end.size()-1] -= 1;

    bool parted1 = hpx::parallel::is_partitioned(policy,
        iterator(boost::begin(c_beg)), iterator(boost::end(c_beg)),
        [](std::size_t n){ return n % 2 == 0; });
    bool parted2 = hpx::parallel::is_partitioned(policy,
        iterator(boost::begin(c_end)), iterator(boost::end(c_end)),
        [](std::size_t n){ return n % 2 == 0; });

    HPX_TEST(!parted1);
    HPX_TEST(!parted2);
}

template <typename ExPolicy, typename IteratorTag>
void test_partitioned3_async(ExPolicy p, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c_beg(10007);
    //fill first half of array with even numbers and second half
    //with odd numbers
    std::fill(boost::begin(c_beg), boost::begin(c_beg) + c_beg.size()/2,
        2*(std::rand() % 100));
    std::fill(boost::begin(c_beg) + c_beg.size()/2, boost::end(c_beg),
        2*(std::rand() % 100) + 1);
    std::vector<size_t> c_end = c_beg;
    //add odd number to the beginning
    c_beg[0] -= 1;
    //add even number to end
    c_end[c_end.size()-1] -= 1;

    hpx::future<bool> f_beg = hpx::parallel::is_partitioned(p,
        iterator(boost::begin(c_beg)), iterator(boost::end(c_beg)),
        [](std::size_t n){ return n % 2 == 0; });
    hpx::future<bool> f_end = hpx::parallel::is_partitioned(p,
        iterator(boost::begin(c_end)), iterator(boost::end(c_end)),
        [](std::size_t n){ return n % 2 == 0; });

    f_beg.wait();
    HPX_TEST(!f_beg.get());
    f_end.wait();
    HPX_TEST(!f_end.get());
}

template <typename IteratorTag>
void test_partitioned3()
{
    using namespace hpx::parallel;
    test_partitioned3(seq, IteratorTag());
    test_partitioned3(par, IteratorTag());
    test_partitioned3(par_vec, IteratorTag());

    test_partitioned3_async(seq(task), IteratorTag());
    test_partitioned3_async(par(task), IteratorTag());


    test_partitioned3(execution_policy(seq), IteratorTag());
    test_partitioned3(execution_policy(par), IteratorTag());
    test_partitioned3(execution_policy(par_vec), IteratorTag());
    test_partitioned3(execution_policy(seq(task)), IteratorTag());
    test_partitioned3(execution_policy(par(task)), IteratorTag());
}

void partitioned_test3()
{
    test_partitioned3<std::random_access_iterator_tag>();
    test_partitioned3<std::forward_iterator_tag>();
    test_partitioned3<std::input_iterator_tag>();
}

////////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_partitioned_exception(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    //fill first half of array with even numbers and second half
    //with odd numbers
    std::fill(boost::begin(c), boost::begin(c) + c.size()/2,
        2*(std::rand() % 100));
    std::fill(boost::begin(c) + c.size()/2, boost::end(c),
        2*(std::rand() % 100) + 1);

    bool caught_exception = false;
    try{
        hpx::parallel::is_partitioned(policy,
            decorated_iterator(
                boost::begin(c),
                [](){ throw std::runtime_error("test"); }),
            decorated_iterator(
                boost::end(c),
                [](){ throw std::runtime_error("test"); }),
            [](std::size_t n){ return n % 2 == 0; });
    }
    catch(hpx::exception_list const& e) {
        caught_exception = true;
        test::test_num_exceptions<ExPolicy, IteratorTag>::call(policy, e);
    }
    catch(...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
}

template <typename ExPolicy, typename IteratorTag>
void test_partitioned_async_exception(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    //fill first half of array with even numbers and second half
    //with odd numbers
    std::fill(boost::begin(c), boost::begin(c) + c.size()/2,
        2*(std::rand() % 100));
    std::fill(boost::begin(c) + c.size()/2, boost::end(c),
        2*(std::rand() % 100) + 1);

    bool caught_exception = false;
    try {
        hpx::future<bool> f =
            hpx::parallel::is_partitioned(p,
                decorated_iterator(
                    boost::begin(c),
                    [](){ throw std::runtime_error("test"); }),
                decorated_iterator(
                    boost::end(c),
                    [](){ throw std::runtime_error("test"); }),
                [](std::size_t n){ return n % 2 == 0; });
        f.get();

        HPX_TEST(false);
    }
    catch(hpx::exception_list const& e) {
        caught_exception = true;
        test::test_num_exceptions<
            ExPolicy, IteratorTag
            >::call(p, e);
    }
    catch(...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
}

template <typename IteratorTag>
void test_partitioned_exception()
{
    using namespace hpx::parallel;
    //If the execution policy object is of type vector_execution_policy,
    //  std::terminate shall be called. Therefore we do not test exceptions
    //  with a vector execution policy
    test_partitioned_exception(seq, IteratorTag());
    test_partitioned_exception(par, IteratorTag());

    test_partitioned_async_exception(seq(task), IteratorTag());
    test_partitioned_async_exception(par(task), IteratorTag());

    test_partitioned_exception(execution_policy(par), IteratorTag());
    test_partitioned_exception(execution_policy(seq(task)), IteratorTag());
    test_partitioned_exception(execution_policy(par(task)), IteratorTag());
}
void partitioned_exception_test()
{
    test_partitioned_exception<std::random_access_iterator_tag>();
    test_partitioned_exception<std::forward_iterator_tag>();
    test_partitioned_exception<std::input_iterator_tag>();
}

////////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_partitioned_bad_alloc(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    //fill first half of array with even numbers and second half
    //with odd numbers
    std::fill(boost::begin(c), boost::begin(c) + c.size()/2,
        2*(std::rand() % 100));
    std::fill(boost::begin(c) + c.size()/2, boost::end(c),
        2*(std::rand() % 100) + 1);

    bool caught_bad_alloc = false;
    try {
        hpx::parallel::is_partitioned(policy,
            decorated_iterator(
                boost::begin(c),
                [](){ throw std::bad_alloc(); }),
            decorated_iterator(
                boost::end(c),
                [](){ throw std::bad_alloc(); }),
            [](std::size_t n){ return n % 2 == 0; });
        HPX_TEST(false);
    }
    catch(std::bad_alloc const&) {
        caught_bad_alloc = true;
    }
    catch(...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_bad_alloc);
}

template <typename ExPolicy, typename IteratorTag>
void test_partitioned_async_bad_alloc(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    //fill first half of array with even numbers and second half
    //with odd numbers
    std::fill(boost::begin(c), boost::begin(c) + c.size()/2,
        2*(std::rand() % 100));
    std::fill(boost::begin(c) + c.size()/2, boost::end(c),
        2*(std::rand() % 100) + 1);

    bool caught_bad_alloc = false;
    try {
        hpx::future<bool> f =
            hpx::parallel::is_partitioned(p,
                decorated_iterator(
                    boost::begin(c),
                    [](){ throw std::bad_alloc(); }),
                decorated_iterator(
                    boost::end(c),
                    [](){ throw std::bad_alloc(); }),
                [](std::size_t n){ return n % 2 == 0; });

        f.get();
        HPX_TEST(false);
    }
    catch(std::bad_alloc const&) {
        caught_bad_alloc = true;
    }
    catch(...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_bad_alloc);
}

template <typename IteratorTag>
void test_partitioned_bad_alloc()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_partitioned_bad_alloc(par, IteratorTag());
    test_partitioned_bad_alloc(seq, IteratorTag());

    test_partitioned_async_bad_alloc(seq(task), IteratorTag());
    test_partitioned_async_bad_alloc(par(task), IteratorTag());

    test_partitioned_bad_alloc(execution_policy(par), IteratorTag());
    test_partitioned_bad_alloc(execution_policy(seq), IteratorTag());
    test_partitioned_bad_alloc(execution_policy(seq(task)), IteratorTag());
    test_partitioned_bad_alloc(execution_policy(par(task)), IteratorTag());
}

void partitioned_bad_alloc_test()
{
    test_partitioned_bad_alloc<std::random_access_iterator_tag>();
    test_partitioned_bad_alloc<std::forward_iterator_tag>();
    test_partitioned_bad_alloc<std::input_iterator_tag>();
}


int hpx_main(boost::program_options::variables_map& vm)
{
    partitioned_test1();
    partitioned_test2();
    partitioned_test3();
    partitioned_exception_test();
    partitioned_bad_alloc_test();

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    using namespace boost::program_options;
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    std::vector<std::string> cfg;
    cfg.push_back("hpx.os_threads=" +
        boost::lexical_cast<std::string>(hpx::threads::hardware_concurrency()));

    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
