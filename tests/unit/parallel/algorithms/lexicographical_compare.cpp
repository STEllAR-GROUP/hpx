//  Copyright (c) 2014 Grant Mercer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_lexicographical_compare.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/range/functions.hpp>

#include "test_utils.hpp"

////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_lexicographical_compare1(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), 0);

    //d is lexicographical less than c
    std::vector<std::size_t> d(10006);
    std::iota(boost::begin(d), boost::end(d), 0);

    bool res = hpx::parallel::lexicographical_compare(policy,
        iterator(boost::begin(c)), iterator(boost::end(c)),
        boost::begin(d), boost::end(d));

    HPX_TEST(!res);
}

template <typename ExPolicy, typename IteratorTag>
void test_lexicographical_compare1_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), 0);

    // d is lexicographical less than c
    std::vector<std::size_t> d(10006);
    std::iota(boost::begin(d), boost::end(d), 0);

    hpx::future<bool> f =
        hpx::parallel::lexicographical_compare(p,
            iterator(boost::begin(c)), iterator(boost::end(c)),
            boost::begin(d), boost::end(d));

    f.wait();

    bool res = f.get();

    HPX_TEST(!res);
}

template <typename IteratorTag>
void test_lexicographical_compare1()
{
    using namespace hpx::parallel;
    test_lexicographical_compare1(seq, IteratorTag());
    test_lexicographical_compare1(par, IteratorTag());
    test_lexicographical_compare1(par_vec, IteratorTag());

    test_lexicographical_compare1_async(seq(task), IteratorTag());
    test_lexicographical_compare1_async(par(task), IteratorTag());

    test_lexicographical_compare1(execution_policy(seq), IteratorTag());
    test_lexicographical_compare1(execution_policy(par), IteratorTag());
    test_lexicographical_compare1(execution_policy(par_vec), IteratorTag());
    test_lexicographical_compare1(execution_policy(seq(task)), IteratorTag());
    test_lexicographical_compare1(execution_policy(par(task)), IteratorTag());
}

void lexicographical_compare_test1()
{
    test_lexicographical_compare1<std::random_access_iterator_tag>();
    test_lexicographical_compare1<std::forward_iterator_tag>();
    test_lexicographical_compare1<std::input_iterator_tag>();
}

////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_lexicographical_compare2(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    // lexicographically equal, so result is false
    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), 0);

    std::vector<std::size_t> d(10007);
    std::iota(boost::begin(d), boost::end(d), 0);

    bool res = hpx::parallel::lexicographical_compare(policy,
        iterator(boost::begin(c)), iterator(boost::end(c)),
        boost::begin(d), boost::end(d));

    HPX_TEST(!res);
}

template <typename ExPolicy, typename IteratorTag>
void test_lexicographical_compare2_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    // lexicographically equal, so result is false
    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), 0);

    std::vector<std::size_t> d(10007);
    std::iota(boost::begin(d), boost::end(d), 0);

    hpx::future<bool> f =
        hpx::parallel::lexicographical_compare(p,
            iterator(boost::begin(c)), iterator(boost::end(c)),
            boost::begin(d), boost::end(d));

    f.wait();

    HPX_TEST(!f.get());
}

template <typename IteratorTag>
void test_lexicographical_compare2()
{
    using namespace hpx::parallel;
    test_lexicographical_compare2(seq, IteratorTag());
    test_lexicographical_compare2(par, IteratorTag());
    test_lexicographical_compare2(par_vec, IteratorTag());

    test_lexicographical_compare2_async(seq(task), IteratorTag());
    test_lexicographical_compare2_async(par(task), IteratorTag());

    test_lexicographical_compare2(execution_policy(seq), IteratorTag());
    test_lexicographical_compare2(execution_policy(par), IteratorTag());
    test_lexicographical_compare2(execution_policy(par_vec), IteratorTag());
    test_lexicographical_compare2(execution_policy(seq(task)), IteratorTag());
    test_lexicographical_compare2(execution_policy(par(task)), IteratorTag());
}

void lexicographical_compare_test2()
{
    test_lexicographical_compare2<std::random_access_iterator_tag>();
    test_lexicographical_compare2<std::forward_iterator_tag>();
    test_lexicographical_compare2<std::input_iterator_tag>();
}

////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_lexicographical_compare3(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    // C is lexicographically less due to the (std::rand() % size + 1)th
    // element being less than D
    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), 0);
    c[(std::rand() % 5000) + 1] = 0;

    std::vector<std::size_t> d(10007);
    std::iota(boost::begin(d), boost::end(d), 0);

    bool res = hpx::parallel::lexicographical_compare(policy,
        iterator(boost::begin(c)), iterator(boost::end(c)),
        boost::begin(d), boost::end(d));

    HPX_TEST(res);
}

template <typename ExPolicy, typename IteratorTag>
void test_lexicographical_compare3_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), 0);
    c[(std::rand() % 10006) + 1] = 0;

    std::vector<std::size_t> d(10007);
    std::iota(boost::begin(d), boost::end(d), 0);

    hpx::future<bool> f =
        hpx::parallel::lexicographical_compare(p,
            iterator(boost::begin(c)), iterator(boost::end(c)),
            boost::begin(d), boost::end(d));

    f.wait();

    HPX_TEST(f.get());
}

template <typename IteratorTag>
void test_lexicographical_compare3()
{
    using namespace hpx::parallel;
    test_lexicographical_compare3(seq, IteratorTag());
    test_lexicographical_compare3(par, IteratorTag());
    test_lexicographical_compare3(par_vec, IteratorTag());

    test_lexicographical_compare3_async(seq(task), IteratorTag());
    test_lexicographical_compare3_async(par(task), IteratorTag());

    test_lexicographical_compare3(execution_policy(seq), IteratorTag());
    test_lexicographical_compare3(execution_policy(par), IteratorTag());
    test_lexicographical_compare3(execution_policy(par_vec), IteratorTag());
    test_lexicographical_compare3(execution_policy(seq(task)), IteratorTag());
    test_lexicographical_compare3(execution_policy(par(task)), IteratorTag());
}

void lexicographical_compare_test3()
{
    test_lexicographical_compare3<std::random_access_iterator_tag>();
    test_lexicographical_compare3<std::forward_iterator_tag>();
    test_lexicographical_compare3<std::input_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_lexicographical_compare_exception(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), 0);

    std::vector<std::size_t> h(10007);
    std::iota(boost::begin(h), boost::end(h), 0);

    bool caught_exception = false;
    try {
        hpx::parallel::lexicographical_compare(policy,
            decorated_iterator(
                boost::begin(c),
                [](){ throw std::runtime_error("test"); }),
            decorated_iterator(
                boost::end(c),
                [](){ throw std::runtime_error("test"); }),
            boost::begin(h), boost::end(h));
        HPX_TEST(false);
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
void test_lexicographical_compare_async_exception(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    std::fill(boost::begin(c), boost::end(c), std::rand() + 1);

    std::vector<std::size_t> h(10006);
    std::fill(boost::begin(h), boost::end(h), std::rand() + 1);

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try {
        hpx::future<bool> f =
            hpx::parallel::lexicographical_compare(p,
                decorated_iterator(
                    boost::begin(c),
                    [](){ throw std::runtime_error("test"); }),
                decorated_iterator(
                    boost::end(c),
                    [](){ throw std::runtime_error("test"); }),
            boost::begin(h), boost::end(h));
        returned_from_algorithm = true;
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
    HPX_TEST(returned_from_algorithm);
}

template <typename IteratorTag>
void test_lexicographical_compare_exception()
{
    using namespace hpx::parallel;
    //If the execution policy object is of type vector_execution_policy,
    //  std::terminate shall be called. therefore we do not test exceptions
    //  with a vector execution policy
    test_lexicographical_compare_exception(seq, IteratorTag());
    test_lexicographical_compare_exception(par, IteratorTag());

    test_lexicographical_compare_async_exception(seq(task), IteratorTag());
    test_lexicographical_compare_async_exception(par(task), IteratorTag());

    test_lexicographical_compare_exception(execution_policy(par), IteratorTag());
    test_lexicographical_compare_exception(execution_policy(seq(task)), IteratorTag());
    test_lexicographical_compare_exception(execution_policy(par(task)), IteratorTag());
}

void lexicographical_compare_exception_test()
{
    test_lexicographical_compare_exception<std::random_access_iterator_tag>();
    test_lexicographical_compare_exception<std::forward_iterator_tag>();
}

//////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_lexicographical_compare_bad_alloc(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    std::fill(boost::begin(c), boost::end(c), std::rand() + 1);

    std::vector<std::size_t> h(10006);
    std::fill(boost::begin(h), boost::end(h), std::rand() + 1);

    bool caught_bad_alloc = false;
    try {
        hpx::parallel::lexicographical_compare(policy,
            decorated_iterator(
                boost::begin(c),
                [](){ throw std::bad_alloc(); }),
            decorated_iterator(
                boost::end(c),
                [](){ throw std::bad_alloc(); }),
            boost::begin(h), boost::end(h));
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
void test_lexicographical_compare_async_bad_alloc(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    std::fill(boost::begin(c), boost::end(c), std::rand() + 1);

    std::vector<std::size_t> h(10006);
    std::fill(boost::begin(h), boost::end(h), std::rand() + 1);

    bool caught_bad_alloc = false;
    bool returned_from_algorithm = false;
    try {
        hpx::future<bool> f =
            hpx::parallel::lexicographical_compare(p,
                decorated_iterator(
                    boost::begin(c),
                    [](){ throw std::bad_alloc(); }),
                decorated_iterator(
                    boost::end(c),
                    [](){ throw std::bad_alloc(); }),
                boost::begin(h), boost::end(h));
        returned_from_algorithm = true;

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
    HPX_TEST(returned_from_algorithm);
}

template <typename IteratorTag>
void test_lexicographical_compare_bad_alloc()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_lexicographical_compare_bad_alloc(par, IteratorTag());
    test_lexicographical_compare_bad_alloc(seq, IteratorTag());

    test_lexicographical_compare_async_bad_alloc(seq(task), IteratorTag());
    test_lexicographical_compare_async_bad_alloc(par(task), IteratorTag());

    test_lexicographical_compare_bad_alloc(execution_policy(par), IteratorTag());
    test_lexicographical_compare_bad_alloc(execution_policy(seq), IteratorTag());
    test_lexicographical_compare_bad_alloc(execution_policy(seq(task)), IteratorTag());
    test_lexicographical_compare_bad_alloc(execution_policy(par(task)), IteratorTag());
}

void lexicographical_compare_bad_alloc_test()
{
    test_lexicographical_compare_bad_alloc<std::random_access_iterator_tag>();
    test_lexicographical_compare_bad_alloc<std::forward_iterator_tag>();
}

int hpx_main(boost::program_options::variables_map& vm)
{

    unsigned int seed = (unsigned int)std::time(0);
    if(vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    lexicographical_compare_test1();
    lexicographical_compare_test2();
    lexicographical_compare_test3();
    lexicographical_compare_exception_test();
    lexicographical_compare_bad_alloc_test();
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    using namespace boost::program_options;
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ("seed,s", value<unsigned int>(),
        "the random number generator seed to use for this run")
        ;

    std::vector<std::string> cfg;
    cfg.push_back("hpx.os_threads=" +
        boost::lexical_cast<std::string>(hpx::threads::hardware_concurrency()));

    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
