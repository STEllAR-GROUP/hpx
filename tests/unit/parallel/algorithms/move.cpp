//  Copyright (c) 2014 Grant Mercer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_move.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/range/functions.hpp>

#include <numeric>
#include <string>
#include <vector>

#include "test_utils.hpp"

////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_move(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::iota(boost::begin(c), boost::end(c), std::rand());
    hpx::parallel::move(policy,
        iterator(boost::begin(c)), iterator(boost::end(c)), boost::begin(d));

    //copy contents of d back into c for testing
    std::copy(boost::begin(d), boost::end(d), boost::begin(d));

    std::size_t count = 0;
    HPX_TEST(std::equal(boost::begin(c), boost::end(c), boost::begin(d),
        [&count](std::size_t v1, std::size_t v2) -> bool {
            HPX_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));
    HPX_TEST_EQ(count, d.size());

}

template <typename ExPolicy, typename IteratorTag>
void test_move_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::iota(boost::begin(c), boost::end(c), std::rand());

    auto f =
        hpx::parallel::move(p,
            iterator(boost::begin(c)), iterator(boost::end(c)),
            boost::begin(d));

    hpx::future<void> g = f.then(
        [&d, &c](hpx::future<void> f)
        {
            HPX_TEST(!f.has_exception());
            std::copy(boost::begin(d), boost::end(d), boost::begin(c));
        });
    g.wait();

    std::size_t count = 0;
    HPX_TEST(std::equal(boost::begin(c), boost::end(c), boost::begin(d),
        [&count](std::size_t v1, std::size_t v2) -> bool {
            HPX_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));
    HPX_TEST_EQ(count, d.size());
}

template <typename ExPolicy, typename IteratorTag>
void test_outiter_move(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef std::back_insert_iterator<std::vector<std::size_t> > outiterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(0);
    std::iota(boost::begin(c), boost::end(c), std::rand());
    hpx::parallel::move(policy,
        iterator(boost::begin(c)), iterator(boost::end(c)), std::back_inserter(d));

    //copy contents of d back into c for testing
    std::copy(boost::begin(d), boost::end(d), boost::begin(d));

    std::size_t count = 0;
    HPX_TEST(std::equal(boost::begin(c), boost::end(c), boost::begin(d),
        [&count](std::size_t v1, std::size_t v2) -> bool {
            HPX_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));
    HPX_TEST_EQ(count, d.size());

}

template <typename ExPolicy, typename IteratorTag>
void test_outiter_move_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef std::back_insert_iterator<std::vector<std::size_t> > outiterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(0);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    auto f =
        hpx::parallel::move(p,
        iterator(boost::begin(c)), iterator(boost::end(c)),
        std::back_inserter(d));

    hpx::future<void> g = f.then(
        [&d, &c](hpx::future<void> f)
        {
            HPX_TEST(!f.has_exception());
            std::copy(boost::begin(d), boost::end(d), boost::begin(c));
        });
    g.wait();

    std::size_t count = 0;
    HPX_TEST(std::equal(boost::begin(c), boost::end(c), boost::begin(d),
        [&count](std::size_t v1, std::size_t v2) -> bool {
            HPX_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));
    HPX_TEST_EQ(count, d.size());
}

template <typename IteratorTag>
void test_move()
{
    using namespace hpx::parallel;
    test_move(seq, IteratorTag());
    test_move(par, IteratorTag());
    test_move(par_vec, IteratorTag());

    test_move_async(seq(task), IteratorTag());
    test_move_async(par(task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_move(execution_policy(seq), IteratorTag());
    test_move(execution_policy(par), IteratorTag());
    test_move(execution_policy(par_vec), IteratorTag());

    test_move(execution_policy(seq(task)), IteratorTag());
    test_move(execution_policy(par(task)), IteratorTag());
#endif

    // output iterator test
    test_outiter_move(seq, IteratorTag());
    test_outiter_move(par, IteratorTag());
    test_outiter_move(par_vec, IteratorTag());

    test_outiter_move_async(seq(task), IteratorTag());
    test_outiter_move_async(par(task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_outiter_move(execution_policy(seq), IteratorTag());
    test_outiter_move(execution_policy(par), IteratorTag());
    test_outiter_move(execution_policy(par_vec), IteratorTag());

    test_outiter_move(execution_policy(seq(task)), IteratorTag());
    test_outiter_move(execution_policy(par(task)), IteratorTag());
#endif
}

void move_test()
{
    test_move<std::random_access_iterator_tag>();
    test_move<std::forward_iterator_tag>();
    test_move<std::input_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_move_exception(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator,IteratorTag> iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::iota(boost::begin(c), boost::end(c), std::rand());

    bool caught_exception = false;
    try {
        hpx::parallel::move(policy,
            decorated_iterator(
                boost::begin(c),
                [](){ throw std::runtime_error("test"); }),
            decorated_iterator(boost::end(c)),
            boost::begin(d));
        HPX_TEST(false);
    }
    catch (hpx::exception_list const& e) {
        caught_exception = true;
        test::test_num_exceptions<ExPolicy, IteratorTag>::call(policy, e);
    }
    catch (...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
}

template <typename ExPolicy, typename IteratorTag>
void test_move_exception_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::iota(boost::begin(c), boost::end(c), std::rand());

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try {
        auto f =
            hpx::parallel::move(p,
                decorated_iterator(
                    boost::begin(c),
                    [](){ throw std::runtime_error("test"); }),
                decorated_iterator(boost::end(c)),
                boost::begin(d));
        returned_from_algorithm = true;
        f.get();

        HPX_TEST(false);
    }
    catch (hpx::exception_list const& e) {
        caught_exception = true;
        test::test_num_exceptions<ExPolicy, IteratorTag>::call(p, e);
    }
    catch (...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
    HPX_TEST(returned_from_algorithm);
}

template <typename IteratorTag>
void test_move_exception()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_move_exception(seq, IteratorTag());
    test_move_exception(par, IteratorTag());

    test_move_exception_async(seq(task), IteratorTag());
    test_move_exception_async(par(task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_move_exception(execution_policy(seq), IteratorTag());
    test_move_exception(execution_policy(par), IteratorTag());

    test_move_exception(execution_policy(seq(task)), IteratorTag());
    test_move_exception(execution_policy(par(task)), IteratorTag());
#endif
}

void move_exception_test()
{
    test_move_exception<std::random_access_iterator_tag>();
    test_move_exception<std::forward_iterator_tag>();
    test_move_exception<std::input_iterator_tag>();
}

//////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_move_bad_alloc(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(100007);
    std::vector<std::size_t> d(c.size());
    std::iota(boost::begin(c), boost::end(c), std::rand());

    bool caught_bad_alloc = false;
    try {
        hpx::parallel::move(policy,
            decorated_iterator(
                boost::begin(c),
                [](){ throw std::bad_alloc(); }),
            decorated_iterator(boost::end(c)),
            boost::begin(d));
        HPX_TEST(false);
    }
    catch (std::bad_alloc const&) {
        caught_bad_alloc = true;
    }
    catch (...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_bad_alloc);
}

template <typename ExPolicy, typename IteratorTag>
void test_move_bad_alloc_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::iota(boost::begin(c), boost::end(c), std::rand());

    bool caught_bad_alloc = false;
    bool returned_from_algorithm = false;
    try {
        auto f =
            hpx::parallel::move(p,
                decorated_iterator(
                    boost::begin(c),
                    [](){ throw std::bad_alloc(); }),
                decorated_iterator(boost::end(c)),
                boost::begin(d));
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
void test_move_bad_alloc()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_move_bad_alloc(seq, IteratorTag());
    test_move_bad_alloc(par, IteratorTag());

    test_move_bad_alloc_async(seq(task), IteratorTag());
    test_move_bad_alloc_async(par(task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_move_bad_alloc(execution_policy(seq), IteratorTag());
    test_move_bad_alloc(execution_policy(par), IteratorTag());

    test_move_bad_alloc(execution_policy(seq(task)), IteratorTag());
    test_move_bad_alloc(execution_policy(par(task)), IteratorTag());
#endif
}

void move_bad_alloc_test()
{
    test_move_bad_alloc<std::random_access_iterator_tag>();
    test_move_bad_alloc<std::forward_iterator_tag>();
    test_move_bad_alloc<std::input_iterator_tag>();
}

int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    move_test();
    move_exception_test();
    move_bad_alloc_test();
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // add command line option which controls the random number generator seed
    using namespace boost::program_options;
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ("seed,s", value<unsigned int>(),
        "the random number generator seed to use for this run")
        ;
    // By default this test should run on all available cores
    std::vector<std::string> cfg;
    cfg.push_back("hpx.os_threads=" +
        std::to_string(hpx::threads::hardware_concurrency()));

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
