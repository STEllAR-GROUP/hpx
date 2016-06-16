//  copyright (c) 2014 Grant Mercer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/parallel/algorithm.hpp>
#include <hpx/include/parallel_find.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/range/functions.hpp>

#include <numeric>
#include <string>
#include <vector>

#include "test_utils.hpp"

////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_find_first_of(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    int find_first_of_pos = std::rand() % 10007;
    int random_sub_seq_pos = std::rand() % 3;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand() + 19);
    std::size_t h[] = {1, 7, 18, 3};
    c[find_first_of_pos] = h[random_sub_seq_pos]; //-V108

    auto op = [](std::size_t a, std::size_t b)
    {
        return !(a != b);
    };

    iterator index = hpx::parallel::find_first_of(policy,
        iterator(boost::begin(c)), iterator(boost::end(c)),
        boost::begin(h), boost::end(h), op);

    base_iterator test_index = boost::begin(c) + find_first_of_pos;

    HPX_TEST(index == iterator(test_index));
}

template <typename ExPolicy, typename IteratorTag>
void test_find_first_of_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;


    int find_first_of_pos = std::rand() % 10007;
    int random_sub_seq_pos = std::rand() % 3;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand() + 19);
    std::size_t h[] = {1, 7, 18, 3};
    c[find_first_of_pos] = h[random_sub_seq_pos]; //-V108

    auto op = [](std::size_t a, std::size_t b)
    {
        return !(a != b);
    };

    hpx::future<iterator> f =
        hpx::parallel::find_first_of(p,
            iterator(boost::begin(c)), iterator(boost::end(c)),
            boost::begin(h), boost::end(h),op);
    f.wait();

    // create iterator at position of value to be found
    base_iterator test_index = boost::begin(c) + find_first_of_pos;

    HPX_TEST(f.get() == iterator(test_index));
}

template <typename IteratorTag>
void test_find_first_of()
{
    using namespace hpx::parallel;
    test_find_first_of(seq, IteratorTag());
    test_find_first_of(par, IteratorTag());
    test_find_first_of(par_vec, IteratorTag());

    test_find_first_of_async(seq(task), IteratorTag());
    test_find_first_of_async(par(task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_find_first_of(execution_policy(seq), IteratorTag());
    test_find_first_of(execution_policy(par), IteratorTag());
    test_find_first_of(execution_policy(par_vec), IteratorTag());

    test_find_first_of(execution_policy(seq(task)), IteratorTag());
    test_find_first_of(execution_policy(par(task)), IteratorTag());
#endif
}

void find_first_of_test()
{
    test_find_first_of<std::random_access_iterator_tag>();
    test_find_first_of<std::forward_iterator_tag>();
    test_find_first_of<std::input_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_find_first_of_exception(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;
    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand() + 1);
    c[c.size()/2] = 1;

    std::size_t h[] = { 1, 2 };

    auto op = [](std::size_t a, std::size_t b)
    {
        return !(a != b);
    };

    bool caught_exception = false;
    try {
        hpx::parallel::find_first_of(policy,
            decorated_iterator(
                boost::begin(c),
                [](){ throw std::runtime_error("test"); }),
            decorated_iterator(boost::end(c)),
            boost::begin(h), boost::end(h),op);
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
void test_find_first_of_exception_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand() + 1);
    c[c.size()/2] = 1;

    std::size_t h[] = { 1, 2 };

    auto op = [](std::size_t a, std::size_t b)
    {
        return !(a != b);
    };

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try {
        hpx::future<decorated_iterator> f =
            hpx::parallel::find_first_of(p,
                decorated_iterator(
                    boost::begin(c),
                    [](){ throw std::runtime_error("test"); }),
                decorated_iterator(boost::end(c)),
                boost::begin(h), boost::end(h),op);
        returned_from_algorithm = true;
        f.get();

        HPX_TEST(false);
    }
    catch(hpx::exception_list const& e) {
        caught_exception = true;
        test::test_num_exceptions<ExPolicy, IteratorTag>::call(p, e);
    }
    catch(...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
    HPX_TEST(returned_from_algorithm);
}

template <typename IteratorTag>
void test_find_first_of_exception()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_find_first_of_exception(seq, IteratorTag());
    test_find_first_of_exception(par, IteratorTag());

    test_find_first_of_exception_async(seq(task), IteratorTag());
    test_find_first_of_exception_async(par(task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_find_first_of_exception(execution_policy(seq), IteratorTag());
    test_find_first_of_exception(execution_policy(par), IteratorTag());

    test_find_first_of_exception(execution_policy(seq(task)), IteratorTag());
    test_find_first_of_exception(execution_policy(par(task)), IteratorTag());
#endif
}

void find_first_of_exception_test()
{
    test_find_first_of_exception<std::random_access_iterator_tag>();
    test_find_first_of_exception<std::forward_iterator_tag>();
    test_find_first_of_exception<std::input_iterator_tag>();
}

//////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_find_first_of_bad_alloc(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(100007);
    std::iota(boost::begin(c), boost::end(c), std::rand() + 1);
    c[c.size()/2] = 1;

    std::size_t h[] = { 1, 2 };

    auto op = [](std::size_t a, std::size_t b)
    {
        return !(a != b);
    };

    bool caught_bad_alloc = false;
    try {
        hpx::parallel::find_first_of(policy,
            decorated_iterator(
                boost::begin(c),
                [](){ throw std::bad_alloc(); }),
            decorated_iterator(boost::end(c)),
            boost::begin(h), boost::end(h),op);
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
void test_find_first_of_bad_alloc_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand() + 1);
    c[c.size()/2] = 1;

    std::size_t h[] = { 1, 2 };

    auto op = [](std::size_t a, std::size_t b)
    {
        return !(a != b);
    };

    bool caught_bad_alloc = false;
    bool returned_from_algorithm = false;
    try {
        hpx::future<decorated_iterator> f =
            hpx::parallel::find_first_of(p,
                decorated_iterator(
                    boost::begin(c),
                    [](){ throw std::bad_alloc(); }),
                decorated_iterator(boost::end(c)),
                boost::begin(h), boost::end(h),op);
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
void test_find_first_of_bad_alloc()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_find_first_of_bad_alloc(seq, IteratorTag());
    test_find_first_of_bad_alloc(par, IteratorTag());

    test_find_first_of_bad_alloc_async(seq(task), IteratorTag());
    test_find_first_of_bad_alloc_async(par(task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_find_first_of_bad_alloc(execution_policy(seq), IteratorTag());
    test_find_first_of_bad_alloc(execution_policy(par), IteratorTag());

    test_find_first_of_bad_alloc(execution_policy(seq(task)), IteratorTag());
    test_find_first_of_bad_alloc(execution_policy(par(task)), IteratorTag());
#endif
}

void find_first_of_bad_alloc_test()
{

    test_find_first_of_bad_alloc<std::random_access_iterator_tag>();
    test_find_first_of_bad_alloc<std::forward_iterator_tag>();
    test_find_first_of_bad_alloc<std::input_iterator_tag>();

}

int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    find_first_of_test();
    find_first_of_exception_test();
    find_first_of_bad_alloc_test();
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
