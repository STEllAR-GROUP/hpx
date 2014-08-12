//  copyright (c) 2014 Grant Mercer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/algorithm.hpp>
#include <hpx/util/lightweight_test.hpp>

#include "test_utils.hpp"

////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_generate_n(ExPolicy const& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    
    auto gen = [](){ return std::size_t(10); };

    hpx::parallel::generate_n(policy,
        iterator(boost::begin(c)), c.size(), gen);

    // verify values
    std::size_t count = 0;
    std::for_each(boost::begin(c), boost::end(c),
        [&count](std::size_t v) {
            HPX_TEST_EQ(v, std::size_t(10));
            ++count;
        });
    HPX_TEST_EQ(count, c.size());
}

template <typename IteratorTag>
void test_generate_n(hpx::parallel::task_execution_policy, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    
    auto gen = [](){ return std::size_t(10); };

    hpx::future<iterator> f =
        hpx::parallel::generate_n(hpx::parallel::task,
            iterator(boost::begin(c)), c.size(),
            gen);
    f.wait();

    std::size_t count =0;
    std::for_each(boost::begin(c), boost::end(c),
        [&count](std::size_t v) {
            HPX_TEST_EQ(v, std::size_t(10));
            ++count;
    });
    HPX_TEST_EQ(count, c.size());
}

template <typename IteratorTag>
void test_generate_n()
{
    using namespace hpx::parallel;
    test_generate_n(seq, IteratorTag());
    test_generate_n(par, IteratorTag());
    test_generate_n(par_vec, IteratorTag());
    test_generate_n(task, IteratorTag());

    test_generate_n(execution_policy(seq), IteratorTag());
    test_generate_n(execution_policy(par), IteratorTag());
    test_generate_n(execution_policy(par_vec), IteratorTag());
    test_generate_n(execution_policy(task), IteratorTag());
}

void generate_n_test()
{
    test_generate_n<std::random_access_iterator_tag>();
    test_generate_n<std::forward_iterator_tag>();
    test_generate_n<std::output_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_generate_n_exception(ExPolicy const& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;
    std::vector<std::size_t> c(10007);
    
    auto gen = [](){ return std::size_t(10); };

    bool caught_exception = false;
    try {
        hpx::parallel::generate_n(policy,
            decorated_iterator(
                boost::begin(c),
                [](){ throw std::runtime_error("test"); }),
            c.size(),
            gen);
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

template <typename IteratorTag>
void test_generate_n_exception(hpx::parallel::task_execution_policy, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    
    auto gen = [](){ return std::size_t(10); };

    bool caught_exception = false;
    try {
        hpx::future<decorated_iterator> f =
            hpx::parallel::generate_n(hpx::parallel::task,
                decorated_iterator(
                    boost::begin(c),
                    [](){ throw std::runtime_error("test"); }),
                c.size(),
                gen);
        f.get();

        HPX_TEST(false);
    }
    catch(hpx::exception_list const& e) {
        caught_exception = true;
        test::test_num_exceptions<
            hpx::parallel::task_execution_policy, IteratorTag
        >::call(hpx::parallel::task, e);
    }
    catch(...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
}

template <typename IteratorTag>
void test_generate_n_exception()
{
    using namespace hpx::parallel;
    //If the execution policy object is of type vector_execution_policy,
    //  std::terminate shall be called. therefore we do not test exceptions
    //  with a vector execution policy
    test_generate_n_exception(seq, IteratorTag());
    test_generate_n_exception(par, IteratorTag());
    test_generate_n_exception(task, IteratorTag());

    test_generate_n_exception(execution_policy(seq), IteratorTag());
    test_generate_n_exception(execution_policy(par), IteratorTag());
    test_generate_n_exception(execution_policy(task), IteratorTag());
}

void generate_n_exception_test()
{
    test_generate_n_exception<std::random_access_iterator_tag>();
    test_generate_n_exception<std::forward_iterator_tag>();
}

//////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_generate_n_bad_alloc(ExPolicy const& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(100007);
    
    auto gen=[](){ return std::size_t(10); };

    bool caught_bad_alloc = false;
    try {
        hpx::parallel::generate_n(policy,
            decorated_iterator(
                boost::begin(c),
                [](){ throw std::bad_alloc(); }),
            c.size(),
            gen);
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
void test_generate_n_bad_alloc(hpx::parallel::task_execution_policy, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    
    auto gen = [](){ return std::size_t(10); };

    bool caught_bad_alloc = false;
    try {
        hpx::future<decorated_iterator> f =
            hpx::parallel::generate_n(hpx::parallel::task,
                decorated_iterator(
                    boost::begin(c),
                    [](){ throw std::bad_alloc(); }),
                c.size(),
                gen);

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
void test_generate_n_bad_alloc()
{
    using namespace hpx::parallel;
    //If the execution policy object is of type vector_execution_policy,
    //  std::terminate shall be called. therefore we do not test exceptions
    //  with a vector execution policy
    test_generate_n_bad_alloc(seq, IteratorTag());
    test_generate_n_bad_alloc(par, IteratorTag());
    test_generate_n_bad_alloc(task, IteratorTag());

    test_generate_n_bad_alloc(execution_policy(seq), IteratorTag());
    test_generate_n_bad_alloc(execution_policy(par), IteratorTag());
    test_generate_n_bad_alloc(execution_policy(task), IteratorTag());
}

void generate_n_bad_alloc_test()
{
    test_generate_n_bad_alloc<std::random_access_iterator_tag>();
    test_generate_n_bad_alloc<std::forward_iterator_tag>();
}

int hpx_main()
{
    generate_n_test();
    generate_n_exception_test();
    generate_n_bad_alloc_test();
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    std::vector<std::string> cfg;
    cfg.push_back("hpx.os_threads=" +
        boost::lexical_cast<std::string>(hpx::threads::hardware_concurrency()));

    HPX_TEST_EQ_MSG(hpx::init(argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();

}
