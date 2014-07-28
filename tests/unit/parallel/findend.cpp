//  copyright (c) 2014 Grant Mercer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_find.hpp>
#include <hpx/util/lightweight_test.hpp>

#include "test_utils.hpp"

////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_find_end(ExPolicy const& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    //fill vector with random values about 1
    std::fill(boost::begin(c), boost::end(c), (std::rand()%100)+ 3);
    c.at(c.size()/2) = 1;
    c.at(c.size()/2 + 1)= 2;

    std::size_t h[] = {1,2};

    iterator index = hpx::parallel::find_end(policy,
        iterator(boost::begin(c)), iterator(boost::end(c)), 
        boost::begin(h), boost::end(h));

    base_iterator test_index = boost::begin(c) + c.size()/2;

    HPX_TEST(index == iterator(test_index));
}

template <typename IteratorTag>
void test_find_end(hpx::parallel::task_execution_policy, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    //fill vector with random values above 1
    std::fill(boost::begin(c), boost::end(c), (std::rand()%100) + 3);
    c.at(c.size()/2) = 1;
    c.at(c.size()/2 + 1)= 2;

    std::size_t h[] = {1,2};

    hpx::future<iterator> f =
        hpx::parallel::find_end(hpx::parallel::task,
            iterator(boost::begin(c)), iterator(boost::end(c)),
            boost::begin(h), boost::end(h));
    f.wait();

    //create iterator at position of value to be found
    base_iterator test_index = boost::begin(c) + c.size()/2;

    HPX_TEST(f.get() == iterator(test_index));
}

template <typename IteratorTag>
void test_find_end()
{
    using namespace hpx::parallel;
    //test_find_end(seq, IteratorTag());
    test_find_end(par, IteratorTag());
    //test_find_end(par_vec, IteratorTag());
    //test_find_end(task, IteratorTag());

    
    //test_find_end(execution_policy(seq), IteratorTag());
    //test_find_end(execution_policy(par), IteratorTag());
    //test_find_end(execution_policy(par_vec), IteratorTag());
    //test_find_end(execution_policy(task), IteratorTag());
    
}

void find_end_test()
{
    test_find_end<std::random_access_iterator_tag>();
    //test_find_end<std::forward_iterator_tag>();
}


///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_find_end_exception(ExPolicy const& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;
    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand()+1);
    c[c.size()/2]=0;

    std::size_t h[] = {1,2};

    bool caught_exception = false;
    try {
        hpx::parallel::find_end(policy,
            decorated_iterator(
                boost::begin(c),
                [](){ throw std::runtime_error("test"); }),
            decorated_iterator(boost::end(c), [](){}),
            boost::begin(h), boost::end(h));
        HPX_TEST(false);
    }
    catch(hpx::exception_list const& e) {
        caught_exception = true;
        test::test_num_exeptions<ExPolicy, IteratorTag>::call(policy, e);
    }
    catch(...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
}

template <typename IteratorTag>
void test_find_end_exception(hpx::parallel::task_execution_policy, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand()+1);
    c[c.size()/2]=0;

    std::size_t h[] = {1,2};

    bool caught_exception = false;
    try {
        hpx::future<decorated_iterator> f =
            hpx::parallel::find_end(hpx::parallel::task,
                decorated_iterator(
                    boost::begin(c),
                    [](){ throw std::runtime_error("test"); }),
                decorated_iterator(boost::end(c), [](){}),
                boost::begin(h), boost::end(h));
        f.get();

        HPX_TEST(false);
    }
    catch(hpx::exception_list const& e) {
        caught_exception = true;
        test::test_num_exeptions<
            hpx::parallel::task_execution_policy, IteratorTag
        >::call(hpx::parallel::task, e);
    }
    catch(...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
}

template <typename IteratorTag>
void test_find_end_exception()
{
    using namespace hpx::parallel;
    //If the execution policy object is of type vector_execution_policy,
    //  std::terminate shall be called. therefore we do not test exceptions
    //  with a vector execution policy
    test_find_end_exception(seq, IteratorTag());
    test_find_end_exception(par, IteratorTag());
    test_find_end_exception(task, IteratorTag());

    test_find_end_exception(execution_policy(seq), IteratorTag());
    test_find_end_exception(execution_policy(par), IteratorTag());
    test_find_end_exception(execution_policy(task), IteratorTag());
}

void find_end_exception_test()
{
    test_find_end_exception<std::random_access_iterator_tag>();
    test_find_end_exception<std::forward_iterator_tag>();
}

//////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_find_end_bad_alloc(ExPolicy const& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(100007);
    std::iota(boost::begin(c), boost::end(c), std::rand()+1);
    c[c.size()/2]=0;

    std::size_t h[] = {1,2};

    bool caught_bad_alloc = false;
    try {
        hpx::parallel::find_end(policy,
            decorated_iterator(
                boost::begin(c),
                [](){ throw std::bad_alloc(); }),
            decorated_iterator(boost::end(c), [](){}),
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

template <typename IteratorTag>
void test_find_end_bad_alloc(hpx::parallel::task_execution_policy, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand()+1);
    c[c.size()/2] = 0;

    std::size_t h[] = {1,2};

    bool caught_bad_alloc = false;
    try {
        hpx::future<decorated_iterator> f =
            hpx::parallel::find_end(hpx::parallel::task,
                decorated_iterator(
                    boost::begin(c),
                    [](){ throw std::bad_alloc(); }),
                decorated_iterator(boost::end(c), [](){}),
                boost::begin(h), boost::end(h));

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
void test_find_end_bad_alloc()
{
    using namespace hpx::parallel;
    //If the execution policy object is of type vector_execution_policy,
    //  std::terminate shall be called. therefore we do not test exceptions
    //  with a vector execution policy
    test_find_end_bad_alloc(seq, IteratorTag());
    test_find_end_bad_alloc(par, IteratorTag());
    test_find_end_bad_alloc(task, IteratorTag());

    test_find_end_bad_alloc(execution_policy(seq), IteratorTag());
    test_find_end_bad_alloc(execution_policy(par), IteratorTag());
    test_find_end_bad_alloc(execution_policy(task), IteratorTag());
}

void find_end_bad_alloc_test()
{
 
    test_find_end_bad_alloc<std::random_access_iterator_tag>();
    test_find_end_bad_alloc<std::forward_iterator_tag>();
  
}

int hpx_main()
{
    int t[] = {1,2,3,1,2,9,2,3,4,9,9,2};
    int f[] = {1,2};
    auto g = hpx::parallel::find_end(hpx::parallel::par(6),
        boost::begin(t), boost::end(t), boost::begin(f),
        boost::end(f));
    //the distance **should** be 3
    std::cout << std::distance(boost::begin(t),g) << std::endl;

    //find_end_test();
    //find_end_exception_test();
    //find_end_bad_alloc_test();
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
