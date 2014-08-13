//  Copyright (c) 2014 Grant Mercer
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
void test_adjacent_find(ExPolicy const& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    //fill vector with random values about 1
    std::iota(boost::begin(c), boost::end(c), (std::rand()%100)+2);

    std::size_t random_pos = std::rand() % 10006;

    c[random_pos] = 1;
    c[random_pos+1] = 1;

    iterator index = hpx::parallel::adjacent_find(policy,
        iterator(boost::begin(c)), iterator(boost::end(c)));

    base_iterator test_index = boost::begin(c) + random_pos;

    HPX_TEST(index == iterator(test_index));
}

template <typename IteratorTag>
void test_adjacent_find(hpx::parallel::task_execution_policy, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    //fill vector with random values above 1
    std::iota(boost::begin(c), boost::end(c), (std::rand()%100) + 2);

    std::size_t random_pos = std::rand() % 10006;

    c[random_pos] = 1;
    c[random_pos+1] = 1;

    hpx::future<iterator> f =
        hpx::parallel::adjacent_find(hpx::parallel::task,
            iterator(boost::begin(c)), iterator(boost::end(c)));
    f.wait();

    //create iterator at position of value to be found
    base_iterator test_index = boost::begin(c) + random_pos;

    HPX_TEST(f.get() == iterator(test_index));
}

template <typename IteratorTag>
void test_adjacent_find()
{
    using namespace hpx::parallel;
    test_adjacent_find(seq, IteratorTag());
    test_adjacent_find(par, IteratorTag());
    test_adjacent_find(par_vec, IteratorTag());
    test_adjacent_find(task, IteratorTag());

    test_adjacent_find(execution_policy(seq), IteratorTag());
    test_adjacent_find(execution_policy(par), IteratorTag());
    test_adjacent_find(execution_policy(par_vec), IteratorTag());
    test_adjacent_find(execution_policy(task), IteratorTag());
}

void adjacent_find_test()
{
    test_adjacent_find<std::random_access_iterator_tag>();
    test_adjacent_find<std::forward_iterator_tag>();
}

int hpx_main()
{
    adjacent_find_test();
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