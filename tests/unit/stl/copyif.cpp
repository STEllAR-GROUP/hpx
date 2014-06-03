//  Copyright (c) 2014 Grant Mercer
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
void test_copy_if(ExPolicy const& policy, IteratorTag)
{
	BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

	typedef std::vector<int>::iterator base_iterator;
	typedef test::test_iterator<base_iterator, IteratorTag> iterator;

	std::vector<int> c(10007);
	std::vector<int> d(c.size());
	auto middle = boost::begin(c) + c.size()/2;
	std::iota(boost::begin(c), middle, std::rand());
	std::fill(middle, boost::end(c), -1);

	base_iterator outiter = hpx::parallel::copy_if(policy,
		iterator(boost::begin(c)), iterator(boost::end(c)),
		boost::begin(d), [](int i){return !(i<0);});

	std::size_t count = 0;
	HPX_TEST(std::equal(boost::begin(c), middle, boost::begin(d),
		[&count](int v1, int v2) {
			HPX_TEST_EQ(v1, v2);
			++count;
			return v1 == v2;
		}));

	HPX_TEST(std::equal(middle,boost::end(c), 
		boost::begin(d) + (1 + d.size()/2),
		[&count](int v1, int v2) {
			HPX_TEST_NEQ(v1,v2);
			++count;
			return v1!=v2;
	}));

	HPX_TEST_EQ(count, d.size());
}

template <typename IteratorTag>
void test_copy_if(hpx::parallel::task_execution_policy, IteratorTag)
{
	typedef std::vector<int>::iterator base_iterator;
	typedef test::test_iterator<base_iterator, IteratorTag> iterator;

	std::vector<int> c(10007);
	std::vector<int> d(c.size());
	auto middle = boost::begin(c) + c.size()/2;
	std::iota(boost::begin(c), middle, std::rand());
	std::fill(middle, boost::end(c), -1);

	hpx::future<base_iterator> f =
		hpx::parallel::copy_if(hpx::parallel::task,
			iterator(boost::begin(c)), iterator(boost::end(c)), 
			boost::begin(d), [](int i){return !(i<0);});
	f.wait();

	std::size_t count = 0;
	HPX_TEST(std::equal(boost::begin(c), middle, boost::begin(d),
		[&count](int v1, int v2) {
			HPX_TEST_EQ(v1, v2);
			++count;
			return v1 == v2;
		}));

	HPX_TEST(std::equal(middle,boost::end(c), 
		boost::begin(d) + (1 + d.size()/2),
		[&count](int v1, int v2) {
			HPX_TEST_NEQ(v1,v2);
			++count;
			return v1!=v2;
	}));

	HPX_TEST_EQ(count, d.size());
}

template <typename IteratorTag>
void test_copy_if()
{
	using namespace hpx::parallel;

	test_copy_if(seq, IteratorTag());
	test_copy_if(par, IteratorTag());
	test_copy_if(vec, IteratorTag());
	test_copy_if(task, IteratorTag());

	test_copy_if(execution_policy(seq), IteratorTag());
	test_copy_if(execution_policy(par), IteratorTag());
	test_copy_if(execution_policy(vec), IteratorTag());
	test_copy_if(execution_policy(task), IteratorTag());
}

void if_copy_test()
{
	test_copy_if<std::random_access_iterator_tag>();
	test_copy_if<std::forward_iterator_tag>();
	test_copy_if<std::input_iterator_tag>();
}

int hpx_main()
{
	if_copy_test();
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
