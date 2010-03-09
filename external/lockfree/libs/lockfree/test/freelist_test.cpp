#include <boost/lockfree/detail/freelist.hpp>
#include <boost/atomic.hpp>
#include <boost/thread.hpp>

#include <climits>
#define BOOST_TEST_MODULE lockfree_tests
#include <boost/test/included/unit_test.hpp>

#include <boost/foreach.hpp>
#include <boost/static_assert.hpp>

#include <boost/type_traits/is_same.hpp>

#include <vector>


class dummy
{
    int foo[64];
};

template <typename freelist_type>
void run_test(void)
{
    freelist_type fl(1024);

    std::vector<dummy*> nodes;

    for (int i = 0; i != 128; ++i)
        nodes.push_back(fl.allocate());

    BOOST_FOREACH(dummy * d, nodes)
        fl.deallocate(d);

    nodes.clear();
    for (int i = 0; i != 128; ++i)
        nodes.push_back(fl.allocate());

    BOOST_FOREACH(dummy * d, nodes)
        fl.deallocate(d);

    for (int i = 0; i != 128; ++i)
        nodes.push_back(fl.allocate());
}

BOOST_AUTO_TEST_CASE( freelist_tests )
{
    run_test<boost::lockfree::caching_freelist<dummy> >();
    run_test<boost::lockfree::static_freelist<dummy> >();
}

template <typename freelist_type>
struct freelist_tester
{
    static const int max_nodes = 1024;
    static const int thread_count = 4;
    static const int loops_per_thread = 1024;

    boost::atomic_int free_nodes;
    boost::thread_group threads;

    freelist_type fl;

    freelist_tester(void):
        free_nodes(0), fl(max_nodes * thread_count)
    {
        for (int i = 0; i != thread_count; ++i)
            threads.create_thread(boost::bind(&freelist_tester::run, this));
        threads.join_all();
    }

    void run(void)
    {
        std::vector<dummy*> nodes;
        nodes.reserve(max_nodes);

        for (int i = 0; i != loops_per_thread; ++i) {
            while (nodes.size() < max_nodes) {
                dummy * node = fl.allocate();
                if (node == NULL)
                    break;
                nodes.push_back(node);
            }

            while (!nodes.empty()) {
                dummy * node = nodes.back();
                assert(node);
                nodes.pop_back();
                fl.deallocate(node);
            }
        }
        BOOST_REQUIRE(nodes.empty());
    }
};

BOOST_AUTO_TEST_CASE( caching_freelist_test )
{
    freelist_tester<boost::lockfree::caching_freelist<dummy> > tester();
}

BOOST_AUTO_TEST_CASE( static_freelist_test )
{
    freelist_tester<boost::lockfree::static_freelist<dummy> > tester();
}

/* using namespace boost; */
/* using namespace boost::mpl; */

/* BOOST_STATIC_ASSERT((is_same<lockfree::detail::select_freelist<int, std::allocator<int>, lockfree::caching_freelist_t>, */
/*                      lockfree::caching_freelist<int, std::allocator<int> > */
/*                      >::value)); */

/* BOOST_STATIC_ASSERT((is_same<lockfree::detail::select_freelist<int, std::allocator<int>, lockfree::static_freelist_t>, */
/*                      lockfree::static_freelist<int, std::allocator<int> > */
/*                      >::value)); */
