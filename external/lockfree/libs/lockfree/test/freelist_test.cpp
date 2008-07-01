#include <boost/lockfree/freelist.hpp>

#include <climits>
#define BOOST_TEST_MODULE lockfree_tests
#include <boost/test/included/unit_test.hpp>

#include <boost/foreach.hpp>

#include <vector>

class dummy
{
    int foo[64];
};

boost::lockfree::freelist<dummy, 64> fl;

BOOST_AUTO_TEST_CASE( freelist_test1 )
{
    std::vector<dummy*> nodes;

    for (int i = 0; i != 128; ++i)
        nodes.push_back(fl.allocate());

    BOOST_FOREACH(dummy * d, nodes)
        fl.deallocate(d);

    for (int i = 0; i != 128; ++i)
        nodes.push_back(fl.allocate());
}

boost::lockfree::caching_freelist<dummy> cfl;

BOOST_AUTO_TEST_CASE( freelist_test2 )
{
    std::vector<dummy*> nodes;

    for (int i = 0; i != 128; ++i)
        nodes.push_back(cfl.allocate());

    BOOST_FOREACH(dummy * d, nodes)
        cfl.deallocate(d);

    for (int i = 0; i != 128; ++i)
        nodes.push_back(cfl.allocate());
}
