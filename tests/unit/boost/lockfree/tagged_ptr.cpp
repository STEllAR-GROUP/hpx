#include <boost/lockfree/detail/tagged_ptr.hpp>

#include <climits>
#define BOOST_TEST_MODULE lockfree_tests
#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_CASE( tagged_ptr_test )
{
    using namespace boost::lockfree;
    int a(1), b(2);

    {
        tagged_ptr<int> i (&a, 0);
        tagged_ptr<int> j (&b, 1);

        i = j;

        BOOST_REQUIRE_EQUAL(i.get_ptr(), &b);
        BOOST_REQUIRE_EQUAL(i.get_tag(), 1);
    }

    {
        tagged_ptr<int> i (&a, 0);
        tagged_ptr<int> j (i);

        BOOST_REQUIRE_EQUAL(i.get_ptr(), j.get_ptr());
        BOOST_REQUIRE_EQUAL(i.get_tag(), j.get_tag());
    }

}
