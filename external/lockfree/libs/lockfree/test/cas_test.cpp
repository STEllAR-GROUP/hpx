#include <boost/lockfree/cas.hpp>

#include <climits>
#define BOOST_TEST_MODULE lockfree_tests
#include <boost/test/included/unit_test.hpp>

#include <cstddef>

struct cas2_tst
{
    long i;
    long j;
};

struct cas2_tst2
{
    void* i;
    long j;
};

using namespace boost::lockfree;


BOOST_AUTO_TEST_CASE( cas_test )
{
    {
        int i = 1;

        BOOST_REQUIRE_EQUAL (CAS(&i, 1, 3), true);
        BOOST_REQUIRE_EQUAL (i, 3);

        BOOST_REQUIRE_EQUAL (CAS(&i, 1, 3), false);
        BOOST_REQUIRE_EQUAL (i, 3);
    }

    {
        cas2_tst tst;
        tst.i = 1;
        tst.j = 2;

        BOOST_REQUIRE_EQUAL (CAS2(&tst, 1, 2, 3, 4), true);
        BOOST_REQUIRE_EQUAL (tst.i, 3);
        BOOST_REQUIRE_EQUAL (tst.j, 4);

        BOOST_REQUIRE_EQUAL (CAS2(&tst, 1, 2, 3, 4), false);
        BOOST_REQUIRE_EQUAL (tst.i, 3);
        BOOST_REQUIRE_EQUAL (tst.j, 4);

        BOOST_REQUIRE_EQUAL (CAS2(&tst, 3, 2, 3, 4), false);
        BOOST_REQUIRE_EQUAL (CAS2(&tst, 1, 4, 3, 4), false);
    }

    {
        cas2_tst2 tst;
        tst.i = NULL;
        tst.j = 2;

        BOOST_REQUIRE_EQUAL (CAS2(&tst, (void*)0, 2, (void*)0x3, 4), true);
        BOOST_REQUIRE_EQUAL (tst.i, (void*)0x3);
        BOOST_REQUIRE_EQUAL (tst.j, 4);

        BOOST_REQUIRE_EQUAL (CAS2(&tst, (void*)0x1, 2, (void*)0x3, 4), false);
        BOOST_REQUIRE_EQUAL (tst.i, (void*)0x3);
        BOOST_REQUIRE_EQUAL (tst.j, 4);

        BOOST_REQUIRE_EQUAL (CAS2(&tst, (void*)0x3, 2, (void*)0x3, 4), false);
        BOOST_REQUIRE_EQUAL (CAS2(&tst, (void*)0x1, 4, (void*)0x3, 4), false);
    }

}
