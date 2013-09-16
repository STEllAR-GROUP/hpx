/*=============================================================================
    Copyright (c) 2013 Shuangyang Yang

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
==============================================================================*/

#include <hpx/config.hpp>

#include <cstddef> // NULL
#include <cstdio> // remove
#include <fstream>

#include <boost/config.hpp>
#if defined(BOOST_NO_STDC_NAMESPACE)
namespace std
{
    using ::remove;
}
#endif

#include <hpx/hpx_init.hpp>
#include <hpx/util/any.hpp>

#include "serialization_test_tools.hpp"
#include "small_big_object.hpp"

#include <boost/serialization/access.hpp>
#include <boost/serialization/level.hpp>
#include <boost/serialization/nvp.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;

using hpx::util::basic_any;

using hpx::init;
using hpx::finalize;

// note: version can be assigned only to objects whose implementation
// level is object_class_info.  So, doing the following will result in
// a static assertion
// BOOST_CLASS_VERSION(A, 2);

template <typename A>
void out(const char *testfile, A & a)
{
    test_ostream os(testfile, TEST_STREAM_FLAGS);
    test_oarchive oa(os, TEST_ARCHIVE_FLAGS);
    oa << BOOST_SERIALIZATION_NVP(a);
}

template <typename A>
void in(const char *testfile, A & a)
{
    test_istream is(testfile, TEST_STREAM_FLAGS);
    test_iarchive ia(is, TEST_ARCHIVE_FLAGS);
    ia >> BOOST_SERIALIZATION_NVP(a);
}

int hpx_main(variables_map& vm)
{
    const char * testfile = boost::archive::tmpnam(NULL);
    BOOST_REQUIRE(NULL != testfile);

    {
        if (sizeof(small_object) <= sizeof(void*))
            std::cout << "object is small\n";
        else
            std::cout << "object is large\n";

        small_object const f(17);

        basic_any<test_iarchive, test_oarchive> any(f);

        out(testfile, any);
        in(testfile, any);
    }

    {
        if (sizeof(big_object) <= sizeof(void*))
            std::cout << "object is small\n";
        else
            std::cout << "object is large\n";

        big_object const f(5, 12);

        basic_any<test_iarchive, test_oarchive> any(f);

        out(testfile, any);
        in(testfile, any);
    }

    std::remove(testfile);

    finalize();

    return 0;
}

int
test_main( int argc, char* argv[] )
{

    // Configure application-specific options
    options_description cmdline("Usage: " HPX_APPLICATION_STRING " [options]");

    // Initialize and run HPX
    init(cmdline, argc, argv);

    return EXIT_SUCCESS;
}
// EOF
