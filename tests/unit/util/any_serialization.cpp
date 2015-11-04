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
#include <hpx/util/lightweight_test.hpp>
#include <hpx/runtime/serialization/serialize.hpp>

#include "small_big_object.hpp"

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
void out(std::vector<char> & out_buffer, A & a)
{
    hpx::serialization::output_archive archive(out_buffer);
    archive << a;
}

template <typename A>
void in(std::vector<char> & in_buffer, A & a)
{
    hpx::serialization::input_archive archive(in_buffer, in_buffer.size());
    archive >> a;
}

int hpx_main(variables_map& vm)
{
    typedef
        basic_any<hpx::serialization::input_archive, hpx::serialization::output_archive>
        any_type;

    {
        std::vector<char> buffer;

        small_object const f(17);
        HPX_TEST_LTE(sizeof(small_object), sizeof(void*));

        any_type any(f);

        out(buffer, any);
        any_type any_in;
        in(buffer, any_in);
        HPX_TEST(!any_in.empty());
        HPX_TEST_EQ(any, any_in);
    }

    {
        std::vector<char> buffer;

        big_object const f(5, 12);
        HPX_TEST_LT(sizeof(void*), sizeof(big_object));

        any_type any(f);

        out(buffer, any);
        any_type any_in;
        in(buffer, any_in);
        HPX_TEST(!any_in.empty());
        HPX_TEST_EQ(any, any_in);
    }

    return finalize();
}

int main( int argc, char* argv[] )
{

    // Configure application-specific options
    options_description cmdline("Usage: " HPX_APPLICATION_STRING " [options]");

    // Initialize and run HPX
    init(cmdline, argc, argv);

    return EXIT_SUCCESS;
}
// EOF
