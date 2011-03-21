////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <vector>

#include <boost/iostreams/stream.hpp>
#include <boost/archive/basic_binary_iarchive.hpp>
#include <boost/archive/basic_binary_oarchive.hpp>

#include <hpx/util/container_device.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/util/serialize_exception.hpp>

typedef std::vector<char> buffer_type;
typedef hpx::util::container_device<buffer_type> device_type;
typedef boost::iostreams::stream<device_type> stream_type;

#if HPX_USE_PORTABLE_ARCHIVES != 0
    typedef hpx::util::portable_binary_oarchive oarchive_type;
    typedef hpx::util::portable_binary_iarchive iarchive_type;
#else
    typedef boost::archive::binary_oarchive oarchive_type;
    typedef boost::archive::binary_iarchive iarchive_type;
#endif

using boost::exception_ptr;
using boost::current_exception;
using boost::current_exception_cast;
using boost::throw_exception;
using boost::rethrow_exception;

template <typename T>
void test_std_exception()
{ 
    buffer_type buffer; 
    T data("foo");
    
    HPX_SANITY_EQ(std::string(data.what()), std::string("foo"));

    { // save 
        stream_type stream(buffer);
        oarchive_type archive(stream);

        try { throw_exception(data); }

        catch (...)
        {
            exception_ptr ptr = current_exception();
            archive & ptr;
        }
    
        HPX_SANITY_EQ(std::string(data.what()), std::string("foo"));
    }

    { // load
        stream_type stream(buffer);
        iarchive_type archive(stream);
        
        exception_ptr ptr;
        archive & ptr;

        try { rethrow_exception(ptr); }

        catch (...)
        {        
            T const* loaded = current_exception_cast<T const>();

            HPX_TEST(loaded);
            HPX_TEST_EQ(std::string(loaded->what()), std::string("foo"));
            HPX_TEST_EQ(std::string(data.what()), std::string(loaded->what()));
        }
    }
}

int main()
{
    test_std_exception<std::logic_error>();
    test_std_exception<std::invalid_argument>();
    test_std_exception<std::out_of_range>();
    test_std_exception<std::runtime_error>();

    return hpx::util::report_errors();
}

