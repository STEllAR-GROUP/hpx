////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <string>
#include <vector>

#include <boost/iostreams/stream.hpp>
#include <boost/archive/basic_binary_iarchive.hpp>
#include <boost/archive/basic_binary_oarchive.hpp>

#include <hpx/exception.hpp>
#include <hpx/util/container_device.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/util/serialize_exception.hpp>

int main()
{
    using boost::exception_ptr;
    using boost::current_exception;
    using boost::current_exception_cast;
    using boost::throw_exception;
    using boost::rethrow_exception;
    using boost::system::error_code;

    using hpx::error;
    using hpx::rethrow;
    using hpx::make_error_code;
    using hpx::success;
    using hpx::repeated_request;
    using hpx::exception;

    typedef std::vector<char> buffer_type;
    typedef hpx::util::container_device<buffer_type> device_type;
    typedef boost::iostreams::stream<device_type> stream_type;
    typedef exception data_type;

    #if HPX_USE_PORTABLE_ARCHIVES != 0
        typedef hpx::util::portable_binary_oarchive oarchive_type;
        typedef hpx::util::portable_binary_iarchive iarchive_type;
    #else
        typedef boost::archive::binary_oarchive oarchive_type;
        typedef boost::archive::binary_iarchive iarchive_type;
    #endif

    buffer_type buffer; 
    stream_type stream(buffer);
    
    error_code err = make_error_code(repeated_request);
    data_type data(err); 
        
    HPX_SANITY_EQ(data.code(), err);

    { // save 
        oarchive_type archive(stream);
        
        try { throw_exception(data); }

        catch (...)
        {
            exception_ptr ptr = current_exception();
            archive & ptr;
        }

        HPX_SANITY_EQ(data.code(), err);
    }

    { // load
        stream_type stream(buffer);
        iarchive_type archive(stream);
        
        exception_ptr ptr;
        archive & ptr;

        try { rethrow_exception(ptr); }

        catch (...)
        {        
            data_type const* loaded = current_exception_cast<data_type const>();

            HPX_TEST(loaded);
            HPX_TEST_EQ(make_error_code
                (static_cast<error>(loaded->code().value())), err);
            HPX_TEST_EQ(data.code(), make_error_code
                (static_cast<error>(loaded->code().value())));
        }
    }

    return hpx::util::report_errors();
}

