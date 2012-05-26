//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/iostreams.hpp>

void raise_exception()
{
    HPX_THROW_EXCEPTION(hpx::no_success, "raise_exception", "simulated error");
}
HPX_PLAIN_ACTION(raise_exception, raise_exception_type);

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    raise_exception_type do_it;
    {
        ///////////////////////////////////////////////////////////////////////
        // Error reporting using exceptions
        try {
            // invoke raise_exception() which throws an exception
            do_it(hpx::find_here());
        }
        catch (hpx::exception const& e) {
            // Print just the essential error information.
            hpx::cout << "caught exception: " << e.what() << "\n\n";

            // Print all of the available diagnostic information as stored with
            // the exception.
            hpx::cout << "diagnostic information:"
                << hpx::diagnostic_information(e) << "\n";

            // Print the elements of the diagnostic information separately
            hpx::cout << "[locality-id]: " << hpx::get_locality_id(e) << "\n";
            hpx::cout << "[hostname]: "    << hpx::get_host_name(e) << "\n";
            hpx::cout << "[pid]: "         << hpx::get_process_id(e) << "\n";
            hpx::cout << "[function]: "    << hpx::get_function_name(e) << "\n";
            hpx::cout << "[file]: "        << hpx::get_file_name(e) << "\n";
            hpx::cout << "[line]: "        << hpx::get_line_number(e) << "\n";
            hpx::cout << "[os-thread]: "   << hpx::get_os_thread(e) << "\n";
            hpx::cout << "[thread-id]: "   << std::hex << hpx::get_thread_id(e) << "\n";
            hpx::cout << "[thread-description]: "
                << hpx::get_thread_description(e) << "\n\n";

            hpx::cout << hpx::flush;
        }

        ///////////////////////////////////////////////////////////////////////
        // Error reporting using error code
        hpx::error_code ec;

        // If an instance of an error_code is passed as the last argument while
        // invoking the action, the function will not throw in case of an error
        // but store some limited error information in this error_code instance.
        do_it(hpx::find_here(), ec);

        // Print just the essential error information. There is currently no
        // way to extract the diagnostic information for the exception causing
        // this error (see ticket #412).
        hpx::cout << "returned error: " << ec.get_message() << "\n";

        hpx::cout << hpx::flush;
    }

    // Initiate shutdown of the runtime system.
    return hpx::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    return hpx::init(argc, argv);       // Initialize and run HPX.
}
