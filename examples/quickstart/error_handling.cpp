//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  This example is documented in the Manual under the title "Error Handling"
#include <hpx/config.hpp>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <hpx/iostream.hpp>
#include <hpx/modules/runtime_local.hpp>

//[error_handling_raise_exception
void raise_exception()
{
    HPX_THROW_EXCEPTION(hpx::no_success, "raise_exception", "simulated error");
}
HPX_PLAIN_ACTION(raise_exception, raise_exception_action);
//]

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    {
        ///////////////////////////////////////////////////////////////////////
        // Error reporting using exceptions
        //[exception_diagnostic_information
        hpx::cout << "Error reporting using exceptions\n";
        try {
            // invoke raise_exception() which throws an exception
            raise_exception_action do_it;
            do_it(hpx::find_here());
        }
        catch (hpx::exception const& e) {
            // Print just the essential error information.
            hpx::cout << "caught exception: " << e.what() << "\n\n";

            // Print all of the available diagnostic information as stored with
            // the exception.
            hpx::cout << "diagnostic information:"
                << hpx::diagnostic_information(e) << "\n";
        }
        hpx::cout << hpx::flush;
        //]

        // Detailed error reporting using exceptions
        //[exception_diagnostic_elements
        hpx::cout << "Detailed error reporting using exceptions\n";
        try {
            // Invoke raise_exception() which throws an exception.
            raise_exception_action do_it;
            do_it(hpx::find_here());
        }
        catch (hpx::exception const& e) {
            // Print the elements of the diagnostic information separately.
            hpx::cout << "{what}: "        << hpx::get_error_what(e) << "\n";
            hpx::cout << "{locality-id}: " << hpx::get_error_locality_id(e) << "\n";
            hpx::cout << "{hostname}: "    << hpx::get_error_host_name(e) << "\n";
            hpx::cout << "{pid}: "         << hpx::get_error_process_id(e) << "\n";
            hpx::cout << "{function}: "    << hpx::get_error_function_name(e) << "\n";
            hpx::cout << "{file}: "        << hpx::get_error_file_name(e) << "\n";
            hpx::cout << "{line}: "        << hpx::get_error_line_number(e) << "\n";
            hpx::cout << "{os-thread}: "   << hpx::get_error_os_thread(e) << "\n";
            hpx::cout << "{thread-id}: "   << std::hex << hpx::get_error_thread_id(e)
                << "\n";
            hpx::cout << "{thread-description}: "
                << hpx::get_error_thread_description(e) << "\n";
            hpx::cout << "{state}: "       << std::hex << hpx::get_error_state(e)
                << "\n";
            hpx::cout << "{stack-trace}: " << hpx::get_error_backtrace(e) << "\n";
            hpx::cout << "{env}: "         << hpx::get_error_env(e) << "\n";
        }
        hpx::cout << hpx::flush;
        //]

        ///////////////////////////////////////////////////////////////////////
        // Error reporting using error code
        {
            //[error_handling_diagnostic_information
            hpx::cout << "Error reporting using error code\n";

            // Create a new error_code instance.
            hpx::error_code ec;

            // If an instance of an error_code is passed as the last argument while
            // invoking the action, the function will not throw in case of an error
            // but store the error information in this error_code instance instead.
            raise_exception_action do_it;
            do_it(hpx::find_here(), ec);

            if (ec) {
                // Print just the essential error information.
                hpx::cout << "returned error: " << ec.get_message() << "\n";

                // Print all of the available diagnostic information as stored with
                // the exception.
                hpx::cout << "diagnostic information:"
                    << hpx::diagnostic_information(ec) << "\n";
            }

            hpx::cout << hpx::flush;
            //]
        }

        // Detailed error reporting using error code
        {
            //[error_handling_diagnostic_elements
            hpx::cout << "Detailed error reporting using error code\n";

            // Create a new error_code instance.
            hpx::error_code ec;

            // If an instance of an error_code is passed as the last argument while
            // invoking the action, the function will not throw in case of an error
            // but store the error information in this error_code instance instead.
            raise_exception_action do_it;
            do_it(hpx::find_here(), ec);

            if (ec) {
                // Print the elements of the diagnostic information separately.
                hpx::cout << "{what}: "        << hpx::get_error_what(ec) << "\n";
                hpx::cout << "{locality-id}: " << hpx::get_error_locality_id(ec) << "\n";
                hpx::cout << "{hostname}: "    << hpx::get_error_host_name(ec) << "\n";
                hpx::cout << "{pid}: "         << hpx::get_error_process_id(ec) << "\n";
                hpx::cout << "{function}: "    << hpx::get_error_function_name(ec)
                    << "\n";
                hpx::cout << "{file}: "        << hpx::get_error_file_name(ec) << "\n";
                hpx::cout << "{line}: "        << hpx::get_error_line_number(ec) << "\n";
                hpx::cout << "{os-thread}: "   << hpx::get_error_os_thread(ec) << "\n";
                hpx::cout << "{thread-id}: "   << std::hex
                    << hpx::get_error_thread_id(ec) << "\n";
                hpx::cout << "{thread-description}: "
                    << hpx::get_error_thread_description(ec) << "\n\n";
                hpx::cout << "{state}: "       << std::hex << hpx::get_error_state(ec)
                    << "\n";
                hpx::cout << "{stack-trace}: " << hpx::get_error_backtrace(ec) << "\n";
                hpx::cout << "{env}: "         << hpx::get_error_env(ec) << "\n";
            }

            hpx::cout << hpx::flush;
            //]
        }

        // Error reporting using lightweight error code
        {
            //[lightweight_error_handling_diagnostic_information
            hpx::cout << "Error reporting using an lightweight error code\n";

            // Create a new error_code instance.
            hpx::error_code ec(hpx::lightweight);

            // If an instance of an error_code is passed as the last argument while
            // invoking the action, the function will not throw in case of an error
            // but store the error information in this error_code instance instead.
            raise_exception_action do_it;
            do_it(hpx::find_here(), ec);

            if (ec) {
                // Print just the essential error information.
                hpx::cout << "returned error: " << ec.get_message() << "\n";

                // Print all of the available diagnostic information as stored with
                // the exception.
                hpx::cout << "error code:" << ec.value() << "\n";
            }

            hpx::cout << hpx::flush;
            //]
        }
    }

    // Initiate shutdown of the runtime system.
    return hpx::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    return hpx::init(argc, argv);       // Initialize and run HPX.
}
