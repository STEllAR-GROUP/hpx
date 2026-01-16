//  Copyright (c) 2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// (C) Copyright 2008 CodeRage, LLC (turkanis at coderage dot com)
// (C) Copyright 2004-2007 Jonathan Turkanis

// See http://www.boost.org/libs/iostreams for documentation.

#include <hpx/hpx_main.hpp>
#include <hpx/modules/iostream.hpp>
#include <hpx/modules/testing.hpp>

#include <fcntl.h>
#include <fstream>
#include <string>

#include "detail/file_handle.hpp"
#include "detail/temp_file.hpp"
#include "detail/verification.hpp"

using namespace hpx::iostream;
using namespace hpx::iostream::test;
using std::ifstream;

void file_descriptor_test()
{
    typedef stream<file_descriptor_source> fdistream;
    typedef stream<file_descriptor_sink> fdostream;
    typedef stream<file_descriptor> fdstream;

    test_file test1;
    test_file test2;

    //--------------Test file_descriptor_source-------------------------------//
    {
        fdistream first(file_descriptor_source(test1.name()), 0);
        ifstream second(test2.name().c_str());
        HPX_TEST(first->is_open());
        HPX_TEST_MSG(compare_streams_in_chars(first, second),
            "failed reading from file_descriptor_source in chars with no "
            "buffer");
        first->close();
        HPX_TEST(!first->is_open());
    }

    {
        fdistream first(file_descriptor_source(test1.name()), 0);
        ifstream second(test2.name().c_str());
        HPX_TEST(first->is_open());
        HPX_TEST_MSG(compare_streams_in_chunks(first, second),
            "failed reading from file_descriptor_source in chunks with no "
            "buffer");
        first->close();
        HPX_TEST(!first->is_open());
    }

    {
        file_descriptor_source file(test1.name());
        fdistream first(file);
        ifstream second(test2.name().c_str());
        HPX_TEST(first->is_open());
        HPX_TEST_MSG(compare_streams_in_chars(first, second),
            "failed reading from file_descriptor_source in chars with buffer");
        first->close();
        HPX_TEST(!first->is_open());
    }

    {
        file_descriptor_source file(test1.name());
        fdistream first(file);
        ifstream second(test2.name().c_str());
        HPX_TEST(first->is_open());
        HPX_TEST_MSG(compare_streams_in_chunks(first, second),
            "failed reading from file_descriptor_source in chunks with buffer");
        first->close();
        HPX_TEST(!first->is_open());
    }

    // test illegal flag combinations
    {
        HPX_TEST_THROW(
            file_descriptor_source(test1.name(), std::ios_base::trunc),
            std::ios_base::failure);
        HPX_TEST_THROW(file_descriptor_source(test1.name(),
                           std::ios_base::app | std::ios_base::trunc),
            std::ios_base::failure);
        HPX_TEST_THROW(file_descriptor_source(test1.name(), std::ios_base::out),
            std::ios_base::failure);
        HPX_TEST_THROW(file_descriptor_source(test1.name(),
                           std::ios_base::out | std::ios_base::app),
            std::ios_base::failure);
        HPX_TEST_THROW(file_descriptor_source(test1.name(),
                           std::ios_base::out | std::ios_base::trunc),
            std::ios_base::failure);
        HPX_TEST_THROW(
            file_descriptor_source(test1.name(),
                std::ios_base::out | std::ios_base::app | std::ios_base::trunc),
            std::ios_base::failure);
    }

    //--------------Test file_descriptor_sink---------------------------------//
    {
        temp_file temp;
        file_descriptor_sink file(temp.name(), std::ios_base::trunc);
        fdostream out(file, 0);
        HPX_TEST(out->is_open());
        write_data_in_chars(out);
        out.close();
        HPX_TEST_MSG(compare_files(test1.name(), temp.name()),
            "failed writing to file_descriptor_sink in chars with no buffer");
        file.close();
        HPX_TEST(!file.is_open());
    }

    {
        temp_file temp;
        file_descriptor_sink file(temp.name(), std::ios_base::trunc);
        fdostream out(file, 0);
        HPX_TEST(out->is_open());
        write_data_in_chunks(out);
        out.close();
        HPX_TEST_MSG(compare_files(test1.name(), temp.name()),
            "failed writing to file_descriptor_sink in chunks with no buffer");
        file.close();
        HPX_TEST(!file.is_open());
    }

    {
        temp_file temp;
        file_descriptor_sink file(temp.name(), std::ios_base::trunc);
        fdostream out(file);
        HPX_TEST(out->is_open());
        write_data_in_chars(out);
        out.close();
        HPX_TEST_MSG(compare_files(test1.name(), temp.name()),
            "failed writing to file_descriptor_sink in chars with buffer");
        file.close();
        HPX_TEST(!file.is_open());
    }

    {
        temp_file temp;
        file_descriptor_sink file(temp.name(), std::ios_base::trunc);
        fdostream out(file);
        HPX_TEST(out->is_open());
        write_data_in_chunks(out);
        out.close();
        HPX_TEST_MSG(compare_files(test1.name(), temp.name()),
            "failed writing to file_descriptor_sink in chunks with buffer");
        file.close();
        HPX_TEST(!file.is_open());
    }

    {
        temp_file temp;

        // set up the tests
        {
            file_descriptor_sink file(temp.name(), std::ios_base::trunc);
            fdostream out(file);
            write_data_in_chunks(out);
            out.close();
            file.close();
        }
        // test std::ios_base::app
        {
            file_descriptor_sink file(temp.name(), std::ios_base::app);
            fdostream out(file);
            HPX_TEST(out->is_open());
            write_data_in_chars(out);
            out.close();
            std::string expected(narrow_data());
            expected += narrow_data();
            HPX_TEST_MSG(compare_container_and_file(expected, temp.name()),
                "failed writing to file_descriptor_sink in append mode");
            file.close();
            HPX_TEST(!file.is_open());
        }

        // test std::ios_base::trunc
        {
            file_descriptor_sink file(temp.name(), std::ios_base::trunc);
            fdostream out(file);
            HPX_TEST(out->is_open());
            write_data_in_chars(out);
            out.close();
            HPX_TEST_MSG(compare_files(test1.name(), temp.name()),
                "failed writing to file_descriptor_sink in trunc mode");
            file.close();
            HPX_TEST(!file.is_open());
        }

        // test illegal flag combinations
        {
            HPX_TEST_THROW(file_descriptor_sink(temp.name(),
                               std::ios_base::trunc | std::ios_base::app),
                std::ios_base::failure);
            HPX_TEST_THROW(file_descriptor_sink(temp.name(), std::ios_base::in),
                std::ios_base::failure);
            HPX_TEST_THROW(file_descriptor_sink(temp.name(),
                               std::ios_base::in | std::ios_base::app),
                std::ios_base::failure);
            HPX_TEST_THROW(file_descriptor_sink(temp.name(),
                               std::ios_base::in | std::ios_base::trunc),
                std::ios_base::failure);
            HPX_TEST_THROW(file_descriptor_sink(temp.name(),
                               std::ios_base::in | std::ios_base::trunc |
                                   std::ios_base::app),
                std::ios_base::failure);
        }
    }

    //--Test seeking with file_descriptor_source and file_descriptor_sink-----//
    test_file test3;
    {
        file_descriptor_sink sink(test3.name());
        fdostream out(sink);
        HPX_TEST(out->is_open());
        HPX_TEST_MSG(test_output_seekable(out),
            "failed seeking within a file_descriptor_sink");
        out->close();
        HPX_TEST(!out->is_open());

        file_descriptor_source source(test3.name());
        fdistream in(source);
        HPX_TEST(in->is_open());
        HPX_TEST_MSG(test_input_seekable(in),
            "failed seeking within a file_descriptor_source");
        in->close();
        HPX_TEST(!in->is_open());
    }

    //--------------Test file_descriptor--------------------------------------//
    {
        temp_file temp;
        file_descriptor file(temp.name(),
            std::ios_base::in | std::ios_base::out | std::ios_base::trunc |
                std::ios_base::binary);
        fdstream io(file, BUFSIZ);
        HPX_TEST_MSG(test_seekable_in_chars(io),
            "failed seeking within a file_descriptor, in chars");
    }

    {
        temp_file temp;
        file_descriptor file(temp.name(),
            std::ios_base::in | std::ios_base::out | std::ios_base::trunc |
                std::ios_base::binary);
        fdstream io(file, BUFSIZ);
        HPX_TEST_MSG(test_seekable_in_chunks(io),
            "failed seeking within a file_descriptor, in chunks");
    }

    //--------------Test read-only file_descriptor----------------------------//
    {
        fdstream first(file_descriptor(test1.name(), std::ios_base::in), 0);
        ifstream second(test2.name().c_str());
        HPX_TEST(first->is_open());
        write_data_in_chars(first);
        HPX_TEST(first.fail());
        first.clear();
        HPX_TEST_MSG(compare_streams_in_chars(first, second),
            "failed reading from file_descriptor in chars with no buffer");
        first->close();
        HPX_TEST(!first->is_open());
    }

    {
        fdstream first(file_descriptor(test1.name(), std::ios_base::in), 0);
        ifstream second(test2.name().c_str());
        HPX_TEST(first->is_open());
        write_data_in_chunks(first);
        HPX_TEST(first.fail());
        first.clear();
        HPX_TEST_MSG(compare_streams_in_chunks(first, second),
            "failed reading from file_descriptor in chunks with no buffer");
        first->close();
        HPX_TEST(!first->is_open());
    }

    {
        file_descriptor file(test1.name(), std::ios_base::in);
        fdstream first(file);
        ifstream second(test2.name().c_str());
        HPX_TEST(first->is_open());
        write_data_in_chars(first);
        HPX_TEST(first.fail());
        first.clear();
        first.seekg(0, std::ios_base::beg);
        HPX_TEST_MSG(compare_streams_in_chars(first, second),
            "failed reading from file_descriptor in chars with buffer");
        first->close();
        HPX_TEST(!first->is_open());
    }

    {
        file_descriptor file(test1.name(), std::ios_base::in);
        fdstream first(file);
        ifstream second(test2.name().c_str());
        HPX_TEST(first->is_open());
        write_data_in_chunks(first);
        HPX_TEST(first.fail());
        first.clear();
        first.seekg(0, std::ios_base::beg);
        HPX_TEST_MSG(compare_streams_in_chunks(first, second),
            "failed reading from file_descriptor in chunks with buffer");
        first->close();
        HPX_TEST(!first->is_open());
    }

    //--------------Test write-only file_descriptor---------------------------//
    {
        temp_file temp;
        file_descriptor file(
            temp.name(), std::ios_base::out | std::ios_base::trunc);
        fdstream out(file, 0);
        HPX_TEST(out->is_open());
        out.get();
        HPX_TEST(out.fail());
        out.clear();
        write_data_in_chars(out);
        out.seekg(0, std::ios_base::beg);
        out.get();
        HPX_TEST(out.fail());
        out.clear();
        out.close();
        HPX_TEST_MSG(compare_files(test1.name(), temp.name()),
            "failed writing to file_descriptor in chars with no buffer");
        file.close();
        HPX_TEST(!file.is_open());
    }

    {
        temp_file temp;
        file_descriptor file(
            temp.name(), std::ios_base::out | std::ios_base::trunc);
        fdstream out(file, 0);
        HPX_TEST(out->is_open());
        out.get();
        HPX_TEST(out.fail());
        out.clear();
        write_data_in_chunks(out);
        out.seekg(0, std::ios_base::beg);
        out.get();
        HPX_TEST(out.fail());
        out.clear();
        out.close();
        HPX_TEST_MSG(compare_files(test1.name(), temp.name()),
            "failed writing to file_descriptor_sink in chunks with no buffer");
        file.close();
        HPX_TEST(!file.is_open());
    }

    {
        temp_file temp;
        file_descriptor file(
            temp.name(), std::ios_base::out | std::ios_base::trunc);
        fdstream out(file);
        HPX_TEST(out->is_open());
        out.get();
        HPX_TEST(out.fail());
        out.clear();
        write_data_in_chars(out);
        out.seekg(0, std::ios_base::beg);
        out.get();
        HPX_TEST(out.fail());
        out.clear();
        out.close();
        HPX_TEST_MSG(compare_files(test1.name(), temp.name()),
            "failed writing to file_descriptor_sink in chars with buffer");
        file.close();
        HPX_TEST(!file.is_open());
    }

    {
        temp_file temp;
        file_descriptor file(
            temp.name(), std::ios_base::out | std::ios_base::trunc);
        fdstream out(file);
        HPX_TEST(out->is_open());
        out.get();
        HPX_TEST(out.fail());
        out.clear();
        write_data_in_chunks(out);
        out.seekg(0, std::ios_base::beg);
        out.get();
        HPX_TEST(out.fail());
        out.clear();
        out.close();
        HPX_TEST_MSG(compare_files(test1.name(), temp.name()),
            "failed writing to file_descriptor_sink in chunks with buffer");
        file.close();
        HPX_TEST(!file.is_open());
    }

    // test illegal flag combinations
    {
        HPX_TEST_THROW(
            file_descriptor(test1.name(), std::ios_base::openmode(0)),
            std::ios_base::failure);
        HPX_TEST_THROW(file_descriptor(test1.name(), std::ios_base::trunc),
            std::ios_base::failure);
        HPX_TEST_THROW(file_descriptor(test1.name(),
                           std::ios_base::app | std::ios_base::trunc),
            std::ios_base::failure);
        HPX_TEST_THROW(file_descriptor(test1.name(),
                           std::ios_base::in | std::ios_base::trunc),
            std::ios_base::failure);
        HPX_TEST_THROW(
            file_descriptor(test1.name(),
                std::ios_base::in | std::ios_base::app | std::ios_base::trunc),
            std::ios_base::failure);
        HPX_TEST_THROW(
            file_descriptor(test1.name(),
                std::ios_base::out | std::ios_base::app | std::ios_base::trunc),
            std::ios_base::failure);
        HPX_TEST_THROW(file_descriptor(test1.name(),
                           std::ios_base::in | std::ios_base::out |
                               std::ios_base::app | std::ios_base::trunc),
            std::ios_base::failure);
    }
}

template <class FileDescriptor>
void file_handle_test_impl(FileDescriptor*)
{
    test_file test1;
    test_file test2;

    {
        hpx::iostream::file_handle handle = open_file_handle(test1.name());
        {
            FileDescriptor device1(
                handle, file_descriptor::flags::never_close_handle);
            HPX_TEST(device1.handle() == handle);
        }
        check_handle_open(handle);
        close_file_handle(handle);
    }

    {
        hpx::iostream::file_handle handle = open_file_handle(test1.name());
        {
            FileDescriptor device1(
                handle, file_descriptor::flags::close_handle);
            HPX_TEST(device1.handle() == handle);
        }
        check_handle_closed(handle);
    }

    {
        hpx::iostream::file_handle handle = open_file_handle(test1.name());
        FileDescriptor device1(
            handle, file_descriptor::flags::never_close_handle);
        HPX_TEST(device1.handle() == handle);
        device1.close();
        HPX_TEST(!device1.is_open());
        check_handle_open(handle);
        close_file_handle(handle);
    }

    {
        hpx::iostream::file_handle handle = open_file_handle(test1.name());
        FileDescriptor device1(handle, file_descriptor::flags::close_handle);
        HPX_TEST(device1.handle() == handle);
        device1.close();
        HPX_TEST(!device1.is_open());
        check_handle_closed(handle);
    }

    {
        hpx::iostream::file_handle handle1 = open_file_handle(test1.name());
        hpx::iostream::file_handle handle2 = open_file_handle(test2.name());
        {
            FileDescriptor device1(
                handle1, file_descriptor::flags::never_close_handle);
            HPX_TEST(device1.handle() == handle1);
            device1.open(handle2, file_descriptor::flags::never_close_handle);
            HPX_TEST(device1.handle() == handle2);
        }
        check_handle_open(handle1);
        check_handle_open(handle2);
        close_file_handle(handle1);
        close_file_handle(handle2);
    }

    {
        hpx::iostream::file_handle handle1 = open_file_handle(test1.name());
        hpx::iostream::file_handle handle2 = open_file_handle(test2.name());
        {
            FileDescriptor device1(
                handle1, file_descriptor::flags::close_handle);
            HPX_TEST(device1.handle() == handle1);
            device1.open(handle2, file_descriptor::flags::close_handle);
            HPX_TEST(device1.handle() == handle2);
            check_handle_closed(handle1);
            check_handle_open(handle2);
        }
        check_handle_closed(handle1);
        check_handle_closed(handle2);
    }

    {
        hpx::iostream::file_handle handle1 = open_file_handle(test1.name());
        hpx::iostream::file_handle handle2 = open_file_handle(test2.name());
        {
            FileDescriptor device1(
                handle1, file_descriptor::flags::close_handle);
            HPX_TEST(device1.handle() == handle1);
            device1.open(handle2, file_descriptor::flags::never_close_handle);
            HPX_TEST(device1.handle() == handle2);
            check_handle_closed(handle1);
            check_handle_open(handle2);
        }
        check_handle_closed(handle1);
        check_handle_open(handle2);
        close_file_handle(handle2);
    }

    {
        hpx::iostream::file_handle handle = open_file_handle(test1.name());
        {
            FileDescriptor device1;
            HPX_TEST(!device1.is_open());
            device1.open(handle, file_descriptor::flags::never_close_handle);
            HPX_TEST(device1.handle() == handle);
            check_handle_open(handle);
        }
        check_handle_open(handle);
        close_file_handle(handle);
    }

    {
        hpx::iostream::file_handle handle = open_file_handle(test1.name());
        {
            FileDescriptor device1;
            HPX_TEST(!device1.is_open());
            device1.open(handle, file_descriptor::flags::close_handle);
            HPX_TEST(device1.handle() == handle);
            check_handle_open(handle);
        }
        check_handle_closed(handle);
    }
}

void file_handle_test()
{
    file_handle_test_impl((file_descriptor*) nullptr);
    file_handle_test_impl((file_descriptor_source*) nullptr);
    file_handle_test_impl((file_descriptor_sink*) nullptr);
}

int main(int, char*[])
{
    file_descriptor_test();
    file_handle_test();
    return hpx::util::report_errors();
}
