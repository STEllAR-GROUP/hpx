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

#include <fstream>

#include "detail/closable.hpp"
#include "detail/operation_sequence.hpp"
#include "detail/temp_file.hpp"
#include "detail/verification.hpp"

using namespace std;
using namespace hpx::iostream;
using namespace hpx::iostream::test;

void read_write_test()
{
    {
        test_file src1, src2;
        temp_file dest;
        filtering_istream first, second;
        first.push(tee(file_sink(dest.name(), out_mode)));
        first.push(file_source(src1.name(), in_mode));
        second.push(file_source(src2.name(), in_mode));
        compare_streams_in_chars(first, second);    // ignore return value
        first.reset();
        HPX_TEST_MSG(compare_files(dest.name(), src1.name()),
            "failed reading from a tee_filter in chars");
    }

    {
        test_file src1, src2;
        temp_file dest;
        filtering_istream first, second;
        first.push(tee(file_sink(dest.name(), out_mode)));
        first.push(file_source(src1.name(), in_mode));
        second.push(file_source(src2.name(), in_mode));
        compare_streams_in_chunks(first, second);    // ignore return value
        first.reset();
        HPX_TEST_MSG(compare_files(dest.name(), src1.name()),
            "failed reading from a tee_filter in chunks");
    }

    {
        temp_file dest1;
        temp_file dest2;
        filtering_ostream out;
        out.push(tee(file_sink(dest1.name(), out_mode)));
        out.push(file_sink(dest2.name(), out_mode));
        write_data_in_chars(out);
        out.reset();
        HPX_TEST_MSG(compare_files(dest1.name(), dest2.name()),
            "failed writing to a tee_filter in chars");
    }

    {
        temp_file dest1;
        temp_file dest2;
        filtering_ostream out;
        out.push(tee(file_sink(dest1.name(), out_mode)));
        out.push(file_sink(dest2.name(), out_mode));
        write_data_in_chunks(out);
        out.reset();
        HPX_TEST_MSG(compare_files(dest1.name(), dest2.name()),
            "failed writing to a tee_filter in chunks");
    }

    {
        test_file src1, src2;
        temp_file dest;
        filtering_istream first, second;
        first.push(tee(file_source(src1.name(), in_mode),
            file_sink(dest.name(), out_mode)));
        second.push(file_source(src2.name(), in_mode));
        compare_streams_in_chars(first, second);    // ignore return value
        first.reset();
        HPX_TEST_MSG(compare_files(dest.name(), src1.name()),
            "failed reading from a tee_device in chars");
    }

    {
        test_file src1, src2;
        temp_file dest;
        filtering_istream first, second;
        first.push(tee(file_source(src1.name(), in_mode),
            file_sink(dest.name(), out_mode)));
        second.push(file_source(src2.name(), in_mode));
        compare_streams_in_chunks(first, second);    // ignore return value
        first.reset();
        HPX_TEST_MSG(compare_files(dest.name(), src1.name()),
            "failed reading from a tee_device in chunks");
    }

    {
        temp_file dest1;
        temp_file dest2;
        filtering_ostream out;
        out.push(tee(file_sink(dest1.name(), out_mode),
            file_sink(dest2.name(), out_mode)));
        write_data_in_chars(out);
        out.reset();
        HPX_TEST_MSG(compare_files(dest1.name(), dest2.name()),
            "failed writing to a tee_device in chars");
    }

    {
        temp_file dest1;
        temp_file dest2;
        filtering_ostream out;
        out.push(tee(file_sink(dest1.name(), out_mode),
            file_sink(dest2.name(), out_mode)));
        write_data_in_chunks(out);
        out.reset();
        HPX_TEST_MSG(compare_files(dest1.name(), dest2.name()),
            "failed writing to a tee_device in chunks");
    }
}

void close_test()
{
    // Note: The implementation of tee_device closes the first
    // sink before the second

    // Tee two sinks (Borland <= 5.8.2 needs a little help compiling this case,
    // but it executes the closing algorithm correctly)
    {
        operation_sequence seq;
        chain<output> ch;
        ch.push(
            hpx::iostream::tee(closable_device<output>(seq.new_operation(1)),
                closable_device<output>(seq.new_operation(2))));
        HPX_TEST_NO_THROW(ch.reset());
        HPX_TEST_MSG(seq.is_success(), seq.message());
    }

    // Tee two bidirectional devices
    {
        operation_sequence seq;
        chain<output> ch;
        ch.push(
            hpx::iostream::tee(closable_device<bidirectional>(
                                   seq.new_operation(1), seq.new_operation(2)),
                closable_device<bidirectional>(
                    seq.new_operation(3), seq.new_operation(4))));
        HPX_TEST_NO_THROW(ch.reset());
        HPX_TEST_MSG(seq.is_success(), seq.message());
    }

    // Tee two seekable devices
    {
        operation_sequence seq;
        chain<output> ch;
        ch.push(
            hpx::iostream::tee(closable_device<seekable>(seq.new_operation(1)),
                closable_device<seekable>(seq.new_operation(2))));
        HPX_TEST_NO_THROW(ch.reset());
        HPX_TEST_MSG(seq.is_success(), seq.message());
    }

    // Tee a sink
    {
        operation_sequence seq;
        chain<output> ch;
        ch.push(
            hpx::iostream::tee(closable_device<output>(seq.new_operation(1))));
        ch.push(closable_device<output>(seq.new_operation(2)));
        HPX_TEST_NO_THROW(ch.reset());
        HPX_TEST_MSG(seq.is_success(), seq.message());
    }

    // Tee a bidirectional device
    {
        operation_sequence seq;
        chain<output> ch;
        ch.push(hpx::iostream::tee(closable_device<bidirectional>(
            seq.new_operation(1), seq.new_operation(2))));
        ch.push(closable_device<output>(seq.new_operation(3)));
        HPX_TEST_NO_THROW(ch.reset());
        HPX_TEST_MSG(seq.is_success(), seq.message());
    }

    // Tee a seekable device
    {
        operation_sequence seq;
        chain<output> ch;
        ch.push(hpx::iostream::tee(
            closable_device<seekable>(seq.new_operation(1))));
        ch.push(closable_device<seekable>(seq.new_operation(2)));
        HPX_TEST_NO_THROW(ch.reset());
        HPX_TEST_MSG(seq.is_success(), seq.message());
    }
}

void test_std_io()
{
    {
        temp_file dest1;
        temp_file dest2;
        filtering_ostream out;
        std::ofstream stream1(dest1.name().c_str(), out_mode);
        out.push(tee(stream1));
        out.push(file_sink(dest2.name(), out_mode));
        write_data_in_chunks(out);
        HPX_TEST_MSG(compare_files(dest1.name(), dest2.name()),
            "failed writing to a tee_device in chunks");
    }
    {
        temp_file dest1;
        temp_file dest2;
        filtering_ostream out;
        std::ofstream stream1(dest1.name().c_str(), out_mode);
        out.push(tee(stream1, file_sink(dest2.name(), out_mode)));
        write_data_in_chunks(out);
        HPX_TEST_MSG(compare_files(dest1.name(), dest2.name()),
            "failed writing to a tee_device in chunks");
    }
    {
        temp_file dest1;
        temp_file dest2;
        filtering_ostream out;
        std::ofstream stream2(dest2.name().c_str(), out_mode);
        out.push(tee(file_sink(dest1.name(), out_mode), stream2));
        write_data_in_chunks(out);
        HPX_TEST_MSG(compare_files(dest1.name(), dest2.name()),
            "failed writing to a tee_device in chunks");
    }
    {
        temp_file dest1;
        temp_file dest2;
        filtering_ostream out;
        std::ofstream stream1(dest1.name().c_str(), out_mode);
        std::ofstream stream2(dest2.name().c_str(), out_mode);
        out.push(tee(stream1, stream2));
        write_data_in_chunks(out);
        HPX_TEST_MSG(compare_files(dest1.name(), dest2.name()),
            "failed writing to a tee_device in chunks");
    }
}

void tee_composite_test()
{
    // This test is probably redundant, given the above test and the tests in
    // compose_test.cpp, but it verifies that ticket #1002 is fixed

    // Tee a composite sink with a sink
    {
        operation_sequence seq;
        chain<output> ch;
        ch.push(hpx::iostream::tee(
            hpx::iostream::compose(
                closable_filter<output>(seq.new_operation(1)),
                closable_device<output>(seq.new_operation(2))),
            closable_device<output>(seq.new_operation(3))));
        HPX_TEST_NO_THROW(ch.reset());
        HPX_TEST_MSG(seq.is_success(), seq.message());
    }

    // Tee a composite bidirectional device with a sink
    {
        operation_sequence seq;
        chain<output> ch;
        ch.push(hpx::iostream::tee(
            hpx::iostream::compose(
                closable_filter<bidirectional>(
                    seq.new_operation(2), seq.new_operation(3)),
                closable_device<bidirectional>(
                    seq.new_operation(1), seq.new_operation(4))),
            closable_device<output>(seq.new_operation(5))));
        HPX_TEST_NO_THROW(ch.reset());
        HPX_TEST_MSG(seq.is_success(), seq.message());
    }

    // Tee a composite composite seekable device with a sink
    {
        operation_sequence seq;
        chain<output> ch;
        ch.push(hpx::iostream::tee(
            hpx::iostream::compose(
                closable_filter<seekable>(seq.new_operation(1)),
                closable_device<seekable>(seq.new_operation(2))),
            closable_device<output>(seq.new_operation(3))));
        HPX_TEST_NO_THROW(ch.reset());
        HPX_TEST_MSG(seq.is_success(), seq.message());
    }

    // Tee a composite sink
    {
        operation_sequence seq;
        chain<output> ch;
        ch.push(hpx::iostream::tee(hpx::iostream::compose(
            closable_filter<output>(seq.new_operation(1)),
            closable_device<output>(seq.new_operation(2)))));
        ch.push(closable_device<output>(seq.new_operation(3)));
        HPX_TEST_NO_THROW(ch.reset());
        HPX_TEST_MSG(seq.is_success(), seq.message());
    }

    // Tee a composite bidirectional device with a sink
    {
        operation_sequence seq;
        chain<output> ch;
        ch.push(hpx::iostream::tee(hpx::iostream::compose(
            closable_filter<bidirectional>(
                seq.new_operation(2), seq.new_operation(3)),
            closable_device<bidirectional>(
                seq.new_operation(1), seq.new_operation(4)))));
        ch.push(closable_device<output>(seq.new_operation(5)));
        HPX_TEST_NO_THROW(ch.reset());
        HPX_TEST_MSG(seq.is_success(), seq.message());
    }

    // Tee a composite composite seekable device with a sink
    {
        operation_sequence seq;
        chain<output> ch;
        ch.push(hpx::iostream::tee(hpx::iostream::compose(
            closable_filter<seekable>(seq.new_operation(1)),
            closable_device<seekable>(seq.new_operation(2)))));
        ch.push(closable_device<output>(seq.new_operation(3)));
        HPX_TEST_NO_THROW(ch.reset());
        HPX_TEST_MSG(seq.is_success(), seq.message());
    }
}

int main(int, char*[])
{
    read_write_test();
    close_test();
    return hpx::util::report_errors();
}
