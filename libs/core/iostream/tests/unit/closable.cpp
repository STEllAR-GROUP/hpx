//  Copyright (c) 2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// See http://www.boost.org/libs/iostreams for documentation.
//
// Verifies that the close() member functions of filters and devices
// are called with the correct arguments in the correct order when
// used with chains and streams.
//
// File:        libs/iostreams/test/close_test.cpp
// Date:        Sun Dec 09 16:12:23 MST 2007
// Copyright:   2007 CodeRage
// Author:      Jonathan Turkanis

#include <hpx/hpx_main.hpp>
#include <hpx/modules/iostream.hpp>
#include <hpx/modules/testing.hpp>

#include "detail/closable.hpp"
#include "detail/operation_sequence.hpp"

using namespace std;
using namespace hpx::iostream;
using namespace hpx::iostream::test;
namespace io = hpx::iostream;

void input_chain_test()
{
    // Test input filter and device
    {
        operation_sequence seq;
        filtering_streambuf<input> ch;

        // Test chain::pop()
        ch.push(closable_filter<input>(seq.new_operation(2)));
        ch.push(closable_device<input>(seq.new_operation(1)));
        HPX_TEST_NO_THROW(ch.pop());
        check_operation_sequence(seq);

        // Test filter reuse and io::close()
        seq.reset();
        ch.push(closable_device<input>(seq.new_operation(1)));
        HPX_TEST_NO_THROW(io::close(ch));
        check_operation_sequence(seq);

        // Test filter reuse and chain::reset()
        seq.reset();
        ch.push(closable_device<input>(seq.new_operation(1)));
        HPX_TEST_NO_THROW(ch.reset());
        check_operation_sequence(seq);
    }

    // Test bidirectional filter and device
    {
        operation_sequence seq;
        filtering_streambuf<input> ch;

        // Test chain::pop()
        ch.push(closable_filter<bidirectional>(
            seq.new_operation(2), seq.new_operation(3)));
        ch.push(closable_device<bidirectional>(
            seq.new_operation(1), seq.new_operation(4)));
        HPX_TEST_NO_THROW(ch.pop());
        check_operation_sequence(seq);

        // Test filter reuse and io::close()
        seq.reset();
        ch.push(closable_device<bidirectional>(
            seq.new_operation(1), seq.new_operation(4)));
        HPX_TEST_NO_THROW(io::close(ch));
        check_operation_sequence(seq);

        // Test filter reuse and chain::reset()
        seq.reset();
        ch.push(closable_device<bidirectional>(
            seq.new_operation(1), seq.new_operation(4)));
        HPX_TEST_NO_THROW(ch.reset());
        check_operation_sequence(seq);
    }

    // Test seekable filter and device
    {
        operation_sequence seq;
        filtering_streambuf<input> ch;

        // Test chain::pop()
        ch.push(closable_filter<seekable>(seq.new_operation(1)));
        ch.push(closable_device<seekable>(seq.new_operation(2)));
        HPX_TEST_NO_THROW(ch.pop());
        check_operation_sequence(seq);

        // Test filter reuse and io::close()
        seq.reset();
        ch.push(closable_device<seekable>(seq.new_operation(2)));
        HPX_TEST_NO_THROW(io::close(ch));
        check_operation_sequence(seq);

        // Test filter reuse and chain::reset()
        seq.reset();
        ch.push(closable_device<seekable>(seq.new_operation(2)));
        HPX_TEST_NO_THROW(ch.reset());
        check_operation_sequence(seq);
    }

    // Test dual-user filter
    {
        operation_sequence seq;
        filtering_streambuf<input> ch;
        operation dummy;

        // Test chain::pop()
        ch.push(closable_filter<dual_use>(seq.new_operation(2), dummy));
        ch.push(closable_device<input>(seq.new_operation(1)));
        HPX_TEST_NO_THROW(ch.pop());
        check_operation_sequence(seq);

        // Test filter reuse and io::close()
        seq.reset();
        ch.push(closable_device<input>(seq.new_operation(1)));
        HPX_TEST_NO_THROW(io::close(ch));
        check_operation_sequence(seq);

        // Test filter reuse and chain::reset()
        seq.reset();
        ch.push(closable_device<input>(seq.new_operation(1)));
        HPX_TEST_NO_THROW(ch.reset());
        check_operation_sequence(seq);
    }

    // Test direct source
    {
        operation_sequence seq;
        filtering_streambuf<input> ch;

        // Test chain::pop()
        ch.push(closable_filter<input>(seq.new_operation(2)));
        ch.push(closable_device<direct_input>(seq.new_operation(1)));
        HPX_TEST_NO_THROW(ch.pop());
        check_operation_sequence(seq);

        // Test filter reuse and io::close()
        seq.reset();
        ch.push(closable_device<direct_input>(seq.new_operation(1)));
        HPX_TEST_NO_THROW(io::close(ch));
        check_operation_sequence(seq);

        // Test filter reuse and chain::reset()
        seq.reset();
        ch.push(closable_device<direct_input>(seq.new_operation(1)));
        HPX_TEST_NO_THROW(ch.reset());
        check_operation_sequence(seq);
    }

    // Test direct bidirectional device
    {
        operation_sequence seq;
        filtering_streambuf<input> ch;

        // Test chain::pop()
        ch.push(closable_filter<input>(seq.new_operation(2)));
        ch.push(closable_device<direct_bidirectional>(
            seq.new_operation(1), seq.new_operation(3)));
        HPX_TEST_NO_THROW(ch.pop());
        check_operation_sequence(seq);

        // Test filter reuse and io::close()
        seq.reset();
        ch.push(closable_device<direct_bidirectional>(
            seq.new_operation(1), seq.new_operation(3)));
        HPX_TEST_NO_THROW(io::close(ch));
        check_operation_sequence(seq);

        // Test filter reuse and chain::reset()
        seq.reset();
        ch.push(closable_device<direct_bidirectional>(
            seq.new_operation(1), seq.new_operation(3)));
        HPX_TEST_NO_THROW(ch.reset());
        check_operation_sequence(seq);
    }

    // Test direct seekable device
    {
        operation_sequence seq;
        filtering_streambuf<input> ch;

        // Test chain::pop()
        ch.push(closable_filter<input>(seq.new_operation(1)));
        ch.push(closable_device<direct_seekable>(seq.new_operation(2)));
        HPX_TEST_NO_THROW(ch.pop());
        check_operation_sequence(seq);

        // Test filter reuse and io::close()
        seq.reset();
        ch.push(closable_device<direct_seekable>(seq.new_operation(2)));
        HPX_TEST_NO_THROW(io::close(ch));
        check_operation_sequence(seq);

        // Test filter reuse and chain::reset()
        seq.reset();
        ch.push(closable_device<direct_seekable>(seq.new_operation(2)));
        HPX_TEST_NO_THROW(ch.reset());
        check_operation_sequence(seq);
    }
}

void output_chain_test()
{
    // Test output filter and device
    {
        operation_sequence seq;
        filtering_streambuf<output> ch;

        // Test chain::pop()
        ch.push(closable_filter<output>(seq.new_operation(1)));
        ch.push(closable_device<output>(seq.new_operation(2)));
        HPX_TEST_NO_THROW(ch.pop());
        check_operation_sequence(seq);

        // Test filter reuse and io::close()
        seq.reset();
        ch.push(closable_device<output>(seq.new_operation(2)));
        HPX_TEST_NO_THROW(io::close(ch));
        check_operation_sequence(seq);

        // Test filter reuse and chain::reset()
        seq.reset();
        ch.push(closable_device<output>(seq.new_operation(2)));
        HPX_TEST_NO_THROW(ch.reset());
        check_operation_sequence(seq);
    }

    // Test bidirectional filter and device
    {
        operation_sequence seq;
        filtering_streambuf<output> ch;

        // Test chain::pop()
        ch.push(closable_filter<bidirectional>(
            seq.new_operation(2), seq.new_operation(3)));
        ch.push(closable_device<bidirectional>(
            seq.new_operation(1), seq.new_operation(4)));
        HPX_TEST_NO_THROW(ch.pop());
        check_operation_sequence(seq);

        // Test filter reuse and io::close()
        seq.reset();
        ch.push(closable_device<bidirectional>(
            seq.new_operation(1), seq.new_operation(4)));
        HPX_TEST_NO_THROW(io::close(ch));
        check_operation_sequence(seq);

        // Test filter reuse and chain::reset()
        seq.reset();
        ch.push(closable_device<bidirectional>(
            seq.new_operation(1), seq.new_operation(4)));
        HPX_TEST_NO_THROW(ch.reset());
        check_operation_sequence(seq);
    }

    // Test seekable filter and device
    {
        operation_sequence seq;
        filtering_streambuf<output> ch;

        // Test chain::pop()
        ch.push(closable_filter<seekable>(seq.new_operation(1)));
        ch.push(closable_device<seekable>(seq.new_operation(2)));
        HPX_TEST_NO_THROW(ch.pop());
        check_operation_sequence(seq);

        // Test filter reuse and io::close()
        seq.reset();
        ch.push(closable_device<seekable>(seq.new_operation(2)));
        HPX_TEST_NO_THROW(io::close(ch));
        check_operation_sequence(seq);

        // Test filter reuse and chain::reset()
        seq.reset();
        ch.push(closable_device<seekable>(seq.new_operation(2)));
        HPX_TEST_NO_THROW(ch.reset());
        check_operation_sequence(seq);
    }

    // Test dual-user filter
    {
        operation_sequence seq;
        filtering_streambuf<output> ch;
        operation dummy;

        // Test chain::pop()
        ch.push(closable_filter<dual_use>(dummy, seq.new_operation(1)));
        ch.push(closable_device<output>(seq.new_operation(3)));
        HPX_TEST_NO_THROW(ch.pop());
        check_operation_sequence(seq);

        // Test filter reuse and io::close()
        seq.reset();
        ch.push(closable_device<output>(seq.new_operation(3)));
        HPX_TEST_NO_THROW(io::close(ch));
        check_operation_sequence(seq);

        // Test filter reuse and chain::reset()
        seq.reset();
        ch.push(closable_device<output>(seq.new_operation(3)));
        HPX_TEST_NO_THROW(ch.reset());
        check_operation_sequence(seq);
    }

    // Test direct sink
    {
        operation_sequence seq;
        filtering_streambuf<output> ch;

        // Test chain::pop()
        ch.push(closable_filter<output>(seq.new_operation(1)));
        ch.push(closable_device<direct_output>(seq.new_operation(2)));
        HPX_TEST_NO_THROW(ch.pop());
        check_operation_sequence(seq);

        // Test filter reuse and io::close()
        seq.reset();
        ch.push(closable_device<direct_output>(seq.new_operation(2)));
        HPX_TEST_NO_THROW(io::close(ch));
        check_operation_sequence(seq);

        // Test filter reuse and chain::reset()
        seq.reset();
        ch.push(closable_device<direct_output>(seq.new_operation(2)));
        HPX_TEST_NO_THROW(ch.reset());
        check_operation_sequence(seq);
    }

    // Test direct bidirectional device
    {
        operation_sequence seq;
        filtering_streambuf<output> ch;

        // Test chain::pop()
        ch.push(closable_filter<output>(seq.new_operation(2)));
        ch.push(closable_device<direct_bidirectional>(
            seq.new_operation(1), seq.new_operation(3)));
        HPX_TEST_NO_THROW(ch.pop());
        check_operation_sequence(seq);

        // Test filter reuse and io::close()
        seq.reset();
        ch.push(closable_device<direct_bidirectional>(
            seq.new_operation(1), seq.new_operation(3)));
        HPX_TEST_NO_THROW(io::close(ch));
        check_operation_sequence(seq);

        // Test filter reuse and chain::reset()
        seq.reset();
        ch.push(closable_device<direct_bidirectional>(
            seq.new_operation(1), seq.new_operation(3)));
        HPX_TEST_NO_THROW(ch.reset());
        check_operation_sequence(seq);
    }

    // Test direct seekable device
    {
        operation_sequence seq;
        filtering_streambuf<output> ch;

        // Test chain::pop()
        ch.push(closable_filter<output>(seq.new_operation(1)));
        ch.push(closable_device<direct_seekable>(seq.new_operation(2)));
        HPX_TEST_NO_THROW(ch.pop());
        check_operation_sequence(seq);

        // Test filter reuse and io::close()
        seq.reset();
        ch.push(closable_device<direct_seekable>(seq.new_operation(2)));
        HPX_TEST_NO_THROW(io::close(ch));
        check_operation_sequence(seq);

        // Test filter reuse and chain::reset()
        seq.reset();
        ch.push(closable_device<direct_seekable>(seq.new_operation(2)));
        HPX_TEST_NO_THROW(ch.reset());
        check_operation_sequence(seq);
    }
}

void bidirectional_chain_test()
{
    // Test bidirectional filter and device
    {
        operation_sequence seq;
        filtering_streambuf<bidirectional> ch;

        // Test chain::pop()
        ch.push(closable_filter<bidirectional>(
            seq.new_operation(2), seq.new_operation(3)));
        ch.push(closable_device<bidirectional>(
            seq.new_operation(1), seq.new_operation(4)));
        HPX_TEST_NO_THROW(ch.pop());
        check_operation_sequence(seq);

        // Test filter reuse and io::close()
        seq.reset();
        ch.push(closable_device<bidirectional>(
            seq.new_operation(1), seq.new_operation(4)));
        HPX_TEST_NO_THROW(io::close(ch));
        check_operation_sequence(seq);

        // Test filter reuse and chain::reset()
        seq.reset();
        ch.push(closable_device<bidirectional>(
            seq.new_operation(1), seq.new_operation(4)));
        HPX_TEST_NO_THROW(ch.reset());
        check_operation_sequence(seq);
    }

    // Test direct bidirectional device
    {
        operation_sequence seq;
        filtering_streambuf<bidirectional> ch;

        // Test chain::pop()
        ch.push(closable_filter<bidirectional>(
            seq.new_operation(2), seq.new_operation(3)));
        ch.push(closable_device<direct_bidirectional>(
            seq.new_operation(1), seq.new_operation(4)));
        HPX_TEST_NO_THROW(ch.pop());
        check_operation_sequence(seq);

        // Test filter reuse and io::close()
        seq.reset();
        ch.push(closable_device<direct_bidirectional>(
            seq.new_operation(1), seq.new_operation(4)));
        HPX_TEST_NO_THROW(io::close(ch));
        check_operation_sequence(seq);

        // Test filter reuse and chain::reset()
        seq.reset();
        ch.push(closable_device<direct_bidirectional>(
            seq.new_operation(1), seq.new_operation(4)));
        HPX_TEST_NO_THROW(ch.reset());
        check_operation_sequence(seq);
    }
}

void seekable_chain_test()
{
    // Test seekable filter and device
    {
        operation_sequence seq;
        filtering_streambuf<seekable> ch;

        // Test chain::pop()
        ch.push(closable_filter<seekable>(seq.new_operation(1)));
        ch.push(closable_device<seekable>(seq.new_operation(2)));
        HPX_TEST_NO_THROW(ch.pop());
        check_operation_sequence(seq);

        // Test filter reuse and io::close()
        seq.reset();
        ch.push(closable_device<seekable>(seq.new_operation(2)));
        HPX_TEST_NO_THROW(io::close(ch));
        check_operation_sequence(seq);

        // Test filter reuse and chain::reset()
        seq.reset();
        ch.push(closable_device<seekable>(seq.new_operation(2)));
        HPX_TEST_NO_THROW(ch.reset());
        check_operation_sequence(seq);
    }

    // Test direct seekable device
    {
        operation_sequence seq;
        filtering_streambuf<seekable> ch;

        // Test chain::pop()
        ch.push(closable_filter<seekable>(seq.new_operation(1)));
        ch.push(closable_device<direct_seekable>(seq.new_operation(2)));
        HPX_TEST_NO_THROW(ch.pop());
        check_operation_sequence(seq);

        // Test filter reuse and io::close()
        seq.reset();
        ch.push(closable_device<direct_seekable>(seq.new_operation(2)));
        HPX_TEST_NO_THROW(io::close(ch));
        check_operation_sequence(seq);

        // Test filter reuse and chain::reset()
        seq.reset();
        ch.push(closable_device<direct_seekable>(seq.new_operation(2)));
        HPX_TEST_NO_THROW(ch.reset());
        check_operation_sequence(seq);
    }
}

void stream_test()
{
    // Test source
    {
        operation_sequence seq;
        stream<closable_device<input>> str;
        str.open(closable_device<input>(seq.new_operation(1)));
        HPX_TEST_NO_THROW(str.close());
        check_operation_sequence(seq);
    }

    // Test sink
    {
        operation_sequence seq;
        stream<closable_device<output>> str;
        str.open(closable_device<output>(seq.new_operation(1)));
        HPX_TEST_NO_THROW(str.close());
        check_operation_sequence(seq);
    }

    // Test bidirectional device
    {
        operation_sequence seq;
        stream<closable_device<bidirectional>> str;
        str.open(closable_device<bidirectional>(
            seq.new_operation(1), seq.new_operation(2)));
        HPX_TEST_NO_THROW(str.close());
        check_operation_sequence(seq);
    }

    // Test seekable device
    {
        operation_sequence seq;
        stream<closable_device<seekable>> str;
        str.open(closable_device<seekable>(seq.new_operation(1)));
        HPX_TEST_NO_THROW(str.close());
        check_operation_sequence(seq);
    }
}

int main(int, char*[])
{
    input_chain_test();
    output_chain_test();
    bidirectional_chain_test();
    seekable_chain_test();
    stream_test();
    return hpx::util::report_errors();
}
