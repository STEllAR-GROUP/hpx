// Copyright (c) 2015 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_main.hpp>

#include <hpx/util/lightweight_test.hpp>

void dummy() {}

int hpx_main()
{
    {
        typedef hpx::lcos::promise<void> promise_type;
        typedef promise_type::wrapped_type shared_state_type;

        hpx::lcos::promise<void> promise;
        hpx::future<void> future = promise.get_future();

        hpx::id_type id1 = promise.get_gid();

        auto future_data = hpx::lcos::detail::get_shared_state(future);
        auto shared_state = boost::static_pointer_cast<shared_state_type>(future_data);
        hpx::id_type id2 = shared_state->get_gid();

        HPX_TEST_EQ(id1, id2);

        using hpx::naming::detail::strip_internal_bits_and_locality_from_gid;

        boost::uint64_t msb1 =
            strip_internal_bits_and_locality_from_gid(id1.get_msb());
        boost::uint64_t msb2 =
            strip_internal_bits_and_locality_from_gid(id2.get_msb());

        HPX_TEST_EQ(msb1, msb2);
        HPX_TEST_EQ(id1.get_lsb(), id2.get_lsb());
    }

    return hpx::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    HPX_TEST_EQ(hpx::init(argc, argv), 0);
    return hpx::util::report_errors();
}
