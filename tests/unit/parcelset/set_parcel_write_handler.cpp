//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/hpx.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/atomic.hpp>

#include <vector>

///////////////////////////////////////////////////////////////////////////////
void test(hpx::id_type const& id)
{
    hpx::trigger_lco_event(id);
}
HPX_PLAIN_ACTION(test);     // defines test_action

///////////////////////////////////////////////////////////////////////////////
boost::atomic<std::size_t> write_handler_called(0);

bool is_test_action(hpx::parcelset::parcel const& p)
{
    return dynamic_cast<
            hpx::actions::transfer_action<test_action>*
        >(p.get_action()) != nullptr;
}

void write_handler(boost::system::error_code const&,
    hpx::parcelset::parcel const& p)
{
    if (is_test_action(p))
        ++write_handler_called;
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    hpx::parcel_write_handler_type wh(&write_handler);

    // test that handler is called for every parcel
    hpx::parcel_write_handler_type f1 = hpx::set_parcel_write_handler(wh);
    HPX_TEST(!hpx::util::is_empty_function(f1));

    std::vector<hpx::id_type> localities = hpx::find_remote_localities();

    {
        std::vector<hpx::future<void> > wait_for;
        for (hpx::id_type const& id: localities)
        {
            hpx::lcos::promise<void> p;
            auto f = p.get_future();

            hpx::apply<test_action>(id, p.get_id());

            wait_for.push_back(std::move(f));
        }

        hpx::wait_all(wait_for);
        HPX_TEST_EQ(write_handler_called, localities.size());
    }

    // test that handler is not called anymore
    write_handler_called.store(0);
    hpx::parcel_write_handler_type f2 = hpx::set_parcel_write_handler(f1);

    {
        std::vector<hpx::future<void> > wait_for;
        for (hpx::id_type const& id: localities)
        {
            hpx::lcos::promise<void> p;
            auto f = p.get_future();

            hpx::apply<test_action>(id, p.get_id());

            wait_for.push_back(std::move(f));
        }

        hpx::wait_all(wait_for);
        HPX_TEST_EQ(write_handler_called, std::size_t(0));
    }

    HPX_TEST(f2.target<hpx::parcel_write_handler_type>() ==
        wh.target<hpx::parcel_write_handler_type>());

    return hpx::util::report_errors();
}
