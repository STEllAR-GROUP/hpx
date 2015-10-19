//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/util/lightweight_test.hpp>

#define MAX_ITERATIONS static_cast<std::size_t>(100)

///////////////////////////////////////////////////////////////////////////////
char const* buffer_basename = "/receive_buffer_1733/buffer/";

inline std::size_t idx(std::size_t i, int dir)
{
    HPX_ASSERT(dir == 1 || dir == -1);

    std::size_t size = hpx::get_num_localities_sync();

    if (i == 0 && dir == -1)
        return size - 1;
    if (i == size - 1 && dir == +1)
        return 0;

    HPX_ASSERT((i + dir) < size);

    return i + dir;
}

class test_receive_buffer_server
  : public hpx::components::simple_component_base<test_receive_buffer_server>
{
public:
    test_receive_buffer_server()
      : from_(hpx::find_from_basename(
            buffer_basename, idx(hpx::get_locality_id(), -1)))
    {}

    void from(std::size_t t, std::size_t d)
    {
        buffer_.store_received(t, std::move(d));
    }

    void do_work();

    HPX_DEFINE_COMPONENT_ACTION(test_receive_buffer_server, from, from_action);
    HPX_DEFINE_COMPONENT_ACTION(test_receive_buffer_server, do_work, do_work_action);

protected:
    hpx::future<std::size_t> receive(std::size_t t)
    {
        return buffer_.receive(t);
    }
    void send(std::size_t t, std::size_t d)
    {
        hpx::apply(from_action(), from_.get(), t, d);
    }

private:
    hpx::shared_future<hpx::id_type> from_;
    hpx::lcos::local::receive_buffer<std::size_t> buffer_;
};

typedef hpx::components::simple_component<
        test_receive_buffer_server
    > server_type;
HPX_REGISTER_COMPONENT(server_type, server_type);

typedef server_type::from_action from_action;
HPX_REGISTER_ACTION(from_action);

typedef server_type::do_work_action do_work_action;
HPX_REGISTER_ACTION(do_work_action);

///////////////////////////////////////////////////////////////////////////////
struct test_receive_buffer
  : hpx::components::client_base<test_receive_buffer, test_receive_buffer_server>
{
    typedef hpx::components::client_base<
            test_receive_buffer, test_receive_buffer_server
        > base_type;

    // construct new instances/wrap existing steppers from other localities
    test_receive_buffer()
      : base_type(hpx::new_<test_receive_buffer_server>(hpx::find_here()))
    {
        hpx::register_with_basename(buffer_basename, get_id(),
            hpx::get_locality_id());
    }

    test_receive_buffer(hpx::future<hpx::id_type> && id)
      : base_type(std::move(id))
    {}

    hpx::future<void> do_work()
    {
        return hpx::async(do_work_action(), get_id());
    }
};

void test_receive_buffer_server::do_work()
{
    send(0, 0);        // send initial value

    std::vector<hpx::future<std::size_t> > steps;
    steps.reserve(MAX_ITERATIONS);

    for (std::size_t i = 0; i != MAX_ITERATIONS; ++i)
    {
        hpx::future<std::size_t> f = receive(i);
        steps.push_back(
            f.then(
                [this, i](hpx::future<std::size_t> && f) -> std::size_t
                {
                    std::size_t val = f.get();
                    send(i + 1, val + 1);
                    return val;
                })
        );
    }

    // receive final value
    HPX_TEST_EQ(receive(MAX_ITERATIONS).get(), MAX_ITERATIONS);

    // verify received values
    hpx::wait_all(steps);
    for (std::size_t i = 0; i != MAX_ITERATIONS; ++i)
    {
        HPX_TEST_EQ(steps[i].get(), i);
    }
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    test_receive_buffer buffer;
    hpx::future<void> result = buffer.do_work();

    result.get();
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // This test requires to run hpx_main on all localities
    std::vector<std::string> cfg;
    cfg.push_back("hpx.run_hpx_main!=1");

    HPX_TEST_EQ_MSG(hpx::init(argc, argv, cfg), 0,
        "HPX main exited with non-zero status");
    return hpx::util::report_errors();
}

