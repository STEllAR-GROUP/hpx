#include <boost/lockfree/ringbuffer.hpp>

#include <climits>
#define BOOST_TEST_MODULE lockfree_tests
#include <boost/test/included/unit_test.hpp>


#include <boost/thread.hpp>
#include <iostream>
#include <memory>


#include "test_helpers.hpp"

using namespace boost;
using namespace boost::lockfree;
using namespace std;


BOOST_AUTO_TEST_CASE( simple_ringbuffer_test )
{
    ringbuffer<int, 64> f;

    BOOST_REQUIRE(f.empty());
    f.enqueue(1);
    f.enqueue(2);

    int i1(0), i2(0);

    BOOST_REQUIRE(f.dequeue(&i1));
    BOOST_REQUIRE_EQUAL(i1, 1);

    BOOST_REQUIRE(f.dequeue(&i2));
    BOOST_REQUIRE_EQUAL(i2, 2);
    BOOST_REQUIRE(f.empty());
}

BOOST_AUTO_TEST_CASE( simple_ringbuffer_buffer_test )
{
    const size_t xqueue_size = 6;

    ringbuffer<int, 16> f;

    int data[xqueue_size];

    for (size_t i = 0; i != xqueue_size; ++i)
        data[i] = i*2;

    BOOST_REQUIRE(f.empty());
    BOOST_REQUIRE_EQUAL(f.enqueue(data, xqueue_size), xqueue_size);

    int out[xqueue_size];

    BOOST_REQUIRE_EQUAL(f.dequeue(out, xqueue_size), xqueue_size);

    for (size_t i = 0; i != xqueue_size; ++i)
        BOOST_REQUIRE_EQUAL(data[i], out[i]);
}

BOOST_AUTO_TEST_CASE( simple_ringbuffer_buffer_test_2 )
{
    const size_t xqueue_size = 15;

    ringbuffer<int, 16> f;

    int data[xqueue_size];

    for (size_t i = 0; i != xqueue_size; ++i)
        data[i] = i*2;

    BOOST_REQUIRE(f.empty());
    BOOST_REQUIRE_EQUAL(f.enqueue(data, xqueue_size), xqueue_size);

    int out[xqueue_size];

    BOOST_REQUIRE_EQUAL(f.dequeue(out, xqueue_size), xqueue_size);

    for (size_t i = 0; i != xqueue_size; ++i)
        BOOST_REQUIRE_EQUAL(data[i], out[i]);

    BOOST_REQUIRE(f.empty());
    BOOST_REQUIRE_EQUAL(f.enqueue(data, xqueue_size), xqueue_size);

    BOOST_REQUIRE_EQUAL(f.dequeue(out, xqueue_size), xqueue_size);

    for (size_t i = 0; i != xqueue_size; ++i)
        BOOST_REQUIRE_EQUAL(data[i], out[i]);

    BOOST_REQUIRE(f.empty());
}


static const uint nodes_per_thread = 500000;

struct ringbuffer_tester
{
    ringbuffer<int, 128> sf;

    atomic<long> ringbuffer_cnt, received_nodes;

    static_hashed_set<int, 1<<16 > working_set;

    ringbuffer_tester(void):
        ringbuffer_cnt(0), received_nodes(0)
    {}

    void add(void)
    {
        for (uint i = 0; i != nodes_per_thread; ++i)
        {
            int id = generate_id<int>();

            working_set.insert(id);

            while (sf.enqueue(id) == false)
            {}

            ++ringbuffer_cnt;
        }
    }

    bool get_element(void)
    {
        int data;

        bool success = sf.dequeue(&data);

        if (success)
        {
            ++received_nodes;
            --ringbuffer_cnt;
            bool erased = working_set.erase(data);
            assert(erased);
            return true;
        }
        else
            return false;
    }

    volatile bool running;

    void get(void)
    {
        for(;;)
        {
            bool success = get_element();
            if (not running and not success)
                return;
        }
    }

    void run(void)
    {
        running = true;

        BOOST_REQUIRE(sf.empty());

        thread reader(boost::bind(&ringbuffer_tester::get, this));
        thread writer(boost::bind(&ringbuffer_tester::add, this));
        cout << "reader and writer threads created" << endl;

        writer.join();
        cout << "writer threads joined. waiting for readers to finish" << endl;

        running = false;
        reader.join();

        BOOST_REQUIRE_EQUAL(received_nodes, nodes_per_thread);
        BOOST_REQUIRE_EQUAL(ringbuffer_cnt, 0);
        BOOST_REQUIRE(sf.empty());
        BOOST_REQUIRE(working_set.count_nodes() == 0);
    }
};

BOOST_AUTO_TEST_CASE( ringbuffer_test_caching )
{
    ringbuffer_tester test1;
    //test1.run();
}

struct ringbuffer_tester_buffering
{
    ringbuffer<int, 128> sf;

    atomic<long> ringbuffer_cnt;

    static_hashed_set<int, 1<<16 > working_set;
    atomic<long> received_nodes;

    ringbuffer_tester_buffering(void):
        ringbuffer_cnt(0), received_nodes(0)
    {}

    static const size_t buf_size = 5;

    void add(void)
    {
        boost::array<int, buf_size> input_buffer;
        for (uint i = 0; i != nodes_per_thread; i+=buf_size)
        {
            for (size_t i = 0; i != buf_size; ++i)
            {
                int id = generate_id<int>();
                working_set.insert(id);
                input_buffer[i] = id;
            }

            size_t enqueued = 0;

            do
            {
                enqueued += sf.enqueue(input_buffer.c_array() + enqueued,
                                       input_buffer.size()    - enqueued);
            }
            while (enqueued != buf_size);

            ringbuffer_cnt+=buf_size;
        }
    }

    bool get_elements(void)
    {
        boost::array<int, buf_size> output_buffer;

        size_t dequeued = sf.dequeue(output_buffer.c_array(), output_buffer.size());

        if (dequeued)
        {
            received_nodes += dequeued;
            ringbuffer_cnt -= dequeued;

            for (size_t i = 0; i != dequeued; ++i)
            {
                bool erased = working_set.erase(output_buffer[i]);
                assert(erased);
            }

            return true;
        }
        else
            return false;
    }

    volatile bool running;

    void get(void)
    {
        for(;;)
        {
            bool success = get_elements();
            if (not running and not success)
                return;
        }
    }

    void run(void)
    {
        running = true;

        thread reader(boost::bind(&ringbuffer_tester_buffering::get, this));
        thread writer(boost::bind(&ringbuffer_tester_buffering::add, this));
        cout << "reader and writer threads created" << endl;

        writer.join();
        cout << "writer threads joined. waiting for readers to finish" << endl;

        running = false;
        reader.join();

        BOOST_REQUIRE_EQUAL(received_nodes, nodes_per_thread);
        BOOST_REQUIRE_EQUAL(ringbuffer_cnt, 0);
        BOOST_REQUIRE(sf.empty());
        BOOST_REQUIRE(working_set.count_nodes() == 0);
    }
};

BOOST_AUTO_TEST_CASE( ringbuffer_test_buffering )
{
    ringbuffer_tester_buffering test1;
    test1.run();
}
