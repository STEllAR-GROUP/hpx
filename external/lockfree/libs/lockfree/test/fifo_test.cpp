#include <boost/lockfree/fifo.hpp>

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


BOOST_AUTO_TEST_CASE( simple_fifo_test )
{
    fifo<int> f(64);
    BOOST_CHECK(f.empty());
    f.enqueue(1);
    f.enqueue(2);

    int i1(0), i2(0);

    BOOST_CHECK(f.dequeue(&i1));
    BOOST_CHECK_EQUAL(i1, 1);

    BOOST_CHECK(f.dequeue(&i2));
    BOOST_CHECK_EQUAL(i2, 2);
    BOOST_CHECK(f.empty());
}


BOOST_AUTO_TEST_CASE( fifo_specialization_test )
{
    fifo<int*> f;
    BOOST_CHECK(f.empty());
    f.enqueue(new int(1));
    f.enqueue(new int(2));
    f.enqueue(new int(3));
    f.enqueue(new int(4));

    {
        int * i1;

        BOOST_CHECK(f.dequeue(&i1));
        BOOST_CHECK_EQUAL(*i1, 1);
        delete i1;
    }


    {
        auto_ptr<int> i2;
        BOOST_CHECK(f.dequeue(i2));
        BOOST_CHECK_EQUAL(*i2, 2);
    }

    {
        boost::scoped_ptr<int> i3;
        BOOST_CHECK(f.dequeue(i3));

        BOOST_CHECK_EQUAL(*i3, 3);
    }

    {
        boost::scoped_ptr<int> i4;
        BOOST_CHECK(f.dequeue(i4));

        BOOST_CHECK_EQUAL(*i4, 4);
    }


    BOOST_CHECK(f.empty());
}

fifo<int> sf;

atomic_int<long> fifo_cnt;

static_hashed_set<int, 1<<16 > working_set;

const uint nodes_per_thread = 20000000/* 0 */;

const int reader_threads = 5;
const int writer_threads = 5;

void add(void)
{
    for (uint i = 0; i != nodes_per_thread; ++i)
    {
        while(fifo_cnt > 10000)
            thread::yield();

        int id = generate_id<int>();

        working_set.insert(id);
        sf.enqueue(id);

        ++fifo_cnt;
    }
}

atomic_int<long> received_nodes;

bool get_element(void)
{
    int data;

    bool success = sf.dequeue(&data);

    if (success)
    {
        ++received_nodes;
        --fifo_cnt;
        bool erased = working_set.erase(data);
        assert(erased);
        return true;
    }
    else
        return false;
}

volatile bool running = true;

void get(void)
{
    for(;;)
    {
        bool success = get_element();
        if (not running and not success)
            return;
        if (not success)
            thread::yield();
    }
}


BOOST_AUTO_TEST_CASE( fifo_test )
{
#if 0
    add();
    running = false;
    get();
#else
    thread_group writer;
    thread_group reader;

    for (int i = 0; i != reader_threads; ++i)
        reader.create_thread(static_cast<void(*)(void)>(&get));

    for (int i = 0; i != writer_threads; ++i)
        writer.create_thread(&add);
    cout << "reader and writer threads created" << endl;

    writer.join_all();
    cout << "writer threads joined. waiting for readers to finish" << endl;

    running = false;
    reader.join_all();

#endif

    BOOST_CHECK_EQUAL(received_nodes, writer_threads * nodes_per_thread);
    BOOST_CHECK_EQUAL(fifo_cnt, 0);
    BOOST_CHECK(sf.empty());
    BOOST_CHECK(working_set.count_nodes() == 0);
}
