//  Copyright (c) 2014 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/components/iostreams/standard_streams.hpp>
#include <hpx/util/format.hpp>
#include <hpx/util/yield_while.hpp>
#include <hpx/lcos/barrier.hpp>
//
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/parcelset/rma/rma_object.hpp>
#include <hpx/runtime/parcelset/rma/allocator.hpp>
//
#include <boost/assert.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>
//
#include <simple_profiler.hpp>

//
// This is a test program which reads and writes chunks of memory to storage
// distributed across localities.
//
// The principal problem can be summarized as follows:
//
// Put data into remote memory
//    1 Pass user data pointer into serialize_buffer (0 copy)
//    2 Copy data into parcelport for transmission. (1 copy)
//    3 Transmit data into remote parcelport buffer
//    4 Copy data from parcelport into serialize_buffer (2 copy)
//    5 Copy data from serialize buffer into remote host storage (3 copy ?)
//
// Get data from remote memory
//    1 Allocate temporary buffer and copy from host storage into it (1 copy)
//    2 Wrap storage in serialize_buffer and give to parcelport for return
//    3 Copy data from serialize_buffer into parcelport for transmission (2 copy)
//    4 Receive data into local parcelport buffer
//    5 Copy parcelport buffer to serialize_buffer(user data pointer) (3 copy)
//
// The ideal situation would be as follows
//
// Put into remote memory
//    1 Pass user data pointer into serialize_buffer (0 copy)
//    2 Copy data into parcelport for transmission. (1 copy)
//    3 Transmit data into remote parcelport buffer
//    4 Copy data from parcelport into serialize_buffer(user data pointer) (2 copy)
//
// Get data from remote memory
//    1 Request serialize buffer from parcelport (0 copy)
//    2 Copy from storage into serialize_buffer (1 copy)
//    4 Receive data into local parcelport buffer
//    5 Copy parcelport buffer to serialize_buffer(user data pointer) (2 copy)
//
// To make each process run a main function and participate in the test,
// use a command line of the kind (no mpiexec assumed)
//
//     network_storage.exe -l%1 -%%x --hpx:run-hpx-main --hpx:threads=4
//
// (+or more) where %l is num_ranks, and %%x is rank of current process

//----------------------------------------------------------------------------
// @TODO : add support for custom network executor
// local_priority_queue_os_executor exec(4, "thread:0-3=core:12-15.pu:0");
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// #defines to control program
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// Array allocation on start assumes a certain maximum number of localities will be used
#define MAX_RANKS 16384

//----------------------------------------------------------------------------
// Define this to make memory access asynchronous and return a future
//#define ASYNC_MEMORY
#ifdef ASYNC_MEMORY
 typedef hpx::future<int> async_mem_result_type;
 #define ASYNC_MEM_RESULT(x) (hpx::make_ready_future<int>(x));
#else
 typedef int async_mem_result_type;
 #define ASYNC_MEM_RESULT(x) (x);
#endif

//----------------------------------------------------------------------------
// control the amount of debug messaging that is output
#define DEBUG_LEVEL 0

//----------------------------------------------------------------------------
// if we have access to boost logging via the libfabric parcelport include this
// otherwise use this
#include <plugins/parcelport/parcelport_logging.hpp>

#if DEBUG_LEVEL>0
# define LOG_DEBUG_MSG(x) std::cout << "Network storage " << x << std::endl
# define DEBUG_OUTPUT(level,x)   \
    if (DEBUG_LEVEL>=level) {    \
        LOG_DEBUG_MSG(x);        \
    }
# define DEBUG_ONLY(x) x
#else
# define DEBUG_OUTPUT(level,x)
# define DEBUG_ONLY(x)
#endif

//----------------------------------------------------------------------------
#define TEST_FAIL    0
#define TEST_SUCCESS 1

//----------------------------------------------------------------------------
// global vars
//----------------------------------------------------------------------------
//static std::vector<std::vector<hpx::future<int> > > ActiveFutures;
static std::array<std::atomic<int>, MAX_RANKS>     FuturesWaiting;

struct unusual {
    std::array<char,256> some_data;
    std::pair<int, int>  a_pair;
};

HPX_IS_BITWISE_SERIALIZABLE(unusual);

static_assert(
        hpx::traits::is_rma_elegible<unusual>::value,
        "we need this to be serializable"
);

//----------------------------------------------------------------------------
//
// Each locality allocates a buffer of memory which is used to host transfers
//
static hpx::parcelset::rma::rma_vector<char> rma_storage;
static char *local_storage = nullptr;
static hpx::lcos::local::spinlock storage_mutex;

//
typedef struct {
    std::uint64_t iterations;
    std::uint64_t local_storage_MB;
    std::uint64_t global_storage_MB;
    std::uint64_t transfer_size_B;
    std::uint64_t threads;
    std::uint64_t semaphore;
    std::string   network;
    bool          warmup;
    bool          all2all;
    bool          distribution;
    bool          nolocal;
} test_options;

//----------------------------------------------------------------------------
void allocate_local_storage(uint64_t local_storage_bytes)
{
    hpx::parcelset::rma::rma_object<int> x =
        hpx::parcelset::rma::make_rma_object<int>();
    rma_storage.reserve(local_storage_bytes);
    local_storage = new char[local_storage_bytes];
}

//----------------------------------------------------------------------------
void delete_local_storage()
{
    rma_storage.reset();
    delete[] local_storage;
}

//----------------------------------------------------------------------------
void release_storage_lock(char *p)
{
    DEBUG_OUTPUT(6, "Release lock and delete memory");
    delete []p;
//  storage_mutex.unlock();
}

//----------------------------------------------------------------------------
// This routine simply copies from the source buffer into the local memory
// at the address offset.
//
// The function does not need to be asynchronous as it completes immediately,
// but we return a future as this test needs to mimic "asynchronous" storage
async_mem_result_type copy_to_local_storage(char const* src,
    uint32_t offset, uint64_t length)
{
    char *dest = &local_storage[offset];
    std::copy(src, src + length, dest);
    return ASYNC_MEM_RESULT(TEST_SUCCESS);
}

//----------------------------------------------------------------------------
// This routine simply copies from local memory at the address offset
// into a provided buffer
//
// The function does not need to be asynchronous as it completes immediately,
// but we return a future as this test needs to mimic "asynchronous" storage
async_mem_result_type copy_from_local_storage(char *dest,
    uint32_t offset, uint64_t length)
{
    char const* src = &local_storage[offset];
    std::copy(src, src + length, dest);
    return ASYNC_MEM_RESULT(TEST_SUCCESS);
}

//----------------------------------------------------------------------------
// A custom allocator which takes a pointer in its constructor and then returns
// this pointer in response to any allocate request. It is here to try to fool
// the hpx serialization into copying directly into a user provided buffer
// without copying from a result into another buffer.
//
template <typename T>
class pointer_allocator
{
public:
  typedef T value_type;
  typedef T* pointer;
  typedef const T* const_pointer;
  typedef T& reference;
  typedef const T& const_reference;
  typedef std::size_t size_type;
  typedef std::ptrdiff_t difference_type;

  pointer_allocator() noexcept
    : pointer_(nullptr), size_(0)
  {
  }

  pointer_allocator(pointer p, size_type size) noexcept
    : pointer_(p), size_(size)
  {
  }

  pointer address(reference value) const { return &value; }
  const_pointer address(const_reference value) const { return &value; }

  pointer allocate(size_type n, void const* /*hint*/ = nullptr)
  {
    HPX_ASSERT(n == size_);
    return static_cast<T*>(pointer_);
  }

  void deallocate(pointer p, size_type n)
  {
    HPX_ASSERT(p == pointer_ && n == size_);
  }

private:
  // serialization support
  friend class hpx::serialization::access;

  template <typename Archive>
  void load(Archive& ar, unsigned int const /*version*/)
  {
    std::size_t t = 0;
    ar >> size_ >> t;
    pointer_ = reinterpret_cast<pointer>(t);
  }

  template <typename Archive>
  void save(Archive& ar, unsigned int const /*version*/) const
  {
    std::size_t t = reinterpret_cast<std::size_t>(pointer_);
    ar << size_ << t;
  }

  HPX_SERIALIZATION_SPLIT_MEMBER()

private:
  pointer pointer_;
  size_type size_;
};

//----------------------------------------------------------------------------
// A simple Buffer for sending data, it does not need any special allocator
// user data may be sent to another locality using zero copy by wrapping
// it in one of these buffers
typedef hpx::serialization::serialize_buffer<char> general_buffer_type;

// When receiving data, we receive a hpx::serialize_buffer, we try to minimize
// copying of data by providing a receive buffer with a fixed data pointer
// so that data is placed directly into it.
typedef pointer_allocator<char> PointerAllocator;
typedef hpx::serialization::serialize_buffer<char,
    PointerAllocator> transfer_buffer_type;

//----------------------------------------------------------------------------
struct buffer_deleter
{
    uint64_t index_;
    std::shared_ptr<general_buffer_type> buffer_;
    //
//    buffer_deleter(uint64_t index, std::shared_ptr<general_buffer_type> buffer)
//        : index_(index), buffer_(buffer) {}
//    buffer_deleter(buffer_deleter &other)
//        : index_(other.index_), buffer_(other.buffer_) {}
    //
    ~buffer_deleter() {
        DEBUG_OUTPUT(7, "Deleting buffer index " << index_);
        index_  = 0;
        buffer_ = nullptr;
    }
};

//
void async_callback(buffer_deleter deleter, boost::system::error_code const& /*ec*/,
    hpx::parcelset::parcel const& /*p*/)
{
    DEBUG_OUTPUT(7, "Async callback triggered for index " << deleter.index);
}

//----------------------------------------------------------------------------
// Two actions that are called from remote or local localities
//
// Copy to storage, just invokes the local copy function and returns the future
// from it
//
// Copy from storage allocates a buffer for the return memory and then wraps
// it into a serialize buffer which is returned and passed into the local
// process.
//
// This unfortunately means memory is copied from the storage, into a buffer
// and not zero copied as we would like.
// I have not been successful in removing this copy.
namespace Storage {
    //------------------------------------------------------------------------
    // A PUT into memory on this locality from a requester sending a
    // general_buffer_type
    async_mem_result_type CopyToStorage(general_buffer_type const& srcbuffer,
        uint32_t address, uint64_t length)
    {
        boost::shared_array<char> src = srcbuffer.data_array();
        return copy_to_local_storage(src.get(), address, length);
    }

    //------------------------------------------------------------------------
    // A GET from memory on this locality is returned to the requester in the
    // transfer_buffer_type
     hpx::future<transfer_buffer_type> CopyFromStorage(
        uint32_t address, uint64_t length, std::size_t remote_buffer)
    {
        // we must allocate a temporary buffer to copy from storage into
        // we can't use the remote buffer supplied because it is a handle to memory on
        // the (possibly) remote node. We allocate here using
        // a nullptr deleter so the array will
        // not be released by the shared_pointer.
        //
        // The memory must be freed after final use.
        std::allocator<char> local_allocator;
        boost::shared_array<char> local_buffer(local_allocator.allocate(length),
            [](char*){
                DEBUG_OUTPUT(6, "Not deleting memory");
            }
        );

        // allow the storage class to asynchronously copy the data into buffer
#ifdef ASYNC_MEMORY
        async_mem_result_type fut =
#endif
            copy_from_local_storage(local_buffer.get(), address, length);
        DEBUG_OUTPUT(6, "create local buffer count " << local_buffer.use_count());

        // wrap the remote buffer pointer in an allocator for return
        pointer_allocator<char> return_allocator(
          reinterpret_cast<char*>(remote_buffer), length);

        // lock the mutex, will be unlocked by the transfer buffer's deleter
//        storage_mutex.lock();
#ifdef ASYNC_MEMORY
        return fut.then(
            hpx::launch::sync,
            // return the data in a transfer buffer
            [=](hpx::future<int> fut) -> transfer_buffer_type {
                DEBUG_OUTPUT(6, ".then local buffer count " << local_buffer.use_count());
                return transfer_buffer_type(
                    local_buffer.get(), length,
                    transfer_buffer_type::take,
                    [](char *p) {
                        release_storage_lock(p);
                    },
                    return_allocator);
            }
        );
#else
        return hpx::make_ready_future<transfer_buffer_type>(
            transfer_buffer_type(
                local_buffer.get(), length,
                transfer_buffer_type::take,
                [](char *p) {
                    release_storage_lock(p);
                },
                return_allocator
            ));
#endif
    }

    struct simple_barrier
    {
        static std::atomic<uint32_t> process_count_;
        static bool                  first_time_;
        uint32_t                     rank_;
        uint32_t                     nranks_;
        hpx::id_type                 agas_;
        //
        simple_barrier() {}

        void init() {
            // these only need to be done once and are relatively expensive (agas)
            if (simple_barrier::first_time_) {
                simple_barrier::first_time_ = false;
                hpx::id_type here = hpx::find_here();
                rank_             = hpx::naming::get_locality_id_from_id(here);
                agas_             = hpx::agas::get_console_locality();
                nranks_           = hpx::get_num_localities().get();
            }
            // this must always be done to reset the counter
            process_count_ = nranks_;
            DEBUG_OUTPUT(1, "resetting barrier on rank " << rank_ << " " << process_count_);
        }

        static simple_barrier local_barrier;

        static simple_barrier &get_local_barrier() {
            return simple_barrier::local_barrier;
        }
    };

    // instantiate the static barrier and vars
    simple_barrier              simple_barrier::local_barrier;
    std::atomic<unsigned int>   simple_barrier::process_count_;
    bool                        simple_barrier::first_time_ = true;

    // this function will be a remote action
    void decrement_barrier(uint32_t rank) {
        simple_barrier &barrier = simple_barrier::get_local_barrier();
        DEBUG_OUTPUT(1, "Decrement barrier from rank " << rank << " " << barrier.process_count_);
        barrier.process_count_--;
        DEBUG_OUTPUT(1, "Decrement barrier from rank " << rank << " " << barrier.process_count_);
    }

    // this function will be a remote action
    void reset_barrier() {
        simple_barrier &barrier = simple_barrier::get_local_barrier();
        barrier.process_count_ = 0;
        DEBUG_OUTPUT(1, "Reset barrier rank " << barrier.rank_ << " " << barrier.process_count_);
    }

    // this function will be a remote action
    static void init_barrier() {
        simple_barrier &barrier = simple_barrier::get_local_barrier();
        barrier.init();
        DEBUG_OUTPUT(1, "Init barrier rank " << barrier.rank_ << " " << barrier.process_count_);
    }

} // namespace storage

//----------------------------------------------------------------------------
HPX_DEFINE_PLAIN_ACTION(Storage::decrement_barrier, DecrementBarrier_action);
HPX_DEFINE_PLAIN_ACTION(Storage::reset_barrier, ResetBarrier_action);
HPX_REGISTER_ACTION_DECLARATION(DecrementBarrier_action);
HPX_REGISTER_ACTION_DECLARATION(ResetBarrier_action);

// normally these are in a header
HPX_DEFINE_PLAIN_ACTION(Storage::CopyToStorage, CopyToStorage_action);
HPX_REGISTER_ACTION_DECLARATION(CopyToStorage_action);

HPX_DEFINE_PLAIN_ACTION(Storage::CopyFromStorage, CopyFromStorage_action);
//HPX_REGISTER_ACTION_DECLARATION(CopyFromStorage_action);

// and these in a cpp
HPX_REGISTER_ACTION(CopyToStorage_action);
HPX_REGISTER_ACTION(CopyFromStorage_action);
HPX_REGISTER_ACTION(DecrementBarrier_action);
HPX_REGISTER_ACTION(ResetBarrier_action);

void simple_barrier_count_down() {
    Storage::simple_barrier &barrier = Storage::simple_barrier::get_local_barrier();
    if (barrier.rank_==0) {
        DEBUG_OUTPUT(1, "Decrement count " << barrier.rank_ << " " << barrier.process_count_);
        --barrier.process_count_;
        // wait until everyone has counted down
        DEBUG_OUTPUT(1, "Before yield A rank " << barrier.rank_ << " " << barrier.process_count_);
        hpx::util::yield_while( [&](){
            bool done = (barrier.process_count_==0);
            if (done) barrier.init();
            return !done;
        });
        DEBUG_OUTPUT(1, "After yield A rank " << barrier.rank_);
        ResetBarrier_action reset_action;
        std::vector<hpx::id_type> remotes = hpx::find_remote_localities();
        for (auto &&r : remotes) {
            hpx::async(reset_action, r);
        }
        DEBUG_OUTPUT(1, "After actions rank " << barrier.rank_);
    }
    else {
        DecrementBarrier_action barrier_action;
        hpx::async(barrier_action, barrier.agas_, barrier.rank_);
        // now wait until our barrier is reset to zero
        DEBUG_OUTPUT(1, "Before yield B rank " << barrier.rank_ << " " << barrier.process_count_);
        hpx::util::yield_while( [&](){
            // once the condition is met, reinitialize the barrier before continuing
            // this ensures that the barrier is always ready for the next entrance
            // regardless of order of arrival of processes
            bool done = (barrier.process_count_==0);
            if (done) barrier.init();
            return !done;
        } );
        DEBUG_OUTPUT(1, "After yield B rank " << barrier.rank_);
    }
}

//----------------------------------------------------------------------------
static std::atomic<int> in_flight;

//----------------------------------------------------------------------------
// Test speed of write/put
void test_write(
    uint64_t rank, uint64_t nranks, uint64_t num_transfer_slots,
    std::mt19937& gen, std::uniform_int_distribution<uint64_t>& random_rank,
    std::uniform_int_distribution<uint64_t>& random_slot,
    test_options &options
    )
{
    CopyToStorage_action actWrite;
    //
    Storage::simple_barrier storage_barrier();
    DEBUG_OUTPUT(1, "Entering Barrier at start of write on rank " << rank);
//    hpx::lcos::barrier b1("b1_write");
//    hpx::lcos::barrier b2("b2_write");
    simple_barrier_count_down();
//    // block at the barrier b1
//    b1.wait(hpx::launch::async).get();
    DEBUG_OUTPUT(1, "Passed Barrier at start of write on rank " << rank);
    //
    hpx::util::high_resolution_timer timerWrite;
    hpx::util::simple_profiler level1("Write function", rank==0 && !options.warmup);
    //
    // used to track callbacks to free buffers for async_cb
    uint64_t buffer_index = 0;
    in_flight = 0;
    bool active = (rank==0) || (rank>0 && options.all2all);
    if (rank==0) std::cout << "Iteration ";
    for (std::uint64_t i = 0; active && i < options.iterations; i++) {
        hpx::util::simple_profiler iteration(level1, "Iteration");
        if (rank==0) {
            if (i%10==0)  {
                std::cout << "x" << std::flush;
            }
            else {
                std::cout << "." << std::flush;
            }
        }

        DEBUG_OUTPUT(2, "Starting iteration " << i << " on rank " << rank);

        //
        // Start main message sending loop
        //
        for (uint64_t i = 0; i < num_transfer_slots; i++) {
            hpx::util::simple_profiler prof_setup(iteration, "Setup slots");
            uint64_t send_rank;
            if (options.distribution==0) {
              // pick a random locality to send to
              send_rank = random_rank(gen);
              while (options.nolocal && send_rank==rank) {
                  send_rank = random_rank(gen);
              }
            }
            else {
              send_rank = static_cast<uint64_t>((rank + i) % nranks);
              while (options.nolocal && send_rank==rank) {
                  send_rank = random_rank(gen);
              }
            }

            // get the pointer to the current packet send buffer
            char *buffer = &local_storage[i*options.transfer_size_B];
            // Get the HPX locality from the dest rank
            hpx::id_type locality = hpx::naming::get_id_from_locality_id(send_rank);
            // pick a random slot to write our data into
            int memory_slot = random_slot(gen);
            uint32_t memory_offset = static_cast<uint32_t>
                (memory_slot*options.transfer_size_B);
            DEBUG_OUTPUT(5,
                "Rank " << rank << " sending block " << i << " to rank " << send_rank
            );
            prof_setup.done();

            // Execute a PUT on whatever locality we chose
            // Create a serializable memory buffer ready for sending (zero copy on send).
            {
                hpx::util::simple_profiler prof_put(iteration, "Put");
                DEBUG_OUTPUT(5,
                    "Put from rank " << rank << " on rank " << send_rank
                );

                std::shared_ptr<general_buffer_type> temp_buffer =
                    std::make_shared<general_buffer_type>(
                        static_cast<char*>(buffer), options.transfer_size_B,
                        general_buffer_type::reference);
                using hpx::util::placeholders::_1;
                using hpx::util::placeholders::_2;

                buffer_deleter keep_alive{buffer_index, temp_buffer};
                auto temp_future =
                    hpx::async_cb(hpx::launch::fork, actWrite, locality,
                            hpx::util::bind(&async_callback, keep_alive, _1, _2),
                            *temp_buffer,
                            memory_offset, options.transfer_size_B
                    ).then(
                        hpx::launch::sync,
                        [send_rank](hpx::future<int> &&fut) -> int {
                            int result = fut.get();
                            // decrement counter of tasks in flight
                            --FuturesWaiting[send_rank];
                            --in_flight;
                            return result;
                        });
                buffer_index++;

                // increment counter of tasks in flight
                ++in_flight;
            }
        }
//        simple_barrier_count_down();
        DEBUG_OUTPUT(3, "Completed iteration " << i << " on rank " << rank);
    }

    hpx::util::yield_while(
        [](){
            return in_flight.load(std::memory_order_relaxed)>0;
        }
    );

    if (rank==0) std::cout << std::endl;
    DEBUG_OUTPUT(2, "Exited iterations loop on rank " << rank);

    hpx::util::simple_profiler prof_barrier(level1, "Final Barrier");
    DEBUG_OUTPUT(1, "Entering Barrier at end of write on rank " << rank);
//    b2.wait(hpx::launch::async).get();
    simple_barrier_count_down();
    DEBUG_OUTPUT(1, "Passed Barrier at end of write on rank " << rank);
    //
    uint64_t active_ranks = options.all2all ? nranks : 1;
    double writeMB   = static_cast<double>
        (active_ranks*options.local_storage_MB*options.iterations);
    double writeTime = timerWrite.elapsed();
    double writeBW   = writeMB / writeTime;
    double IOPS      = static_cast<double>(options.iterations*num_transfer_slots);
    double IOPs_s    = IOPS/writeTime;
    if (rank == 0) {
        std::cout << "Total time           : " << writeTime << "\n";
        std::cout << "Memory Transferred   : " << writeMB   << " MB\n";
        std::cout << "Number of local IOPs : " << IOPS      << "\n";
        std::cout << "IOPs/s (local)       : " << IOPs_s    << "\n";
        std::cout << "Aggregate BW Write   : " << writeBW   << " MB/s" << std::endl;
        // a complete set of results that our python matplotlib script will ingest
        char const* msg = "CSVData, write, network, "
            "{1}, ranks, {2}, threads, {3}, Memory, {4}, IOPsize, {5}, "
            "IOPS/s, {6}, BW(MB/s), {7}, ";
        if (!options.warmup) {
            hpx::util::format_to(std::cout, msg,
                options.network,
                nranks, options.threads, writeMB, options.transfer_size_B,
                IOPs_s, writeBW ) << std::endl;
        }
        std::cout << std::endl;
    }
}

//----------------------------------------------------------------------------
// Copy the data once into the destination buffer if the get() operation was
// entirely local (no data copies have been made so far).
static void transfer_data(general_buffer_type recv,
  hpx::future<transfer_buffer_type> &&f)
{
  transfer_buffer_type buffer(f.get());
//  if (buffer.data() != recv.data())
  {
    std::copy(buffer.data(), buffer.data() + buffer.size(), recv.data());
  }
/*
  else {
    DEBUG_OUTPUT(2, "Skipped copy due to matching pointers");
  }
  */
}

//----------------------------------------------------------------------------
// Test speed of read/get
void test_read(
    uint64_t rank, uint64_t nranks, uint64_t num_transfer_slots,
    std::mt19937& gen, std::uniform_int_distribution<uint64_t>& random_rank,
    std::uniform_int_distribution<uint64_t>& random_slot,
    test_options &options
    )
{
    CopyFromStorage_action actRead;
    //
    DEBUG_OUTPUT(1, "Entering Barrier at start of read on rank " << rank);
////    hpx::lcos::barrier::synchronize();
//    hpx::lcos::barrier b1("b1_read");
//    hpx::lcos::barrier b2("b2_read");
//    b1.wait(hpx::launch::async).get();
    simple_barrier_count_down();

    DEBUG_OUTPUT(1, "Passed Barrier at start of read on rank " << rank);
    //
    // this is mostly the same as the put loop, except that the received future
    // is not an int, but a transfer buffer which we have to copy out of.
    //
    hpx::util::high_resolution_timer timerRead;
    //
    in_flight = 0;
    if (rank==0) std::cout << "Iteration ";
    bool active = (rank==0) || (rank>0 && options.all2all);
    for (std::uint64_t i = 0; active && i < options.iterations; i++) {
        if (rank==0) {
            if (i%10==0)  {
                std::cout << "x" << std::flush;
            }
            else {
                std::cout << "." << std::flush;
            }
        }

        DEBUG_OUTPUT(2, "Starting iteration " << i << " on rank " << rank);
        //
        // Start main message sending loop
        //
        //
        for (uint64_t i = 0; i < num_transfer_slots; i++) {
            hpx::util::high_resolution_timer looptimer;
            uint64_t send_rank;
            if (options.distribution==0) {
              // pick a random locality to send to
              send_rank = random_rank(gen);
              while (options.nolocal && send_rank==rank) {
                  send_rank = random_rank(gen);
              }
            }
            else {
              send_rank = static_cast<uint64_t>((rank + i) % nranks);
              while (options.nolocal && send_rank==rank) {
                  send_rank = random_rank(gen);
              }
            }

            // get the pointer to the current packet send buffer
//            char *buffer = &local_storage[i*options.transfer_size_B];

            // Get the HPX locality from the dest rank
            hpx::id_type locality = hpx::naming::get_id_from_locality_id(send_rank);

            // pick a random slot to write our data into
            int memory_slot = random_slot(gen);
            uint32_t memory_offset =
                static_cast<uint32_t>(memory_slot*options.transfer_size_B);

            // create a transfer buffer object to receive the data being returned to us
            general_buffer_type general_buffer(&local_storage[memory_offset],
              options.transfer_size_B, general_buffer_type::reference);

            // Execute a GET on whatever locality we chose
            // We pass the pointer to our local memory in the PUT, and it is used
            // by the serialization routines so that the copy from parcelport memory
            // is performed directly into our user memory. This avoids the need
            // to copy the data from a serialization buffer into our memory
            {
                using hpx::util::placeholders::_1;
                std::size_t buffer_address =
                    reinterpret_cast<std::size_t>(general_buffer.data());
                //
                auto temp_future =
                    hpx::async(
                        hpx::launch::fork, actRead, locality, memory_offset,
                        options.transfer_size_B, buffer_address
                    ).then(
                        hpx::launch::sync,
                        [=](hpx::future<transfer_buffer_type> &&buffer) -> void {
                            return transfer_data(general_buffer, std::move(buffer));
                        }
                    ).then(
                        hpx::launch::sync,
                        [=](hpx::future<void> fut) -> int {
                            // Retrieve the serialized data buffer that was
                            // returned from the action
                            // try to minimize copies by receiving into our
                            // custom buffer
                            fut.get();
                            --FuturesWaiting[send_rank];
                            --in_flight;
                            return TEST_SUCCESS;
                        }
                    );

                // increment counter of tasks in flight
                ++in_flight;
            }
        }
//        simple_barrier_count_down();
        DEBUG_OUTPUT(3, "Completed iteration " << i << " on rank " << rank);
    }

    hpx::util::yield_while(
        [](){
            return in_flight.load(std::memory_order_relaxed)>0;
        }
    );

    if (rank==0) std::cout << std::endl;
    DEBUG_OUTPUT(2, "Exited iterations loop on rank " << rank);

    DEBUG_OUTPUT(1, "Entering Barrier at end of read on rank " << rank);
    simple_barrier_count_down();
    DEBUG_OUTPUT(1, "Passed Barrier at end of read on rank " << rank);
    //
    if (rank==0) std::cout << std::endl;
    //
    uint64_t active_ranks = options.all2all ? nranks : 1;
    double readMB   = static_cast<double>
        (active_ranks*options.local_storage_MB*options.iterations);
    double readTime = timerRead.elapsed();
    double readBW = readMB / readTime;
    double IOPS      = static_cast<double>(options.iterations*num_transfer_slots);
    double IOPs_s    = IOPS/readTime;
    if (rank == 0) {
        std::cout << "Total time           : " << readTime << "\n";
        std::cout << "Memory Transferred   : " << readMB << " MB \n";
        std::cout << "Number of local IOPs : " << IOPS      << "\n";
        std::cout << "IOPs/s (local)       : " << IOPs_s    << "\n";
        std::cout << "Aggregate BW Read    : " << readBW << " MB/s" << std::endl;
        // a complete set of results that our python matplotlib script will ingest
        char const* msg = "CSVData, read, network, {1}, ranks, "
            "{2}, threads, {3}, Memory, {4}, IOPsize, {5}, IOPS/s, {6}, "
            "BW(MB/s), {7}, ";
        hpx::util::format_to(std::cout, msg, options.network, nranks,
            options.threads, readMB, options.transfer_size_B,
            IOPs_s, readBW) << std::endl;
    }
}

//----------------------------------------------------------------------------
// Main test loop which randomly sends packets of data from one locality to
// another looping over the entire buffer address space and timing the total
// transmit/receive time to see how well we're doing.
int hpx_main(boost::program_options::variables_map& vm)
{
    hpx::util::high_resolution_timer timer_main;
    DEBUG_OUTPUT(3,"HPX main");
    //
    hpx::id_type                    here = hpx::find_here();
    uint64_t                        rank = hpx::naming::get_locality_id_from_id(here);
    std::string                     name = hpx::get_locality_name();
    uint64_t                      nranks = hpx::get_num_localities().get();
    std::size_t                  current = hpx::get_worker_thread_num();
    std::vector<hpx::id_type>    remotes = hpx::find_remote_localities();
    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    //
    if (nranks>MAX_RANKS) {
      std::cerr << "This test can only be run using " << MAX_RANKS
        << " nodes, please recompile this test with the"
           " MAX_RANKS set to a higher number " << std::endl;
      return 1;
    }

    if (rank==0) {
        char const* msg = "hello world from OS-thread {:02} on locality "
            "{:04} rank {:04} hostname {}";
        hpx::util::format_to(std::cout, msg, current, hpx::get_locality_id(),
            rank, name.c_str()) << std::endl;
    }

    // extract command line argument
    test_options options;
    options.transfer_size_B   = vm["transferKB"].as<std::uint64_t>() * 1024;
    options.local_storage_MB  = vm["localMB"].as<std::uint64_t>();
    options.global_storage_MB = vm["globalMB"].as<std::uint64_t>();
    options.iterations        = vm["iterations"].as<std::uint64_t>();
    options.threads           = hpx::get_os_thread_count();
    options.network           = vm["parceltype"].as<std::string>();
    options.all2all           = vm["all-to-all"].as<bool>();
    options.distribution      = vm["distribution"].as<std::uint64_t>() ? true : false;
    options.nolocal           = vm["no-local"].as<bool>();
    options.semaphore         = vm["semaphore"].as<std::uint64_t>();
    options.warmup            = false;

    //
    if (options.global_storage_MB>0) {
      options.local_storage_MB = options.global_storage_MB/nranks;
    }

    DEBUG_OUTPUT(2, "Allocating local storage on rank " << rank);
    allocate_local_storage(options.local_storage_MB*1024*1024);
    //
    uint64_t num_transfer_slots = 1024*1024*options.local_storage_MB
        / options.transfer_size_B;
    DEBUG_OUTPUT(1,
        "num ranks " << nranks << ", num_transfer_slots "
        << num_transfer_slots << " on rank " << rank
    );
    //
    if (options.nolocal && nranks==1) {
        std::cout << "Fatal error, cannot use nolocal with a single rank" << std::endl;
        std::terminate();
    }
    //
    std::mt19937 gen;
    std::uniform_int_distribution<uint64_t> random_rank(0, (int)nranks - 1);
    std::uniform_int_distribution<uint64_t> random_slot(0,
        (int)num_transfer_slots - 1);
    //
    for (uint64_t i = 0; i < nranks; i++) {
        FuturesWaiting[i].store(0);
    }

    DEBUG_OUTPUT(1, "Initialize barrier before first use on rank " << rank);
    Storage::init_barrier();
    DEBUG_OUTPUT(1, "Entering startup_barrier on rank " << rank);
//    hpx::lcos::barrier start_barrier("startup_barrier");
//    hpx::lcos::barrier end_barrier("end_barrier");
//    start_barrier.wait(hpx::launch::async).get();
    simple_barrier_count_down();
    DEBUG_OUTPUT(1, "Passed startup_barrier on rank " << rank);

    test_options warmup = options;
    warmup.iterations = 1;
    warmup.warmup = true;
    test_write(rank, nranks, num_transfer_slots, gen, random_rank, random_slot, warmup);
    //
    test_write(rank, nranks, num_transfer_slots, gen, random_rank, random_slot, options);
    test_read (rank, nranks, num_transfer_slots, gen, random_rank, random_slot, options);
    //
    DEBUG_OUTPUT(1, "Entering end_barrier on rank " << rank);
//    end_barrier.wait(hpx::launch::async).get();
    simple_barrier_count_down();
    DEBUG_OUTPUT(1, "Passed end_barrier on rank " << rank);

    delete_local_storage();

    if (rank==0) {
        double s = timer_main.elapsed();
        double m = std::floor(s/60.0);
        s = std::round((s - (m*60.0))*10.0)/10.0;
        std::cout << "Total test time " << m << ":";
        std::cout << std::setfill('0') << std::setw(4) << std::noshowbase << std::dec;
        std::cout.precision(3);
        std::cout << s << std::endl;
        DEBUG_OUTPUT(2, "Calling finalize " << rank);
        return hpx::finalize();
    }
    else return 0;
}

//----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    // Configure application-specific options
    boost::program_options::options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ( "localMB",
          boost::program_options::value<std::uint64_t>()->default_value(256),
          "Sets the storage capacity (in MB) on each node.\n"
          "The total storage will be num_ranks * localMB")
        ;

    desc_commandline.add_options()
        ( "globalMB",
          boost::program_options::value<std::uint64_t>()->default_value(0),
          "Sets the storage capacity (in MB) for the entire job.\n"
          "The storage per node will be globalMB / num_ranks\n"
          "By default, localMB is used, setting this overrides localMB value."
          )
        ;

    desc_commandline.add_options()
        ( "transferKB",
          boost::program_options::value<std::uint64_t>()->default_value(64),
          "Sets the default block transfer size (in KB).\n"
          "Each put/get IOP will be this size")
        ;

    desc_commandline.add_options()
        ( "semaphore",
          boost::program_options::value<std::uint64_t>()->default_value(16),
          "The max amount of simultaneous put/get operations to allow.\n")
        ;

    desc_commandline.add_options()
        ( "iterations",
          boost::program_options::value<std::uint64_t>()->default_value(5),
          "The number of iterations over the global memory.\n")
        ;

    desc_commandline.add_options()
        ( "all-to-all",
          boost::program_options::value<bool>()->default_value(true),
          "When set, all ranks send to all others, when off, "
          "only rank 0 sends to the others.\n")
        ;

    desc_commandline.add_options()
        ( "no-local",
          boost::program_options::value<bool>()->default_value(false),
          "When set, non local transfers are made, "
          "ranks send to the others but not to themselves.\n")
        ;

    // if the user does not set parceltype on the command line,
    // we use a default of unknowm so we don't mistake plots
    desc_commandline.add_options()
        ( "parceltype",
          boost::program_options::value<std::string>()->default_value("unknown"),
          "Pass in the parcelport network type being tested,"
          "this value has no effect on the code and is used only in output "
          "so that plotting scripts know what network type was active during \
            the test.\n")
        ;

    desc_commandline.add_options()
        ( "distribution",
          boost::program_options::value<std::uint64_t>()->default_value(1),
          "Specify the distribution of data blocks to send/receive,\n"
          "in random mode, blocks of data are sent from one rank to \
            any other rank (including itself),"
          "in block-cyclic mode, blocks of data are sent from one rank to \
            other ranks in block-cyclic order,"
          "0 : random \n"
          "1 : block cyclic")
        ;

    // Initialize and run HPX, this test requires to run hpx_main on all localities
    std::vector<std::string> const cfg = {
        "hpx.run_hpx_main!=1"
    };

    DEBUG_OUTPUT(3,"Calling hpx::init");
    return hpx::init(desc_commandline, argc, argv, cfg);
}
