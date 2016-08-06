//  Copyright (c) 2014 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>

#include <boost/array.hpp>
#include <boost/assert.hpp>
#include <boost/atomic.hpp>
#include <boost/random.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <hpx/runtime/serialization/serialize.hpp>

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
// These are used to track how many requests are pending for each locality
// Many requests for read/write of memory may be issued at a time, resulting
// in thousands of futures that need to be waited on. To reduce this number
// a background thread can be spawned to check for ready futures and remove
// them from the waiting list. The vars are used for this bookkeeping task.
//
#define MAX_RANKS 64

#define USE_CLEANING_THREAD

std::vector<std::vector<hpx::future<int> > > ActiveFutures;
boost::array<boost::atomic<int>, 64>       FuturesWaiting;

#ifdef USE_CLEANING_THREAD
boost::atomic<bool>                        FuturesActive;
hpx::lcos::local::spinlock                 FuturesMutex;
#endif

//----------------------------------------------------------------------------
// Used at start and end of each loop for synchronization
hpx::lcos::barrier unique_barrier;

//----------------------------------------------------------------------------
//
// Each locality allocates a buffer of memory which is used to host transfers
//
char                      *local_storage = nullptr;
hpx::lcos::local::spinlock storage_mutex;

//
typedef struct {
    boost::uint64_t iterations;
    boost::uint64_t local_storage_MB;
    boost::uint64_t global_storage_MB;
    boost::uint64_t transfer_size_B;
    boost::uint64_t threads;
    std::string     network;
    bool            all2all;
    bool            distribution;
} test_options;

//----------------------------------------------------------------------------
#define DEBUG_LEVEL 0
#define DEBUG_OUTPUT(level,x)                                                \
    if (DEBUG_LEVEL>=level) {                                                \
        x                                                                    \
    }
//
#define TEST_FAIL    0
#define TEST_SUCCESS 1

//----------------------------------------------------------------------------
void allocate_local_storage(uint64_t local_storage_bytes)
{
    local_storage = new char[local_storage_bytes];
}

//----------------------------------------------------------------------------
void delete_local_storage()
{
    delete[] local_storage;
}

//----------------------------------------------------------------------------
void release_storage_lock()
{
  storage_mutex.unlock();
}

//----------------------------------------------------------------------------
// This routine simply copies from the source buffer into the local memory
// at the address offset.
//
// The function does not need to be asynchronous as it completes immediately,
// but we return a future as this test needs to mimic "asynchronous" storage
hpx::future<int> copy_to_local_storage(char const* src, uint32_t offset, uint64_t length)
{
    char *dest = &local_storage[offset];
    std::copy(src, src + length, dest);
    //  memcpy(dest, src, length);
    return hpx::make_ready_future<int>(TEST_SUCCESS);
}

//----------------------------------------------------------------------------
// This routine simply copies from local memory at the address offset
// into a provided buffer
//
// The function does not need to be asynchronous as it completes immediately,
// but we return a future as this test needs to mimic "asynchronous" storage
hpx::future<int> copy_from_local_storage(char *dest, uint32_t offset, uint64_t length)
{
    char const* src = &local_storage[offset];
    std::copy(src, src + length, dest);
    //  memcpy(dest, src, length);
    return hpx::make_ready_future<int>(TEST_SUCCESS);
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

  pointer_allocator() HPX_NOEXCEPT
    : pointer_(nullptr), size_(0)
  {
  }

  pointer_allocator(pointer p, size_type size) HPX_NOEXCEPT
    : pointer_(p), size_(size)
  {
  }

  pointer address(reference value) const { return &value; }
  const_pointer address(const_reference value) const { return &value; }

  pointer allocate(size_type n, void const* hint = nullptr)
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
  void load(Archive& ar, unsigned int const version)
  {
    std::size_t t = 0;
    ar >> size_ >> t;
    pointer_ = reinterpret_cast<pointer>(t);
  }

  template <typename Archive>
  void save(Archive& ar, unsigned int const version) const
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
typedef pointer_allocator<char>                             PointerAllocator;
typedef hpx::serialization::serialize_buffer<char,
    PointerAllocator> transfer_buffer_type;

//----------------------------------------------------------------------------
//
// Two actions which are called from remote or local localities
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
    hpx::future<int> CopyToStorage(general_buffer_type const& srcbuffer,
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
        // not be released by the shared_pointer
        std::allocator<char> local_allocator;
        boost::shared_array<char> local_buffer(local_allocator.allocate(length),
            [](char*){});

        // allow the storage class to asynchronously copy the data into buffer
        hpx::future<int> fut = copy_from_local_storage(local_buffer.get(),
            address, length);

        // wrap the remote buffer pointer in an allocator for return
        pointer_allocator<char> return_allocator(
          reinterpret_cast<char*>(remote_buffer), length);

        // lock the mutex, will be unlocked by the transfer buffer's deleter
        storage_mutex.lock();

        return fut.then(
//            hpx::launch::async,
            // return the data in a transfer buffer
            [=](hpx::future<int> fut) -> transfer_buffer_type {
                return transfer_buffer_type(
                    local_buffer.get(), length,
                    transfer_buffer_type::take,
                    hpx::util::bind(&release_storage_lock),
                    return_allocator);
            }
        );

    }
} // namespace storage

//----------------------------------------------------------------------------
// normally these are in a header
HPX_DEFINE_PLAIN_ACTION(Storage::CopyToStorage, CopyToStorage_action);
HPX_REGISTER_ACTION_DECLARATION(CopyToStorage_action);
//HPX_ACTION_INVOKE_NO_MORE_THAN(CopyToStorage_action, 5);

HPX_DEFINE_PLAIN_ACTION(Storage::CopyFromStorage, CopyFromStorage_action);
HPX_REGISTER_ACTION_DECLARATION(CopyFromStorage_action);
HPX_ACTION_INVOKE_NO_MORE_THAN(CopyFromStorage_action, 5);

// and these in a cpp
HPX_REGISTER_ACTION(CopyToStorage_action);
HPX_REGISTER_ACTION(CopyFromStorage_action);

//----------------------------------------------------------------------------
#ifdef USE_CLEANING_THREAD
// the main message sending loop may generate many thousands of send requests
// and each is associated with a future. To reduce the number we must wait on
// this loop runs in a background thread and simply removes any completed futures
// from the main list of active futures.
int RemoveCompletions()
{
    int num_removed = 0;
    while(FuturesActive)
    {
        {
            std::lock_guard<hpx::lcos::local::spinlock> lk(FuturesMutex);
            for(std::vector<hpx::future<int> > &futvec : ActiveFutures) {
                for(std::vector<hpx::future<int> >::iterator fut = futvec.begin();
                    fut != futvec.end(); /**/)
                {
                    if(fut->is_ready()){
                        int ret = fut->get();
                        if(ret != TEST_SUCCESS) {
                            throw std::runtime_error("Remote put/get failed");
                        }
                        num_removed++;
                        fut = futvec.erase(fut);
                    } else {
                        ++fut;
                    }
                }
            }
        }
        hpx::this_thread::suspend(std::chrono::microseconds(10));
    }
    return num_removed;
}
#endif

//----------------------------------------------------------------------------
// Take a vector of futures representing pass/fail and reduce to a single pass fail
int reduce(hpx::future<std::vector<hpx::future<int> > > futvec)
{
    int res = TEST_SUCCESS;
    std::vector<hpx::future<int> > vfs = futvec.get();
    for(hpx::future<int>& f : vfs) {
        if(f.get() == TEST_FAIL) return TEST_FAIL;
    }
    return res;
}

//----------------------------------------------------------------------------
// Create a new barrier and register its gid with the given symbolic name.
hpx::lcos::barrier create_barrier(std::size_t num_localities, char const* symname)
{
    DEBUG_OUTPUT(2,
        std::cout << "Creating barrier based on N localities "
                  << num_localities << std::endl;
    );

    hpx::lcos::barrier b = hpx::lcos::barrier::create(hpx::find_here(), num_localities);
    hpx::agas::register_name_sync(symname, b.get_id());
    return b;
}

//----------------------------------------------------------------------------
void barrier_wait()
{
    unique_barrier.wait();
}

//----------------------------------------------------------------------------
// Test speed of write/put
void test_write(
    uint64_t rank, uint64_t nranks, uint64_t num_transfer_slots,
    boost::random::mt19937& gen, boost::random::uniform_int_distribution<>& random_rank,
    boost::random::uniform_int_distribution<>& random_slot,
    test_options &options
    )
{
    CopyToStorage_action actWrite;
    //
    DEBUG_OUTPUT(1,
        std::cout << "Entering Barrier at start of write on rank " << rank << std::endl;
    );
    //
    barrier_wait();
    //
    DEBUG_OUTPUT(1,
        std::cout << "Passed Barrier at start of write on rank " << rank << std::endl;
    );
    //
    hpx::util::high_resolution_timer timerWrite;

    bool active = (rank==0) | (rank>0 && options.all2all);
    for(boost::uint64_t i = 0; active && i < options.iterations; i++) {
        DEBUG_OUTPUT(1,
            std::cout << "Starting iteration " << i << " on rank " << rank << std::endl;
        );
#ifdef USE_CLEANING_THREAD
        //
        // start a thread which will clear any completed futures from our list.
        //
        FuturesActive = true;
        hpx::future<int> cleaner = hpx::async(RemoveCompletions);
#endif
        //
        // Start main message sending loop
        //
        for(uint64_t i = 0; i < num_transfer_slots; i++) {
            hpx::util::high_resolution_timer looptimer;
            int send_rank;
            if(options.distribution==0) {
              // pick a random locality to send to
              send_rank = random_rank(gen);
            }
            else {
              send_rank = static_cast<int>(i % nranks);
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
                std::cout << "Rank " << rank << " sending block "
                << i << " to rank " << send_rank << std::endl;
            );

            // Execute a PUT on whatever locality we chose
            // Create a serializable memory buffer ready for sending.
            // Do not copy any data. Protect this with a mutex to ensure the
            // background thread removing completed futures doesn't collide
            {
                DEBUG_OUTPUT(5,
                    std::cout << "Put from rank " << rank << " on rank "
                    << send_rank << std::endl;
                );
#ifdef USE_CLEANING_THREAD
                ++FuturesWaiting[send_rank];
                std::lock_guard<hpx::lcos::local::spinlock> lk(FuturesMutex);
#endif
                ActiveFutures[send_rank].push_back(
                    hpx::async(actWrite, locality,
                        general_buffer_type(static_cast<char*>(buffer),
                            options.transfer_size_B, general_buffer_type::reference),
                        memory_offset, options.transfer_size_B
                    ).then(
                        hpx::launch::sync,
                        [send_rank](hpx::future<int> &&fut) -> int {
                            int result = fut.get();
                            --FuturesWaiting[send_rank];
                            return result;
                        })
                );
            }
            DEBUG_OUTPUT(5,
                std::cout << "Loop timer " << looptimer.elapsed() << std::endl;
            );
        }
        int removed = 0;
#ifdef USE_CLEANING_THREAD
        // tell the cleaning thread it's time to stop
        FuturesActive = false;
        // wait for cleanup thread to terminate before we reduce any remaining futures
        removed = cleaner.get();
        DEBUG_OUTPUT(2,
            std::cout << "Cleaning thread rank " << rank << " removed "
            << removed << std::endl;
        );
#endif
        //
        hpx::util::high_resolution_timer movetimer;
        std::vector<hpx::future<int> > final_list;
        for(uint64_t i = 0; i < nranks; i++) {
            // move the contents of intermediate vector into final list
            final_list.reserve(final_list.size() + ActiveFutures[i].size());
            std::move(ActiveFutures[i].begin(), ActiveFutures[i].end(),
                std::back_inserter(final_list));
            ActiveFutures[i].clear();
        }
        double movetime = movetimer.elapsed();
        //
        int numwait = static_cast<int>(final_list.size());
        hpx::util::high_resolution_timer futuretimer;
        hpx::future<int> result = when_all(final_list).then(hpx::launch::sync, reduce);
        result.get();
        int total = numwait+removed;
        DEBUG_OUTPUT(3,
            std::cout << "Future timer, rank " << rank
            << " waiting on " << numwait << " total " << total << " "
            << futuretimer.elapsed() << " Move time " << movetime << std::endl;
        );
    }
    barrier_wait();
    //
    double writeMB   = static_cast<double>
        (nranks*options.local_storage_MB*options.iterations);
    double writeTime = timerWrite.elapsed();
    double writeBW   = writeMB / writeTime;
    double IOPS      = static_cast<double>(options.iterations*num_transfer_slots);
    double IOPs_s    = IOPS/writeTime;
    if(rank == 0) {
        std::cout << "Total time         : " << writeTime << "\n";
        std::cout << "Memory Transferred : " << writeMB   << "MB\n";
        std::cout << "Number of IOPs     : " << IOPS      << "\n";
        std::cout << "IOPs/s             : " << IOPs_s    << "\n";
        std::cout << "Aggregate BW Write : " << writeBW   << "MB/s" << std::endl;
        // a complete set of results that our python matplotlib script will ingest
        char const* msg = "CSVData, write, network, \
            %1%, ranks, %2%, threads, %3%, Memory, %4%, IOPsize, %5%, \
            IOPS/s, %6%, BW(MB/s), %7%, ";
        std::cout << (boost::format(msg) % options.network
            % nranks % options.threads % writeMB % options.transfer_size_B
          % IOPs_s % writeBW ) << std::endl;
    }
}

//----------------------------------------------------------------------------
// Copy the data once into the destination buffer if the get() operation was
// entirely local (no data copies have been made so far).
static void transfer_data(general_buffer_type recv,
  hpx::future<transfer_buffer_type> f)
{
  transfer_buffer_type buffer(f.get());
  if (buffer.data() != recv.data())
  {
    std::copy(buffer.data(), buffer.data() + buffer.size(), recv.data());
  }
  else {
    DEBUG_OUTPUT(5,
      std::cout << "Skipped copy due to matching pointers" << std::endl;
    );
  }
}

//----------------------------------------------------------------------------
// Test speed of read/get
void test_read(
    uint64_t rank, uint64_t nranks, uint64_t num_transfer_slots,
    boost::random::mt19937& gen, boost::random::uniform_int_distribution<>& random_rank,
    boost::random::uniform_int_distribution<>& random_slot,
    test_options &options
    )
{
    CopyFromStorage_action actRead;
    //
    DEBUG_OUTPUT(1,
        std::cout << "Entering Barrier at start of read on rank " << rank << std::endl;
    );
    //
    barrier_wait();
    //
    DEBUG_OUTPUT(1,
        std::cout << "Passed Barrier at start of read on rank " << rank << std::endl;
    );
    //
    // this is mostly the same as the put loop, except that the received future
    // is not an int, but a transfer buffer which we have to copy out of.
    //
    hpx::util::high_resolution_timer timerRead;
    //
    bool active = (rank==0) || (rank>0 && options.all2all);
    for(boost::uint64_t i = 0; active && i < options.iterations; i++) {
      DEBUG_OUTPUT(1,
          std::cout << "Starting iteration " << i << " on rank " << rank << std::endl;
      );
#ifdef USE_CLEANING_THREAD
        //
        // start a thread which will clear any completed futures from our list.
        //
        FuturesActive = true;
        hpx::future<int> cleaner = hpx::async(RemoveCompletions);
#endif
        //
        // Start main message sending loop
        //
        for(uint64_t i = 0; i < num_transfer_slots; i++) {
            hpx::util::high_resolution_timer looptimer;
            int send_rank;
            if(options.distribution==0) {
              // pick a random locality to send to
              send_rank = random_rank(gen);
            }
            else {
              send_rank = static_cast<int>(i % nranks);
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
#ifdef USE_CLEANING_THREAD
                ++FuturesWaiting[send_rank];
                std::lock_guard<hpx::lcos::local::spinlock> lk(FuturesMutex);
#endif
                using hpx::util::placeholders::_1;
                std::size_t buffer_address =
                    reinterpret_cast<std::size_t>(general_buffer.data());
                //
                ActiveFutures[send_rank].push_back(
                    hpx::async(
                        actRead, locality, memory_offset,
                        options.transfer_size_B, buffer_address
                    ).then(
                        hpx::launch::sync,
                        hpx::util::bind(&transfer_data, general_buffer, _1)
                    ).then(
                        hpx::launch::sync,
                        [=](hpx::future<void> fut) -> int {
                            // Retrieve the serialized data buffer that was
                            // returned from the action
                            // try to minimize copies by receiving into our
                            // custom buffer
                            fut.get();
                            --FuturesWaiting[send_rank];
                            return TEST_SUCCESS;
                        })
                    );
            }
        }
#ifdef USE_CLEANING_THREAD
        // tell the cleaning thread it's time to stop
        FuturesActive = false;
        // wait for cleanup thread to terminate before we reduce any remaining
        // futures
        int removed = cleaner.get();
        DEBUG_OUTPUT(2,
            std::cout << "Cleaning thread rank " << rank << " removed "
            << removed << std::endl;
        );
#endif
        //
        hpx::util::high_resolution_timer movetimer;
        std::vector<hpx::future<int> > final_list;
        for(uint64_t i = 0; i < nranks; i++) {
            // move the contents of intermediate vector into final list
            final_list.reserve(final_list.size() + ActiveFutures[i].size());
            std::move(ActiveFutures[i].begin(), ActiveFutures[i].end(),
                std::back_inserter(final_list));
            ActiveFutures[i].clear();
        }
        double movetime = movetimer.elapsed();
        //
        int numwait = static_cast<int>(final_list.size());
        hpx::util::high_resolution_timer futuretimer;
        hpx::future<int> result = when_all(final_list).then(hpx::launch::sync, reduce);
        result.get();
        int total = numwait+removed;
        DEBUG_OUTPUT(3,
            std::cout << "Future timer, rank " << rank << " waiting on "
            << numwait << " total " << total << " "
            << futuretimer.elapsed() << " Move time " << movetime << std::endl;
        );
    }
    barrier_wait();
    //
    double readMB = static_cast<double>
        (nranks*options.local_storage_MB*options.iterations);
    double readTime = timerRead.elapsed();
    double readBW = readMB / readTime;
    double IOPS      = static_cast<double>(options.iterations*num_transfer_slots);
    double IOPs_s    = IOPS/readTime;
    if(rank == 0) {
        std::cout << "Total time         : " << readTime << "\n";
        std::cout << "Memory Transferred : " << readMB << "MB \n";
        std::cout << "Number of IOPs     : " << IOPS      << "\n";
        std::cout << "IOPs/s             : " << IOPs_s    << "\n";
        std::cout << "Aggregate BW Read  : " << readBW << "MB/s" << std::endl;
        // a complete set of results that our python matplotlib script will ingest
        char const* msg = "CSVData, read, network, %1%, ranks, \
          %2%, threads, %3%, Memory, %4%, IOPsize, %5%, IOPS/s, %6%, \
            BW(MB/s), %7%, ";
        std::cout << (boost::format(msg) % options.network % nranks
            % options.threads % readMB % options.transfer_size_B
          % IOPs_s % readBW ) << std::endl;
    }
}

//----------------------------------------------------------------------------
void create_barrier_startup()
{
    hpx::id_type here = hpx::find_here();
    uint64_t rank = hpx::naming::get_locality_id_from_id(here);

    // create a barrier we will use at the start and end of each run to
    // synchronize
    if(0 == rank) {
        uint64_t nranks = hpx::get_num_localities().get();
        unique_barrier = create_barrier(nranks, "/0/DSM_barrier");
    }
}

//----------------------------------------------------------------------------
void find_barrier_startup()
{
    hpx::id_type here = hpx::find_here();
    uint64_t rank = hpx::naming::get_locality_id_from_id(here);

    if (rank != 0) {
        hpx::id_type id = hpx::agas::resolve_name_sync("/0/DSM_barrier");
        unique_barrier = hpx::lcos::barrier(id);
    }
}

//----------------------------------------------------------------------------
// Main test loop which randomly sends packets of data from one locality to
// another looping over the entire buffer address space and timing the total
// transmit/receive time to see how well we're doing.
int hpx_main(boost::program_options::variables_map& vm)
{
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
        << " nodes, please recompile this test with the \
              MAX_RANKS set to a higher number " << std::endl;
      return 1;
    }

    char const* msg = "hello world from OS-thread %1% on locality \
                        %2% rank %3% hostname %4%";
    std::cout << (boost::format(msg) % current % hpx::get_locality_id()
        % rank % name.c_str()) << std::endl;
    //
    // extract command line argument
    test_options options;
    options.transfer_size_B   = vm["transferKB"].as<boost::uint64_t>() * 1024;
    options.local_storage_MB  = vm["localMB"].as<boost::uint64_t>();
    options.global_storage_MB = vm["globalMB"].as<boost::uint64_t>();
    options.iterations        = vm["iterations"].as<boost::uint64_t>();
    options.threads           = hpx::get_os_thread_count();
    options.network           = vm["parceltype"].as<std::string>();
    options.all2all           = vm["all-to-all"].as<bool>();
    options.distribution      = vm["distribution"].as<boost::uint64_t>() ? true : false;

    //
    if (options.global_storage_MB>0) {
      options.local_storage_MB = options.global_storage_MB/nranks;
    }
    allocate_local_storage(options.local_storage_MB*1024*1024);
    //
    uint64_t num_transfer_slots = 1024*1024*options.local_storage_MB
        / options.transfer_size_B;
    DEBUG_OUTPUT(1,
        std::cout << "num ranks " << nranks << ", num_transfer_slots "
                  << num_transfer_slots << " on rank " << rank << std::endl;
    );
    //
    boost::random::mt19937 gen;
    boost::random::uniform_int_distribution<> random_rank(0, (int)nranks - 1);
    boost::random::uniform_int_distribution<> random_slot(0,
        (int)num_transfer_slots - 1);
    //
    ActiveFutures.reserve(nranks);
    for(uint64_t i = 0; i < nranks; i++) {
        FuturesWaiting[i].store(0);
        ActiveFutures.push_back(std::vector<hpx::future<int> >());
    }

    test_write(rank, nranks, num_transfer_slots, gen, random_rank, random_slot, options);
    test_read (rank, nranks, num_transfer_slots, gen, random_rank, random_slot, options);
    //
    delete_local_storage();

    // release barrier object
    unique_barrier = hpx::invalid_id;
    DEBUG_OUTPUT(2,
        std::cout << "Unregistering Barrier " << rank << std::endl;
    );
    if (0 == rank)
        hpx::agas::unregister_name_sync("/0/DSM_barrier");

    DEBUG_OUTPUT(2,
        std::cout << "Calling finalize" << rank << std::endl;
    );
    if (rank==0)
      return hpx::finalize();
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
          boost::program_options::value<boost::uint64_t>()->default_value(256),
          "Sets the storage capacity (in MB) on each node.\n"
          "The total storage will be num_ranks * localMB")
        ;

    desc_commandline.add_options()
        ( "globalMB",
          boost::program_options::value<boost::uint64_t>()->default_value(0),
          "Sets the storage capacity (in MB) for the entire job.\n"
          "The storage per node will be globalMB / num_ranks\n"
          "By default, localMB is used, setting this overrides localMB value."
          )
        ;

    desc_commandline.add_options()
        ( "transferKB",
          boost::program_options::value<boost::uint64_t>()->default_value(64),
          "Sets the default block transfer size (in KB).\n"
          "Each put/get IOP will be this size")
        ;

    desc_commandline.add_options()
        ( "iterations",
          boost::program_options::value<boost::uint64_t>()->default_value(5),
          "The number of iterations over the global memory.\n")
        ;

    desc_commandline.add_options()
        ( "all-to-all",
          boost::program_options::value<bool>()->default_value(true),
          "When set, all ranks send to all others, when off, \
                only rank 0 send to the others.\n")
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
          boost::program_options::value<boost::uint64_t>()->default_value(1),
          "Specify the distribution of data blocks to send/receive,\n"
          "in random mode, blocks of data are sent from one rank to \
            any other rank (including itself),"
          "in block-cyclic mode, blocks of data are sent from one rank to \
            other ranks in block-cyclic order,"
          "0 : random \n"
          "1 : block cyclic")
        ;

    // make sure our barrier was already created before hpx_main runs
    DEBUG_OUTPUT(2,
        std::cout << "Registering create_barrier startup function " << std::endl;
    );
    hpx::register_pre_startup_function(&create_barrier_startup);
    DEBUG_OUTPUT(2,
        std::cout << "Registering find_barrier startup function " << std::endl;
    );
    hpx::register_startup_function(&find_barrier_startup);

    // Initialize and run HPX, this test requires to run hpx_main on all localities
    std::vector<std::string> const cfg = {
        "hpx.run_hpx_main!=1"
    };

    return hpx::init(desc_commandline, argc, argv, cfg);
}


