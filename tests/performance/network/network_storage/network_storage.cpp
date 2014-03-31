//  Copyright (c) 2014 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>

#include <boost/assert.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/atomic.hpp>
#include <boost/array.hpp>

#include <algorithm>
#include <iostream>
#include <vector>
#include <memory>
#include <cstdio>
#include <random>

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
std::vector<std::vector<hpx::future<int>>> ActiveFutures;
hpx::lcos::local::spinlock                 FuturesMutex;
boost::atomic<bool>                        FuturesActive;
boost::array<boost::atomic<int>, 64>       FuturesWaiting;

//----------------------------------------------------------------------------
// Used at start and end of each loop for synchronization
hpx::lcos::barrier unique_barrier;

//----------------------------------------------------------------------------
//
// Each locality allocates a buffer of memory which is used to host transfers
//
char *local_storage = NULL;
//
typedef struct {
  boost::uint64_t iterations;
  boost::uint64_t local_storage_MB;
  boost::uint64_t global_storage_MB;
  boost::uint64_t transfer_size_B;
  boost::uint64_t threads;
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
// This routine simply copies from the source buffer into the local memory
// at the address offset.
//
// The function does not need to be asynchronous as it completes immediately,
// but we return a future as this test needs to mimic "asynchronous" storage
hpx::future<int> copy_to_local_storage(char const* src, uint32_t offset, int length)
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
hpx::future<int> copy_from_local_storage(char *dest, uint32_t offset, int length)
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
class AllocatorPointer : public std::allocator<T>
{
public:
    typedef T              value_type;
    typedef T*             pointer;
    typedef const T*       const_pointer;
    typedef T&             reference;
    typedef const T&       const_reference;
    typedef std::size_t    size_type;
    typedef std::ptrdiff_t difference_type;

    pointer the_pointer;

    AllocatorPointer() throw() {}

    pointer address(reference value) const { return &value; }
    const_pointer address(const_reference value) const { return &value; }

    pointer allocate(size_type n, const void *hint = 0)
    {
        return static_cast<T*>(the_pointer);
    }

    void deallocate(pointer p, size_type n) {}

    AllocatorPointer(pointer a_pointer) throw() : std::allocator<T>()
    {
        this->the_pointer = a_pointer;
    }
    AllocatorPointer(const AllocatorPointer &a) throw() : std::allocator<T>(a)
    {
        this->the_pointer = a.the_pointer;
    }

private:
    // serialization support
    friend class boost::serialization::access;

    template <typename Archive>
    void load(Archive& ar, const unsigned int version)
    {
        std::size_t t = 0;
        ar >> t;
        the_pointer = reinterpret_cast<pointer>(t);
    }

    template <typename Archive>
    void save(Archive& ar, const unsigned int version) const
    {
        std::size_t t = reinterpret_cast<std::size_t>(the_pointer);
        ar << t;
    }

    BOOST_SERIALIZATION_SPLIT_MEMBER()
};

//----------------------------------------------------------------------------
// A simple Buffer for sending data, it does not need any special allocator
// user data may be sent to another locality using zero copy by wrapping
// it in one of these buffers
typedef hpx::util::serialize_buffer<char, std::allocator<char>> TransferBuffer;
//
// When receiving data, we receive a hpx::serialize_buffer, we try to minimize
// copying of data by providing a receive buffer with a fixed data pointer
// so that data is placed directly into it.
// It doesn't produce any speedup, but is here to provide a basis for experimentation
typedef AllocatorPointer<char>                              PointerAllocator;
typedef hpx::util::serialize_buffer<char, PointerAllocator> SerializeToPointer;

//----------------------------------------------------------------------------
// The TransferBufferReceive provides a constructor which copied from the
// input buffer into the final memory location
class TransferBufferReceive : public SerializeToPointer
{
public:
    TransferBufferReceive() {}      // needed for serialization

    TransferBufferReceive(char* buffer, std::size_t length,
            std::size_t remote_buffer, std::allocator<char> deallocator) throw()
      : SerializeToPointer(buffer, length, SerializeToPointer::take,
            PointerAllocator(reinterpret_cast<PointerAllocator::pointer>(remote_buffer)),
            deallocator)
    {}

    ~TransferBufferReceive() {}
};

//----------------------------------------------------------------------------
//
// Two actions which are called from remote or local localities
//
// Copy to storage, just invokes the local copy function and returns the future from it
//
// Copy from storage allocates a buffer for the return memory and then wraps it into
// a serialize buffer which is returned and passed into the local process.
// This unfortunately means memory is copied from the storage, into a buffer
// and not zero copied as we would like.
// I have not been successful in removing this copy.
namespace Storage {
    //----------------------------------------------------------------------------
    // A PUT into memory on this locality from a requester sending a TransferBuffer
    hpx::future<int> CopyToStorage(TransferBuffer const& srcbuffer, uint32_t address, int length)
    {
        boost::shared_array<char> src = srcbuffer.data_array();
        return copy_to_local_storage(src.get(), address, length);
    }

    //----------------------------------------------------------------------------
    // A GET from memory on this locality is returned to the requester in the TransferBuffer
    hpx::future<TransferBufferReceive> CopyFromStorage(
        uint32_t address, int length, std::size_t remote_buffer)
    {
        // we must allocate a return buffer
        std::allocator<char> allocator;
        boost::shared_array<char> dest(allocator.allocate(length), [](char*){});

        // allow the storage class to asynchronously copy the data into dest buffer
        hpx::future<int> fut = copy_from_local_storage(dest.get(), address, length);

        // when the task completes, return a TransferBuffer
        return fut.then(hpx::launch::sync,
            [=](hpx::future<int> f) -> TransferBufferReceive {
                int success = f.get();
                if(success != TEST_SUCCESS) {
                    throw std::runtime_error("Fail in Get");
                }
                // return the result buffer in a serializable hpx TransferBuffer
                // tell the return buffer that it now owns the buffer using ::take mode
                return TransferBufferReceive(dest.get(), length, remote_buffer, allocator);
            }
        );
    }
} // namespace storage

//----------------------------------------------------------------------------
// normally these are in a header
HPX_DEFINE_PLAIN_ACTION(Storage::CopyToStorage, CopyToStorage_action);
HPX_REGISTER_PLAIN_ACTION_DECLARATION(CopyToStorage_action);

HPX_DEFINE_PLAIN_ACTION(Storage::CopyFromStorage, CopyFromStorage_action);
HPX_REGISTER_PLAIN_ACTION_DECLARATION(CopyFromStorage_action);

// and these in a cpp
HPX_REGISTER_PLAIN_ACTION(CopyToStorage_action);
HPX_REGISTER_PLAIN_ACTION(CopyFromStorage_action);

//----------------------------------------------------------------------------
// the main message sending loop may generate many thousands of send requests
// and each is associated with a future. To reduce the number we must wait on
// this loop runs in a background thread and simply removes any completed futures
// from the main list of active futures.
int RemoveCompletions()
{
    int num_removed = 0;
    //
    while(FuturesActive) {
        int last_removed = num_removed;
        {
            hpx::lcos::local::spinlock::scoped_lock lk(FuturesMutex);
            for(std::vector<hpx::future<int>> &futvec : ActiveFutures) {
                for(std::vector<hpx::future<int>>::iterator fut = futvec.begin();
                    fut != futvec.end(); /**/)
                {
                    if(fut->is_ready()){
                        if(fut->get() != TEST_SUCCESS) {
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
        hpx::this_thread::suspend(boost::posix_time::microseconds(10));
    }
    return num_removed;
}

//----------------------------------------------------------------------------
// Take a vector of futures representing pass/fail and reduce to a single pass fail
int reduce(hpx::future<std::vector<hpx::future<int>>> futvec)
{
    int res = TEST_SUCCESS;
    std::vector<hpx::future<int>> vfs = futvec.get();
    for(hpx::future<int>& f : vfs) {
        if(f.get() == TEST_FAIL) return TEST_FAIL;
    }
    return res;
}

//----------------------------------------------------------------------------
// Create a new barrier and register its gid with the given symbolic name.
hpx::lcos::barrier create_barrier(std::size_t num_localities, char const* symname)
{
    hpx::lcos::barrier b;
    DEBUG_OUTPUT(2,
        std::cout << "Creating barrier based on N localities "
                  << num_localities << std::endl;
    );
    b.create(hpx::find_here(), num_localities);
    hpx::agas::register_name_sync(symname, b.get_gid());
    return b;
}

//----------------------------------------------------------------------------
void barrier_wait()
{
    hpx::id_type here = hpx::find_here();
    uint64_t rank = hpx::naming::get_locality_id_from_id(here);

    hpx::id_type id = hpx::invalid_id;
    for (std::size_t i = 0; i != HPX_MAX_NETWORK_RETRIES; ++i)
    {
        hpx::error_code ec;
        id = hpx::agas::resolve_name_sync("/DSM_barrier", ec);
        if (!ec) {
            DEBUG_OUTPUT(2,
              if (id==hpx::invalid_id) std::cout << "1 Invalid Barrier on rank " << rank << std::endl;
              else std::cout << "1 Barrier resolved on rank " << rank << std::endl;
            );
            break;
        }
        boost::thread::sleep( boost::get_system_time() +
                      boost::posix_time::milliseconds(HPX_NETWORK_RETRIES_SLEEP));
    }
    DEBUG_OUTPUT(2,
        if (id==hpx::invalid_id) std::cout << "2 Invalid Barrier on rank " << rank << std::endl;
        else std::cout << "2 Barrier resolved on rank " << rank << std::endl;
    );
    unique_barrier = hpx::lcos::barrier(id);
    unique_barrier.wait();
}

//----------------------------------------------------------------------------
// Test speed of write/put
void test_write(
    uint64_t rank, uint64_t nranks, uint64_t num_transfer_slots,
    std::mt19937& gen, std::uniform_int_distribution<>& random_rank,
    std::uniform_int_distribution<>& random_slot,
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
    //
    for(int i = 0; i < options.iterations; i++) {
        DEBUG_OUTPUT(1,
            std::cout << "Starting iteration " << i << " on rank " << rank << std::endl;
        );
        //
        // start a thread which will clear any completed futures from our list.
        //
        FuturesActive = true;
        hpx::future<int> cleaner = hpx::async(RemoveCompletions);
        //
        // Start main message sending loop
        //
        for(int i = 0; i < num_transfer_slots; i++) {
            // pick a random locality to send to
            int send_rank = random_rank(gen);
            // get the pointer to the current packet send buffer
            char *buffer = &local_storage[i*options.transfer_size_B];
            // Get the HPX locality from the dest rank
            hpx::id_type locality = hpx::naming::get_id_from_locality_id(send_rank);
            // pick a random slot to write our data into
            int memory_slot = random_slot(gen);
            uint32_t memory_offset = memory_slot*options.transfer_size_B;

            // Execute a PUT on whatever locality we chose
            // Create a serializable memory buffer ready for sending.
            // Do not copy any data. Protect this with a mutex to ensure the
            // background thread removing completed futures doesn't collide
            {
                DEBUG_OUTPUT(5,
                    std::cout << "Put from rank " << rank << " on rank " << send_rank << std::endl;
                );
                ++FuturesWaiting[send_rank];
                hpx::lcos::local::spinlock::scoped_lock lk(FuturesMutex);
                ActiveFutures[send_rank].push_back(
                    hpx::async(actWrite, locality,
                        TransferBuffer(static_cast<char*>(buffer), options.transfer_size_B, TransferBuffer::reference),
                        memory_offset, options.transfer_size_B
                    ).then(
                        hpx::launch::sync,
                        [=](hpx::future<int> fut) -> int {
                            int result = fut.get();
                            --FuturesWaiting[send_rank];
                            return result;
                        })
                );
            }
        }
        // tell the cleaning thread it's time to stop
        FuturesActive = false;
        // wait for cleanup thread to terminate before we reduce any remaining futures
        int removed = cleaner.get();
        DEBUG_OUTPUT(2,
            std::cout << "Cleaning thread removed " << removed << std::endl;
        );
        //
        std::vector<hpx::future<int>> final_list;
        for(int i = 0; i < nranks; i++) {
            // move the contents of intermediate vector into final list
            final_list.reserve(final_list.size() + ActiveFutures[i].size());
            std::move(ActiveFutures[i].begin(), ActiveFutures[i].end(), std::back_inserter(final_list));
            ActiveFutures[i].clear();
        }

        hpx::future<int> result = when_all(final_list).then(hpx::launch::sync, reduce);
        result.get();
    }
    barrier_wait();
    //
    double writeMB   = nranks*options.local_storage_MB*options.iterations;
    double writeTime = timerWrite.elapsed();
    double writeBW   = writeMB / writeTime;
    double IOPS      = options.iterations*num_transfer_slots;
    double IOPs_s    = IOPS/writeTime;
    if(rank == 0) {
        std::cout << "Total time         : " << writeTime << "\n";
        std::cout << "Memory Transferred : " << writeMB   << "MB\n";
        std::cout << "Number of IOPs     : " << IOPS      << "\n";
        std::cout << "IOPs/s             : " << IOPs_s    << "\n";
        std::cout << "Aggregate BW Write : " << writeBW   << "MB/s" << std::endl;
        // a complete set of results that our python matplotlib script will ingest
        char const* msg = "CSVData, write, ranks, %1%, threads, %2%, Memory, %3%, IOPsize, %4%, IOPS/s, %5%, BW, %6%, ";
        std::cout << (boost::format(msg) % nranks % options.threads % writeMB % options.transfer_size_B 
          % IOPs_s % writeBW ) << std::endl;
    }
}

//----------------------------------------------------------------------------
// Test speed of read/get
void test_read(
    uint64_t rank, uint64_t nranks, uint64_t num_transfer_slots,
    std::mt19937& gen, std::uniform_int_distribution<>& random_rank,
    std::uniform_int_distribution<>& random_slot,
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
    // this is mostly the same as the put loop, except that the received future is not
    // an int, but a transfer buffer which we have to copy out of.
    //
    hpx::util::high_resolution_timer timerRead;
    //
    for(int i = 0; i < options.iterations; i++) {
        //
        // start a thread which will clear any completed futures from our list.
        //
        FuturesActive = true;
        hpx::future<int> cleaner = hpx::async(RemoveCompletions);
        //
        // Start main message sending loop
        //
        for(int i = 0; i < num_transfer_slots; i++) {
            // pick a random locality to send to
            int send_rank = random_rank(gen);
            // get the pointer to the current packet send buffer
            char *buffer = &local_storage[i*options.transfer_size_B];
            // Get the HPX locality from the dest rank
            hpx::id_type locality = hpx::naming::get_id_from_locality_id(send_rank);
            // pick a random slot to write our data into
            int memory_slot = random_slot(gen);
            uint32_t memory_offset = memory_slot*options.transfer_size_B;

            // Execute a GET on whatever locality we chose
            // We pass the pointer to our local memory in the PUT, and it is used
            // by the serialization routines so that the copy from parcelport memory
            // is performed directly into our user memory. This avoids the need
            // to copy the data from a serialization buffer into our memory
            {
                ++FuturesWaiting[send_rank];
                hpx::lcos::local::spinlock::scoped_lock lk(FuturesMutex);
                ActiveFutures[send_rank].push_back(
                    hpx::async(
                        actRead, locality, memory_offset, options.transfer_size_B,
                        reinterpret_cast<std::size_t>(buffer)
                    ).then(
                        hpx::launch::sync,
                        [=](hpx::future<TransferBufferReceive> fut) -> int {
                            fut.get();
                            --FuturesWaiting[send_rank];
                            return TEST_SUCCESS;
                        })
                    );
            }
        }
        // tell the cleaning thread it's time to stop
        FuturesActive = false;
        // wait for cleanup thread to terminate before we reduce any remaining futures
        int removed = cleaner.get();
        DEBUG_OUTPUT(2,
            std::cout << "Cleaning thread removed " << removed << std::endl;
        );
        //
        std::vector<hpx::future<int>> final_list;
        for(int i = 0; i < nranks; i++) {
            // move the contents of intermediate vector into final list
            final_list.reserve(final_list.size() + ActiveFutures[i].size());
            std::move(ActiveFutures[i].begin(), ActiveFutures[i].end(), std::back_inserter(final_list));
            ActiveFutures[i].clear();
        }

        hpx::future<int> result = when_all(final_list).then(hpx::launch::sync, reduce);
        result.get();
    }
    barrier_wait();
    //
    double readMB = nranks*options.local_storage_MB*options.iterations;
    double readTime = timerRead.elapsed();
    double readBW = readMB / readTime;
    double IOPS      = options.iterations*num_transfer_slots;
    double IOPs_s    = IOPS/readTime;
    if(rank == 0) {
        std::cout << "Total time         : " << readTime << "\n";
        std::cout << "Memory Transferred : " << readMB << "MB \n";
        std::cout << "Number of IOPs     : " << IOPS      << "\n";
        std::cout << "IOPs/s             : " << IOPs_s    << "\n";
        std::cout << "Aggregate BW Read  : " << readBW << "MB/s" << std::endl;
        // a complete set of results that our python matplotlib script will ingest
        char const* msg = "CSVData, read, ranks, %1%, threads, %2%, Memory, %3%, IOPsize, %4%, IOPS/s, %5%, BW, %6%, ";
        std::cout << (boost::format(msg) % nranks % options.threads % readMB % options.transfer_size_B 
          % IOPs_s % readBW ) << std::endl;
    }
}

//----------------------------------------------------------------------------
void create_barrier_startup()
{
    hpx::id_type here = hpx::find_here();
    uint64_t rank = hpx::naming::get_locality_id_from_id(here);

    // create a barrier we will use at the start and end of each run to synchronize
    if(0 == rank) {
        uint64_t nranks = hpx::get_num_localities().get();
        unique_barrier = create_barrier(nranks, "/DSM_barrier");
    }
}

//----------------------------------------------------------------------------
void find_barrier_startup()
{
    hpx::id_type here = hpx::find_here();
    uint64_t rank = hpx::naming::get_locality_id_from_id(here);

    if (rank!=0) {
        hpx::id_type id = hpx::invalid_id;
        for (std::size_t i = 0; i != HPX_MAX_NETWORK_RETRIES; ++i)
        {
            hpx::error_code ec;
            id = hpx::agas::resolve_name_sync("/DSM_barrier", ec);
            if (!ec) {
                DEBUG_OUTPUT(2,
                    std::cout << "Barrier resolved on rank " << rank << std::endl;
                );
                break;
            }
            boost::thread::sleep( boost::get_system_time() +
                         boost::posix_time::milliseconds(HPX_NETWORK_RETRIES_SLEEP));
        }
        DEBUG_OUTPUT(2,
            std::cout << (id==hpx::invalid_id ?
                "Barrier ok on rank " : "Invalid Barier on rank ") << rank << std::endl;
        );
        unique_barrier = hpx::lcos::barrier(id);
    }
}

//----------------------------------------------------------------------------
// Main test loop which randomly sends packets of data from one locality to another
// looping over the entire buffer address space and timing the total transmit/receive time
// to see how well we're doing.
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
        << " nodes, please recompile this test with the MAX_RANKS set to a higher number " << std::endl;
      return 1;
    }

    char const* msg = "hello world from OS-thread %1% on locality %2% rank %3% hostname %4%";
    std::cout << (boost::format(msg) % current % hpx::get_locality_id() % rank % name.c_str()) << std::endl;
    //
    // extract command line argument
    test_options options;
    options.transfer_size_B   = vm["transferKB"].as<boost::uint64_t>() * 1024;
    options.local_storage_MB  = vm["localMB"].as<boost::uint64_t>();
    options.global_storage_MB = vm["globalMB"].as<boost::uint64_t>();
    options.iterations        = vm["iterations"].as<boost::uint64_t>();
    options.threads           = boost::lexical_cast<boost::uint64_t>(vm["hpx:threads"].as<std::string>());

    //
    if (options.global_storage_MB>0) {
      options.local_storage_MB = options.global_storage_MB/nranks;
    }
    allocate_local_storage(options.local_storage_MB*1024*1024);
    //
    uint64_t num_transfer_slots = 1024*1024*options.local_storage_MB / options.transfer_size_B;
    //
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> random_rank(0, (int)nranks - 1);
    std::uniform_int_distribution<> random_slot(0, (int)num_transfer_slots - 1);
    //
    ActiveFutures.resize(nranks);
    for(int i = 0; i < nranks; i++) {
        FuturesWaiting[i] = 0;
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
        hpx::agas::unregister_name_sync("/DSM_barrier");

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
/*  
    desc_commandline.add_options()
        ( "distribution",
          boost::program_options::value<boost::uint64_t>()->default_value(5),
          "Specify the distribution of data blocks to send/receive\n"
          "0 : random \n"
          "1 : block cyclic")
        ;
*/  
    // make sure our barrier was already created before hpx_main runs
    DEBUG_OUTPUT(2,
        std::cout << "Registering create_barrier startup function " << std::endl;
    );
    hpx::register_pre_startup_function(&create_barrier_startup);
    DEBUG_OUTPUT(2,
        std::cout << "Registering find_barrier startup function " << std::endl;
    );
    hpx::register_startup_function(&find_barrier_startup);

    // Initialize and run HPX
    return hpx::init(desc_commandline, argc, argv);
}

