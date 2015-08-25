//  Copyright (c) 2015 Thomas Heller
//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//
// This code is based on the STREAM benchmark:
// https://www.cs.virginia.edu/stream/ref.html
//
// We adopted the code and HPXifyed it.
//

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_algorithm.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/include/parallel_transform.hpp>
#include <hpx/include/iostreams.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>
#include <boost/range/functions.hpp>

#include <vector>
#include <string>

#ifndef STREAM_TYPE
#define STREAM_TYPE double
#endif

///////////////////////////////////////////////////////////////////////////////
template <typename T, typename ExPolicy>
class numa_allocator
{
public:
    // typedefs
    typedef T value_type;
    typedef value_type* pointer;
    typedef value_type const* const_pointer;
    typedef value_type& reference;
    typedef value_type const& const_reference;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;

public:
    // convert an allocator<T> to allocator<U>
    template <typename U>
    struct rebind
    {
        typedef numa_allocator<U, ExPolicy> other;
    };

public:
    numa_allocator(ExPolicy const& policy)
      : policy_(policy)
    {}

    explicit numa_allocator(numa_allocator const& rhs)
      : policy_(rhs.policy_)
    {}

    template <typename U>
    explicit numa_allocator(numa_allocator<U, ExPolicy> const& rhs)
      : policy_(rhs.policy_)
    {}

    // address
    pointer address(reference r) { return &r; }
    const_pointer address(const_reference r) { return &r; }

    // memory allocation
    pointer allocate(size_type cnt,
        typename std::allocator<void>::const_pointer = 0)
    {
        // allocate memory
        pointer p = reinterpret_cast<pointer>(::operator new(cnt * sizeof(T)));

        // first touch policy, letting execution policy do the right thing
        hpx::parallel::for_each(policy_, p, p + cnt,
            [](T& val)
            {
                // touch first byte of every object
                *reinterpret_cast<char*>(&val) = 0;
            });

        // return the overall memory block
        return p;
    }

    void deallocate(pointer p, size_type)
    {
        ::operator delete(p);
    }

    // size
    size_type max_size() const
    {
        return (std::numeric_limits<size_type>::max)() / sizeof(T);
    }

    // construction/destruction
    void construct(pointer p, const T& t) { new(p) T(t); }
    void destroy(pointer p) { p->~T(); }

    friend bool operator==(numa_allocator const&, numa_allocator const&)
    {
        return true;
    }

    friend bool operator!=(numa_allocator const& l, numa_allocator const& r)
    {
        return !(l == r);
    }

private:
    template <typename, typename>
    friend class numa_allocator;

    ExPolicy policy_;
};

///////////////////////////////////////////////////////////////////////////////
hpx::threads::topology const& retrieve_topology()
{
    static hpx::threads::topology const& topo = hpx::threads::create_topology();
    return topo;
}

double mysecond()
{
    return hpx::util::high_resolution_clock::now() * 1e-9;
}

int checktick()
{
    static const std::size_t M = 20;
    int minDelta, Delta;
    double t1, t2, timesfound[M];

    // Collect a sequence of M unique time values from the system.
    for (std::size_t i = 0; i < M; i++) {
        t1 = mysecond();
        while( ((t2=mysecond()) - t1) < 1.0E-6 )
            ;
        timesfound[i] = t1 = t2;
    }

    // Determine the minimum difference between these M values.
    // This result will be our estimate (in microseconds) for the
    // clock granularity.
    minDelta = 1000000;
    for (std::size_t i = 1; i < M; i++) {
        Delta = (int)( 1.0E6 * (timesfound[i]-timesfound[i-1]));
        minDelta = (std::min)(minDelta, (std::max)(Delta,0));
    }

    return(minDelta);
}

template <typename Vector>
void check_results(std::size_t iterations,
    Vector const & a, Vector const & b, Vector const & c)
{
    STREAM_TYPE aj,bj,cj,scalar;
    STREAM_TYPE aSumErr,bSumErr,cSumErr;
    STREAM_TYPE aAvgErr,bAvgErr,cAvgErr;
    double epsilon;
    int ierr,err;

    /* reproduce initialization */
    aj = 1.0;
    bj = 2.0;
    cj = 0.0;
    /* a[] is modified during timing check */
    aj = 2.0E0 * aj;
    /* now execute timing loop */
    scalar = 3.0;
    for (std::size_t k=0; k<iterations; k++)
        {
            cj = aj;
            bj = scalar*cj;
            cj = aj+bj;
            aj = bj+scalar*cj;
        }

    /* accumulate deltas between observed and expected results */
    aSumErr = 0.0;
    bSumErr = 0.0;
    cSumErr = 0.0;
    for (std::size_t j=0; j<a.size(); j++) {
        aSumErr += std::abs(a[j] - aj);
        bSumErr += std::abs(b[j] - bj);
        cSumErr += std::abs(c[j] - cj);
        // if (j == 417) printf("Index 417: c[j]: %f, cj: %f\n",c[j],cj);   // MCCALPIN
    }
    aAvgErr = aSumErr / (STREAM_TYPE) a.size();
    bAvgErr = bSumErr / (STREAM_TYPE) a.size();
    cAvgErr = cSumErr / (STREAM_TYPE) a.size();

    if (sizeof(STREAM_TYPE) == 4) {
        epsilon = 1.e-6;
    }
    else if (sizeof(STREAM_TYPE) == 8) {
        epsilon = 1.e-13;
    }
    else {
        printf("WEIRD: sizeof(STREAM_TYPE) = %zu\n", sizeof(STREAM_TYPE));
        epsilon = 1.e-6;
    }

    err = 0;
    if (std::abs(aAvgErr/aj) > epsilon) {
        err++;
        printf ("Failed Validation on array a[], AvgRelAbsErr > epsilon (%e)\n",
            epsilon);
        printf ("     Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n",
            aj,aAvgErr,std::abs(aAvgErr)/aj);
        ierr = 0;
        for (std::size_t j=0; j<a.size(); j++) {
            if (std::abs(a[j]/aj-1.0) > epsilon) {
                ierr++;
#ifdef VERBOSE
                if (ierr < 10) {
                    printf("         array a: index: %ld, expected: %e, "
                        "observed: %e, relative error: %e\n",
                        j,aj,a[j],std::abs((aj-a[j])/aAvgErr));
                }
#endif
            }
        }
        printf("     For array a[], %d errors were found.\n",ierr);
    }
    if (std::abs(bAvgErr/bj) > epsilon) {
        err++;
        printf ("Failed Validation on array b[], AvgRelAbsErr > epsilon (%e)\n",
            epsilon);
        printf ("     Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n",
            bj,bAvgErr,std::abs(bAvgErr)/bj);
        printf ("     AvgRelAbsErr > Epsilon (%e)\n",epsilon);
        ierr = 0;
        for (std::size_t j=0; j<a.size(); j++) {
            if (std::abs(b[j]/bj-1.0) > epsilon) {
                ierr++;
#ifdef VERBOSE
                if (ierr < 10) {
                    printf("         array b: index: %ld, expected: %e, "
                        "observed: %e, relative error: %e\n",
                        j,bj,b[j],std::abs((bj-b[j])/bAvgErr));
                }
#endif
            }
        }
        printf("     For array b[], %d errors were found.\n",ierr);
    }
    if (std::abs(cAvgErr/cj) > epsilon) {
        err++;
        printf ("Failed Validation on array c[], AvgRelAbsErr > epsilon (%e)\n",
            epsilon);
        printf ("     Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n",
            cj,cAvgErr,std::abs(cAvgErr)/cj);
        printf ("     AvgRelAbsErr > Epsilon (%e)\n",epsilon);
        ierr = 0;
        for (std::size_t j=0; j<a.size(); j++) {
            if (std::abs(c[j]/cj-1.0) > epsilon) {
                ierr++;
#ifdef VERBOSE
                if (ierr < 10) {
                    printf("         array c: index: %ld, expected: %e, "
                        "observed: %e, relative error: %e\n",
                        j,cj,c[j],std::abs((cj-c[j])/cAvgErr));
                }
#endif
            }
        }
        printf("     For array c[], %d errors were found.\n",ierr);
    }
    if (err == 0) {
        printf ("Solution Validates: avg error less than %e on all three arrays\n",
            epsilon);
    }
#ifdef VERBOSE
    printf ("Results Validation Verbose Results: \n");
    printf ("    Expected a(1), b(1), c(1): %f %f %f \n",aj,bj,cj);
    printf ("    Observed a(1), b(1), c(1): %f %f %f \n",a[1],b[1],c[1]);
    printf ("    Rel Errors on a, b, c:     %e %e %e \n",std::abs(aAvgErr/aj),
        std::abs(bAvgErr/bj),std::abs(cAvgErr/cj));
#endif
}

template <typename Vector>
double numa_domain_worker(std::size_t domain, hpx::lcos::local::latch& l,
    std::size_t part_size, std::size_t offset, std::size_t iterations,
    int quantum, Vector& a, Vector& b, Vector& c)
{
    hpx::threads::topology const& topo = retrieve_topology();
    std::size_t pus = topo.get_number_of_numa_node_pus(domain);

    std::string bind_desc = boost::str(
            boost::format("thread:0-%d=numanode:%d.pu:0-%d") %
               (pus-1) % domain % (pus-1)
        );

    // create executor for this NUMA domain
    hpx::threads::executors::local_priority_queue_os_executor exec(
        pus, bind_desc);

    typedef typename Vector::iterator iterator;
    iterator a_begin = a.begin() + offset;
    iterator b_begin = b.begin() + offset;
    iterator c_begin = c.begin() + offset;

    iterator a_end = a_begin + part_size;
    iterator b_end = b_begin + part_size;
    iterator c_end = c_begin + part_size;


    // Initialize arrays
    auto policy = hpx::parallel::par.on(exec);
    hpx::parallel::fill(policy, a_begin, a_end, 1.0);
    hpx::parallel::fill(policy, b_begin, b_end, 2.0);
    hpx::parallel::fill(policy, c_begin, c_end, 0.0);

    double t = mysecond();
    hpx::parallel::for_each(policy, a_begin, a_end,
        [](STREAM_TYPE & v)
        {
            v = 2.0 * v;
        });
    t = 1.0E6 * (mysecond() - t);

    if (domain == 0)
    {
        std::cout
            << "Each test below will take on the order"
            << " of " << (int) t << " microseconds.\n"
            << "   (= " << (int) (t/quantum) << " clock ticks)\n"
            << "Increase the size of the arrays if this shows that\n"
            << "you are not getting at least 20 clock ticks per test.\n"
            << "-------------------------------------------------------------\n"
            ;

        std::cout
            << "WARNING -- The above is only a rough guideline.\n"
            << "For best results, please be sure you know the\n"
            << "precision of your system timer.\n"
            << "-------------------------------------------------------------\n"
            ;
    }

    // synchronize across NUMA domains
    l.count_down_and_wait();

    ///////////////////////////////////////////////////////////////////////////
    // Main Loop
    t = mysecond();

    double scalar = 3.0;
    for(std::size_t iteration = 0; iteration != iterations; ++iteration)
    {
        // Copy
        hpx::parallel::copy(policy, a_begin, a_end, c_begin);

        // Scale
        hpx::parallel::transform(policy,
            c_begin, c_end, b_begin,
            [scalar](STREAM_TYPE val)
            {
                return scalar * val;
            }
        );

        // Add
        hpx::parallel::transform(policy,
            a_begin, a_end, b_begin, b_end, c_begin,
            [](STREAM_TYPE val1, STREAM_TYPE val2)
            {
                return val1 + val2;
            }
        );

        // Triad
        hpx::parallel::transform(policy,
            b_begin, b_end, c_begin, c_end, a_begin,
            [scalar](STREAM_TYPE val1, STREAM_TYPE val2)
            {
                return val1 + scalar * val2;
            }
        );
    }

    t = mysecond() - t;

    return t;
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    // extract hardware topology
    hpx::threads::topology const& topo = retrieve_topology();
    std::size_t numa_nodes = topo.get_number_of_numa_nodes();
    std::size_t pus_per_numa_node = topo.get_number_of_numa_node_pus(0);

    std::size_t vector_size = vm["vector_size"].as<std::size_t>();
    std::size_t offset = vm["offset"].as<std::size_t>();
    std::size_t iterations = vm["iterations"].as<std::size_t>();

    // Get initial value for system clock.
    int quantum = checktick();
    if(quantum >= 1)
    {
        std::cout
            << "Your clock granularity/precision appears to be " << quantum
            << " microseconds.\n"
            ;
    }
    else
    {
        std::cout
            << "Your clock granularity appears to be less than one microsecond.\n"
            ;
        quantum = 1;
    }

    std::cout
        << "-------------------------------------------------------------\n"
        << "Modified STREAM bechmark based on\nHPX version: "
            << hpx::build_string() << "\n"
        << "-------------------------------------------------------------\n"
        << "This system uses " << sizeof(STREAM_TYPE)
            << " bytes per array element.\n"
        << "-------------------------------------------------------------\n"
        << "Array size = " << vector_size << " (elements), "
           "Offset = " << offset << " (elements)\n"
        << "Memory per array = "
            << sizeof(STREAM_TYPE) * (vector_size / 1024. / 1024.) << " MiB "
        << "(= "
            <<  sizeof(STREAM_TYPE) * (vector_size / 1024. / 1024. / 1024.)
            << " GiB).\n"
        << "Each kernel will be executed " << iterations << " times.\n"
        << " The *best* time for each kernel (excluding the first iteration)\n"
        << " will be used to compute the reported bandwidth.\n"
        << "-------------------------------------------------------------\n"
        << "Number of Threads requested = "
            << hpx::threads::hardware_concurrency() << "\n"
        << "-------------------------------------------------------------\n"
        ;

    using namespace hpx::parallel;

    // allocate data
    hpx::threads::executors::default_executor exec;
    auto numa_policy = par.on(exec).with(static_chunk_size());

    typedef numa_allocator<STREAM_TYPE, decltype(numa_policy)> allocator_type;
    allocator_type alloc(numa_policy);

    typedef std::vector<STREAM_TYPE, allocator_type> vector_type;
    vector_type a(vector_size, STREAM_TYPE(), alloc);
    vector_type b(vector_size, STREAM_TYPE(), alloc);
    vector_type c(vector_size, STREAM_TYPE(), alloc);

    // perform benchmark
    hpx::lcos::local::latch l(numa_nodes);

    std::vector<hpx::future<double> > workers;
    workers.reserve(numa_nodes);

    std::size_t part_size = vector_size/numa_nodes;
    for (std::size_t i = 0; i != numa_nodes; ++i)
    {
        // create one worker per NUMA domain with part of the data to work on
        hpx::threads::executors::default_executor exec(i);
        workers.push_back(
            hpx::async(exec, &numa_domain_worker<vector_type>,
                i, boost::ref(l), part_size, part_size*i, iterations, quantum,
                boost::ref(a), boost::ref(b), boost::ref(c))
        );
    }

    hpx::wait_all(workers);

    // Check Results ...
    check_results(iterations, a, b, c);

    std::cout
        << "-------------------------------------------------------------\n"
        ;

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // extract hardware topology
    hpx::threads::topology const& topo = retrieve_topology();
    std::size_t numa_nodes = topo.get_number_of_numa_nodes();
    std::size_t pus_per_numa_node = topo.get_number_of_numa_node_pus(0);

    // The idea of this benchmark is to create as many base-threads as we have
    // NUMA domains. Each of those kernel threads are bound to one of the
    // domains such that they can wander between the cores of this domain.
    //
    // The benchmark uses the static_priority scheduler for this which prevents
    // HPX threads from being stolen across the NUMA domain boundaries.
    //
    // The benchmark itself spawns one HPX-thread for each of those kernel
    // threads. Each HPX thread creates a new local_priority os_executor which
    // is then used to run the actual measurements.

    // create one kernel thread per available NUMA domain
    std::vector<std::string> cfg;
    cfg.push_back("hpx.os_threads=" +
        boost::lexical_cast<std::string>(numa_nodes));

    // use full machine
    cfg.push_back("hpx.cores=all");

    // run the static_priority scheduler
    cfg.push_back("hpx.scheduler=static-priority");

    // set affinity domain for the base scheduler threads to 'numa'
    cfg.push_back("hpx.affinity=numa");

    // make sure each of the base kernel-threads run on separate NUMA domain
    cfg.push_back("hpx.pu_step=" +
        boost::lexical_cast<std::string>(pus_per_numa_node));

    boost::program_options::options_description cmdline(
        "usage: " HPX_APPLICATION_STRING " [options]");

    cmdline.add_options()
        (   "vector_size",
            boost::program_options::value<std::size_t>()->default_value(1000),
            "size of vector (default: 1000)")
        (   "offset",
            boost::program_options::value<std::size_t>()->default_value(0),
            "offset (default: 0)")
        (   "iterations",
            boost::program_options::value<std::size_t>()->default_value(10),
            "number of iterations to repeat each test. (default: 10)")
        ;

    return hpx::init(cmdline, argc, argv, cfg);
}

