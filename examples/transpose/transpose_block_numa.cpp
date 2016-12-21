//  Copyright (c) 2014-2015 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>

#include <hpx/include/parallel_algorithm.hpp>
#include <hpx/include/parallel_numeric.hpp>
#include <hpx/include/serialization.hpp>
#include <hpx/util/safe_lexical_cast.hpp>

#include <hpx/parallel/util/numa_allocator.hpp>

#include <boost/range/irange.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#define COL_SHIFT 1000.00           // Constant to shift column index
#define ROW_SHIFT 0.001             // Constant to shift row index

///////////////////////////////////////////////////////////////////////////////
hpx::threads::topology& retrieve_topology()
{
    static hpx::threads::topology& topo = hpx::threads::create_topology();
    return topo;
}

bool verbose = false;

char const* A_block_basename = "/transpose/block/A";
char const* B_block_basename = "/transpose/block/B";

struct sub_block
{
    enum mode {
        reference
      , owning
    };

    sub_block()
      : size_(0)
      , data_(nullptr)
      , mode_(reference)
    {}

    sub_block(double * data, std::uint64_t size)
      : size_(size)
      , data_(data)
      , mode_(reference)
    {}

    ~sub_block()
    {
        if(data_ && mode_ == owning)
        {
            delete[] data_;
        }
    }

    sub_block(sub_block && other)
      : size_(other.size_)
      , data_(other.data_)
      , mode_(other.mode_)
    {
        if(mode_ == owning) { other.data_ = nullptr; other.size_ = 0; }
    }

    sub_block & operator=(sub_block && other)
    {
        size_ = other.size_;
        data_ = other.data_;
        mode_ = other.mode_;
        if(mode_ == owning) { other.data_ = nullptr; other.size_ = 0; }

        return *this;
    }

    double operator[](std::size_t i) const
    {
        HPX_ASSERT(data_);
        return data_[i];
    }

    double & operator[](std::size_t i)
    {
        HPX_ASSERT(data_);
        HPX_ASSERT(mode_ == reference);
        return data_[i];
    }

    void load(hpx::serialization::input_archive & ar, unsigned version)
    {
        ar & size_;
        if(size_ > 0)
        {
            data_ = new double[size_];
            hpx::serialization::array<double> arr(data_, size_);
            ar >> arr;
            mode_ = owning;
        }
    }

    void save(hpx::serialization::output_archive & ar, unsigned version) const
    {
        ar & size_;
        if(size_ > 0)
        {
            hpx::serialization::array<double> arr(data_, size_);
            ar << arr;
        }
    }

    HPX_SERIALIZATION_SPLIT_MEMBER()

    std::uint64_t size_;
    double * data_;
    mode mode_;

    HPX_MOVABLE_ONLY(sub_block);
};

///////////////////////////////////////////////////////////////////////////////
// dirty workaround to avoid serialization of executors
typedef
    hpx::threads::executors::local_priority_queue_attached_executor
    executor_type;
typedef
    std::vector<executor_type>
    executors_vector;
typedef
    hpx::parallel::util::numa_allocator<double, executors_vector>
    allocator_type;

executors_vector execs;

struct block_component
  : hpx::components::component_base<block_component>
{
    block_component()
      : data_(0, 0.0, allocator_type(execs_, retrieve_topology()))
    {
        // This ctor should never be called.
        HPX_ASSERT(false);
    }

    block_component(std::uint64_t size, std::size_t numa_domain)
      : execs_(1, execs[numa_domain])
      , data_(size, 0.0, allocator_type(execs_, retrieve_topology()))
    {}

    sub_block get_sub_block(std::uint64_t offset, std::uint64_t size)
    {
        HPX_ASSERT(!data_.empty());
        return sub_block(&data_[offset], size);
    }

    HPX_DEFINE_COMPONENT_DIRECT_ACTION(block_component, get_sub_block);

    executors_vector execs_;
    std::vector<double, allocator_type> data_;
};

struct block
  : hpx::components::client_base<block, block_component>
{
    typedef hpx::components::client_base<block, block_component> base_type;

    block() {}

    block(hpx::future<hpx::id_type> id)
      : base_type(std::move(id))
    {
    }

    block(
        std::uint64_t id, std::uint64_t size, const char * base_name,
        std::size_t numa_domain)
      : base_type(hpx::new_<block_component>(hpx::find_here(), size, numa_domain))
    {
    }

    hpx::future<sub_block> get_sub_block(std::uint64_t offset, std::uint64_t size)
    {
        block_component::get_sub_block_action act;
        return hpx::async(act, get_id(), offset, size);
    }
};

// The macros below are necessary to generate the code required for exposing
// our partition type remotely.
//
// HPX_REGISTER_COMPONENT() exposes the component creation
// through hpx::new_<>().
typedef hpx::components::component<block_component> block_component_type;
HPX_REGISTER_COMPONENT(block_component_type, block_component);

// HPX_REGISTER_ACTION() exposes the component member function for remote
// invocation.
typedef block_component::get_sub_block_action get_sub_block_action;
HPX_REGISTER_ACTION(get_sub_block_action);

void transpose(hpx::future<sub_block> Af, hpx::future<sub_block> Bf,
    std::uint64_t block_order, std::uint64_t tile_size, std::uint64_t domain);
double test_results(std::uint64_t order, std::uint64_t block_order,
    std::vector<block> & trans, std::uint64_t blocks_start,
    std::uint64_t blocks_end, std::uint64_t domain);

///////////////////////////////////////////////////////////////////////////////
std::size_t get_num_numa_nodes(hpx::threads::topology const& topo,
    boost::program_options::variables_map& vm)
{
    std::size_t numa_nodes = topo.get_number_of_numa_nodes();
    if (numa_nodes == 0)
        numa_nodes = topo.get_number_of_sockets();

    std::string num_numa_domains_str =
        vm["transpose-numa-domains"].as<std::string>();

    if (num_numa_domains_str != "all")
    {
        numa_nodes = hpx::util::safe_lexical_cast<std::size_t>(num_numa_domains_str);
    }

    return numa_nodes;
}

std::pair<std::size_t, std::size_t> get_num_numa_pus(
    hpx::threads::topology const& topo, std::size_t numa_nodes,
    boost::program_options::variables_map& vm)
{
    std::size_t numa_pus = hpx::threads::hardware_concurrency() / numa_nodes;

    std::string num_threads_str = vm["transpose-threads"].as<std::string>();
    std::size_t pus = numa_pus;

    if(num_threads_str != "all")
    {
        pus = hpx::util::safe_lexical_cast<std::size_t>(num_threads_str);
    }

    return std::make_pair(numa_pus, pus);
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    {
        // extract hardware topology
        hpx::threads::topology const& topo = retrieve_topology();

        std::size_t numa_nodes = get_num_numa_nodes(topo, vm);
        std::pair<std::size_t, std::size_t> pus =
            get_num_numa_pus(topo, numa_nodes, vm);

        hpx::id_type here = hpx::find_here();
        bool root = here == hpx::find_root_locality();

        std::uint64_t num_localities = hpx::get_num_localities().get();

        std::uint64_t order = vm["matrix_size"].as<std::uint64_t>();
        std::uint64_t iterations = vm["iterations"].as<std::uint64_t>();
        std::uint64_t num_local_blocks = vm["num_blocks"].as<std::uint64_t>();
        std::uint64_t tile_size = order;

        if(vm.count("tile_size"))
            tile_size = vm["tile_size"].as<std::uint64_t>();

        verbose = vm.count("verbose") ? true : false;

        execs.reserve(numa_nodes);

        // creating our executors ....
        for (std::size_t i = 0; i != numa_nodes; ++i)
        {
            // create executor for this NUMA domain
            execs.emplace_back(i * pus.first, pus.second);
        }

        ///////////////////////////////////////////////////////////////////////
        std::uint64_t bytes =
            static_cast<std::uint64_t>(2.0 * sizeof(double) * order * order);

        std::uint64_t num_blocks = num_localities * num_local_blocks;
        std::uint64_t num_numa_blocks = num_local_blocks / numa_nodes;

        std::uint64_t block_order = order / num_blocks;
        std::uint64_t col_block_size = order * block_order;

        std::uint64_t id = hpx::get_locality_id();

        std::vector<block> A(num_blocks);
        std::vector<block> B(num_blocks);

        std::uint64_t blocks_start = id * num_local_blocks;
        std::uint64_t blocks_end = (id + 1) * num_local_blocks;

        std::vector<boost::integer_range<std::uint64_t> > numa_ranges;
        numa_ranges.reserve(numa_nodes);

        // Actually allocate the block components in AGAS
        std::uint64_t numa_block_begin = blocks_start;
        std::uint64_t block_numa_node = 0;
        std::uint64_t numa_blocks_allocated = 0;

        // Allocate our block components in AGAS
        for(std::uint64_t b = blocks_start; b != blocks_end; ++b)
        {
            A[b] = block(b, col_block_size, A_block_basename, block_numa_node);
            B[b] = block(b, col_block_size, B_block_basename, block_numa_node);
            ++numa_blocks_allocated;
            if(numa_blocks_allocated == num_numa_blocks)
            {
                std::cout << block_numa_node << ": "
                    << numa_block_begin << " " << b + 1 << "\n";
                numa_ranges.push_back(boost::irange(numa_block_begin, b + 1));
                numa_block_begin = b + 1;
                ++block_numa_node;
                numa_blocks_allocated = 0;
            }
        }

        // establish connection between localities, refer to all blocks from
        // everywhere
        std::vector<hpx::future<hpx::id_type> > A_ids
            = hpx::find_all_from_basename(A_block_basename, num_blocks);
        std::vector<hpx::future<hpx::id_type> > B_ids
            = hpx::find_all_from_basename(B_block_basename, num_blocks);

        if(root)
        {
            std::cout
                << "Serial Matrix transpose: B = A^T\n"
                << "Matrix order          = " << order << "\n"
                << "Matrix local columns  = " << block_order << "\n"
                << "Number of blocks      = " << num_blocks << "\n"
                << "Number of localities  = " << num_localities << "\n";
            if(tile_size < order)
                std::cout << "Tile size             = " << tile_size << "\n";
            else
                std::cout << "Untiled\n";
            std::cout
                << "Number of iterations  = " << iterations << "\n";
        }

        using hpx::parallel::for_each;
        using hpx::parallel::execution::par;

        // Fill the original matrix, set transpose to known garbage value.
        auto range = boost::irange(blocks_start, blocks_end);
        for_each(par, boost::begin(range), boost::end(range),
            [&](std::uint64_t b)
            {
                std::shared_ptr<block_component> A_ptr =
                    hpx::get_ptr<block_component>(A[b].get_id()).get();
                std::shared_ptr<block_component> B_ptr =
                    hpx::get_ptr<block_component>(B[b].get_id()).get();

                for(std::uint64_t i = 0; i != order; ++i)
                {
                    for(std::uint64_t j = 0; j != block_order; ++j)
                    {
                        double col_val = COL_SHIFT * (b*block_order + j);
                        A_ptr->data_[i * block_order + j] = col_val + ROW_SHIFT * i;
                        B_ptr->data_[i * block_order + j] = -1.0;
                    }
                }

                // register the blocks for other localities to discover
                hpx::register_with_basename(A_block_basename, A[b].get_id(), b);
                hpx::register_with_basename(B_block_basename, B[b].get_id(), b);
            }
        );

        hpx::wait_all(A_ids);
        hpx::wait_all(B_ids);

        for(std::uint64_t b = 0; b != num_blocks; ++b)
        {
            // Convert id to our client
            if(b < blocks_start || b >= blocks_end)
            {
                A[b] = block(std::move(A_ids[b]));
                B[b] = block(std::move(B_ids[b]));
            }
        }

        double avgtime = 0.0;
        double maxtime = 0.0;
        double mintime = 366.0 * 24.0*3600.0; // set the minimum time to a large value;
                                              // one leap year should be enough

        hpx::lcos::local::barrier b(numa_ranges.size());

        // perform actual transpose
        std::vector<hpx::future<double> > numa_workers;
        numa_workers.reserve(numa_ranges.size());
        for(std::uint64_t domain = 0; domain < numa_ranges.size(); ++domain)
        {
            numa_workers.push_back(
                hpx::async(
                    execs[domain],
                    [&, domain]() -> double
                    {
                        double errsq = 0.0;
                        for(std::uint64_t iter = 0; iter < iterations; ++iter)
                        {
                            b.wait();
                            hpx::util::high_resolution_timer t;

                            auto range = numa_ranges[domain];

                            std::uint64_t block_start = *boost::begin(range);
                            std::uint64_t block_end = *boost::end(range);
                            std::uint64_t blocks_size = block_end - block_start;

                            std::vector<hpx::future<void> > block_futures;
                            block_futures.resize(blocks_size);

                            for_each(par.on(execs[domain]),
                                boost::begin(range), boost::end(range),
                                [
                                    domain, &block_futures, num_blocks,
                                    block_start, block_order, tile_size, &A, &B
                                ]
                                (std::uint64_t b)
                                {
                                    std::vector<hpx::future<void> > phase_futures;
                                    phase_futures.reserve(num_blocks);

                                    auto phase_range = boost::irange(
                                        static_cast<std::uint64_t>(0), num_blocks);
                                    for(std::uint64_t phase: phase_range)
                                    {
                                        const std::uint64_t block_size =
                                            block_order * block_order;
                                        const std::uint64_t from_block = phase;
                                        const std::uint64_t from_phase = b;
                                        const std::uint64_t A_offset =
                                            from_phase * block_size;
                                        const std::uint64_t B_offset =
                                            phase * block_size;

                                        phase_futures.push_back(
                                            hpx::dataflow(
                                                execs[domain]
                                              , &transpose
                                              , A[from_block].get_sub_block(
                                                    A_offset, block_size)
                                              , B[b].get_sub_block(B_offset, block_size)
                                              , block_order
                                              , tile_size
                                              , domain
                                            )
                                        );
                                    }

                                    block_futures[b - block_start] =
                                        hpx::when_all(phase_futures);
                                }
                            );

                            hpx::wait_all(block_futures);

                            b.wait();
                            double elapsed = t.elapsed();

                            if(domain == 0)
                            {
                                // Skip the first iteration
                                if(iter > 0 || iterations == 1)
                                {
                                    avgtime = avgtime + elapsed;
                                    maxtime = (std::max)(maxtime, elapsed);
                                    mintime = (std::min)(mintime, elapsed);
                                }
                            }

                            errsq += test_results(order, block_order, B,
                                block_start, block_end, domain);
                        }
                        return errsq;
                    }
                )
            );
        }
        std::vector<double> errsqs = hpx::util::unwrapped(numa_workers);

        ///////////////////////////////////////////////////////////////////////
        // Analyze and output results
        if(root)
        {
            double errsq = std::accumulate(errsqs.begin(), errsqs.end(), 0.0);

            std::cout << "Solution validates\n";
            avgtime = avgtime/static_cast<double>(
                (std::max)(iterations-1, static_cast<std::uint64_t>(1)));
            std::cout
                << "Rate (MB/s): " << 1.e-6 * bytes/mintime << ", "
                << "Avg time (s): " << avgtime << ", "
                << "Min time (s): " << mintime << ", "
                << "Max time (s): " << maxtime << "\n";

            if(verbose)
                std::cout << "Squared errors: " << errsq << "\n";
        }
    }

    return hpx::finalize();
}

// Launch with something like:
//
// --transpose-numa-domains=2 --transpose-threads=6
//
// There shouldn't be any need for using  --hpx:threads or --hpx:bind, those
// are computed internally.
//
int main(int argc, char* argv[])
{
    using namespace boost::program_options;

    options_description desc_commandline;
    desc_commandline.add_options()
        ("matrix_size", value<std::uint64_t>()->default_value(1024),
         "Matrix Size")
        ("iterations", value<std::uint64_t>()->default_value(10),
         "# iterations")
        ("tile_size", value<std::uint64_t>(),
         "Number of tiles to divide the individual matrix blocks for improved "
         "cache and TLB performance")
        ("num_blocks", value<std::uint64_t>()->default_value(1),
         "Number of blocks to divide the individual matrix blocks for "
         "improved cache and TLB performance")
        ( "verbose", "Verbose output")
        ("transpose-threads",
         boost::program_options::value<std::string>()->default_value("all"),
         "number of threads per NUMA domain to use. (default: all)")
        ("transpose-numa-domains",
         boost::program_options::value<std::string>()->default_value("all"),
         "number of NUMA domains to use. (default: all)")
    ;

    // parse command line here to extract the necessary settings for HPX
    parsed_options opts =
        command_line_parser(argc, argv)
            .allow_unregistered()
            .options(desc_commandline)
            .style(command_line_style::unix_style)
            .run();

    variables_map vm;
    store(opts, vm);

    hpx::threads::topology const& topo = retrieve_topology();
    std::size_t numa_nodes = get_num_numa_nodes(topo, vm);
    std::pair<std::size_t, std::size_t> pus =
        get_num_numa_pus(topo, numa_nodes, vm);
    std::size_t num_cores = topo.get_number_of_numa_node_cores(0);

    // Initialize and run HPX, this example requires to run hpx_main on all
    // localities
    std::vector<std::string> cfg = {
        "hpx.run_hpx_main!=1",
        "hpx.numa_sensitive=2",  // no-cross NUMA stealing
        // block all cores of requested number of NUMA-domains
        boost::str(boost::format("hpx.cores=%d") % (numa_nodes * num_cores)),
        boost::str(boost::format("hpx.os_threads=%d") % (numa_nodes * pus.second))
    };

    std::string node_name("numanode");
    if (topo.get_number_of_numa_nodes() == 0)
        node_name = "socket";

    std::string bind_desc("hpx.bind!=");
    for (std::size_t i = 0; i != numa_nodes; ++i)
    {
        if (i != 0)
            bind_desc += ";";

        std::size_t base_thread = i * pus.second;
        bind_desc += boost::str(
            boost::format("thread:%d-%d=%s:%d.core:0-%d.pu:0")
              % base_thread % (base_thread+pus.second-1)  // thread:%d-%d
              % node_name % i                             // %s:%d
              % (pus.second-1)                            // core:0-%d
        );
    }
    cfg.push_back(bind_desc);

    return hpx::init(desc_commandline, argc, argv, cfg);
}

void transpose(hpx::future<sub_block> Af, hpx::future<sub_block> Bf,
    std::uint64_t block_order, std::uint64_t tile_size,
    std::uint64_t domain)
{
    sub_block const A = Af.get();
    sub_block B = Bf.get();

    if(tile_size < block_order)
    {
        for(std::uint64_t i = 0; i < block_order; i += tile_size)
        {
            for(std::uint64_t j = 0; j < block_order; j += tile_size)
            {
                std::uint64_t max_i = (std::min)(block_order, i + tile_size);
                std::uint64_t max_j = (std::min)(block_order, j + tile_size);

                for(std::uint64_t it = i; it != max_i; ++it)
                {
                    for(std::uint64_t jt = j; jt != max_j; ++jt)
                    {
                        B[it + block_order * jt] = A[jt + block_order * it];
                    }
                }
            }
        }
    }
    else
    {
        for(std::uint64_t i = 0; i != block_order; ++i)
        {
            for(std::uint64_t j = 0; j != block_order; ++j)
            {
                B[i + block_order * j] = A[j + block_order * i];
            }
        }
    }
}

double test_results(std::uint64_t order, std::uint64_t block_order,
    std::vector<block> & trans, std::uint64_t blocks_start,
    std::uint64_t blocks_end, std::uint64_t domain)
{
    using hpx::parallel::transform_reduce;
    using hpx::parallel::execution::par;

    // Fill the original matrix, set transpose to known garbage value.
    auto range = boost::irange(blocks_start, blocks_end);
    double errsq =
        transform_reduce(
            par.on(execs[domain]),
            boost::begin(range), boost::end(range),
            [&](std::uint64_t b) -> double
            {
                sub_block trans_block =
                    trans[b].get_sub_block(0, order * block_order).get();
                double errsq = 0.0;
                for(std::uint64_t i = 0; i < order; ++i)
                {
                    double col_val = COL_SHIFT * i;
                    for(std::uint64_t j = 0; j < block_order; ++j)
                    {
                        double diff = trans_block[i * block_order + j] -
                            (col_val + ROW_SHIFT * (b * block_order + j));
                        errsq += diff * diff;
                    }
                }
                return errsq;
            },
            0.0,
            [](double lhs, double rhs) { return lhs + rhs; }
        );

    if(verbose)
        std::cout << " Squared sum of differences: " << errsq << "\n";

    return errsq;
}
