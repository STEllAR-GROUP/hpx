//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>

#include <hpx/include/parallel_algorithm.hpp>
#include <hpx/include/parallel_numeric.hpp>

#include <boost/range/irange.hpp>

#include <algorithm>
#include <vector>

#define COL_SHIFT 1000.00           // Constant to shift column index
#define ROW_SHIFT 0.001             // Constant to shift row index

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
      , data_(0)
      , mode_(reference)
    {}

    sub_block(double * data, boost::uint64_t size)
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
        if(mode_ == owning) { other.data_ = 0; other.size_ = 0; }
    }

    sub_block & operator=(sub_block && other)
    {
        size_ = other.size_;
        data_ = other.data_;
        mode_ = other.mode_;
        if(mode_ == owning) { other.data_ = 0; other.size_ = 0; }

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

    boost::uint64_t size_;
    double * data_;
    mode mode_;

    HPX_MOVABLE_BUT_NOT_COPYABLE(sub_block);
};

struct block_component
  : hpx::components::component_base<block_component>
{
    block_component() {}

    block_component(boost::uint64_t size)
      : data_(size)
    {}

    sub_block get_sub_block(boost::uint64_t offset, boost::uint64_t size)
    {
        HPX_ASSERT(!data_.empty());
        return sub_block(&data_[offset], size);
    }

    HPX_DEFINE_COMPONENT_ACTION(block_component, get_sub_block);

    std::vector<double> data_;
};

struct block
  : hpx::components::client_base<block, block_component>
{
    typedef hpx::components::client_base<block, block_component> base_type;
    block() {}

    block(boost::uint64_t id, const char * base_name)
      : base_type(hpx::find_from_basename(base_name, id))
    {
        get_gid();
    }

    block(boost::uint64_t id, boost::uint64_t size, const char * base_name)
      : base_type(hpx::new_<block_component>(hpx::find_here(), size))
    {
        hpx::register_with_basename(base_name, get_gid(), id);
    }

    hpx::future<sub_block>
        get_sub_block(boost::uint64_t offset, boost::uint64_t size) const
    {
        block_component::get_sub_block_action act;
        return hpx::async(act, get_gid(), offset, size);
    }
};

// The macros below are necessary to generate the code required for exposing
// our block_component type remotely.
//
// HPX_REGISTER_COMPONENT() exposes the component creation
// through hpx::new_<>().
typedef hpx::components::component<block_component> block_component_type;
HPX_REGISTER_COMPONENT(block_component_type, block_component);

// HPX_REGISTER_ACTION() exposes the component member function for remote
// invocation.
typedef block_component::get_sub_block_action get_sub_block_action;
HPX_REGISTER_ACTION(get_sub_block_action);

void transpose(sub_block const A, sub_block B,
    boost::uint64_t block_order, boost::uint64_t tile_size);

double test_results(boost::uint64_t order, boost::uint64_t block_order,
    std::vector<block> & trans, boost::uint64_t blocks_start,
    boost::uint64_t blocks_end);

///////////////////////////////////////////////////////////////////////////////
// The returned value type has to be the same as the return type used for
// __await below
hpx::future<sub_block> transpose_phase(
    std::vector<block> const& A, std::vector<block>& B,
    boost::uint64_t block_order, boost::uint64_t b,
    boost::uint64_t num_blocks, boost::uint64_t num_local_blocks,
    boost::uint64_t block_size, boost::uint64_t tile_size)
{
    const boost::uint64_t from_phase = b;
    const boost::uint64_t A_offset = from_phase * block_size;

    auto phase_range = boost::irange(
        static_cast<boost::uint64_t>(0), num_blocks);
    for(boost::uint64_t phase: phase_range)
    {
        const boost::uint64_t from_block = phase;
        const boost::uint64_t B_offset = phase * block_size;

        hpx::future<sub_block> from =
            A[from_block].get_sub_block(A_offset, block_size);
        hpx::future<sub_block> to =
            B[b].get_sub_block(B_offset, block_size);

        transpose(__await from, __await to, block_order, tile_size);
    }

    return sub_block();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    {
        hpx::id_type here = hpx::find_here();
        bool root = here == hpx::find_root_locality();

        boost::uint64_t num_localities = hpx::get_num_localities().get();

        boost::uint64_t order = vm["matrix_size"].as<boost::uint64_t>();
        boost::uint64_t iterations = vm["iterations"].as<boost::uint64_t>();
        boost::uint64_t num_local_blocks = vm["num_blocks"].as<boost::uint64_t>();
        boost::uint64_t tile_size = order;

        if(vm.count("tile_size"))
            tile_size = vm["tile_size"].as<boost::uint64_t>();

        verbose = vm.count("verbose") ? true : false;

        boost::uint64_t bytes =
            static_cast<boost::uint64_t>(2.0 * sizeof(double) * order * order);

        boost::uint64_t num_blocks = num_localities * num_local_blocks;

        boost::uint64_t block_order = order / num_blocks;
        boost::uint64_t col_block_size = order * block_order;

        boost::uint64_t id = hpx::get_locality_id();

        std::vector<block> A(num_blocks);
        std::vector<block> B(num_blocks);

        boost::uint64_t blocks_start = id * num_local_blocks;
        boost::uint64_t blocks_end = (id + 1) * num_local_blocks;

        // Actually allocate the block components in AGAS
        for(boost::uint64_t b = 0; b != num_blocks; ++b)
        {
            // Allocate block
            if(b >= blocks_start && b != blocks_end)
            {
                A[b] = block(b, col_block_size, A_block_basename);
                B[b] = block(b, col_block_size, B_block_basename);
            }
            // Retrieve the block by it's symbolic name
            else
            {
                A[b] = block(b, A_block_basename);
                B[b] = block(b, B_block_basename);
            }
        }

        if(root)
        {
            std::cout
                << "Distributed HPX Matrix transpose (await): B = A^T\n"
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
        using hpx::parallel::par;

        // Fill the original matrix, set transpose to known garbage value.
        auto range = boost::irange(blocks_start, blocks_end);
        for_each(par, boost::begin(range), boost::end(range),
            [&](boost::uint64_t b)
            {
                boost::shared_ptr<block_component> A_ptr =
                    hpx::get_ptr<block_component>(A[b].get_gid()).get();
                boost::shared_ptr<block_component> B_ptr =
                    hpx::get_ptr<block_component>(B[b].get_gid()).get();

                for(boost::uint64_t i = 0; i != order; ++i)
                {
                    for(boost::uint64_t j = 0; j != block_order; ++j)
                    {
                        double col_val = COL_SHIFT * (b*block_order + j);
                        A_ptr->data_[i * block_order + j] = col_val + ROW_SHIFT * i;
                        B_ptr->data_[i * block_order + j] = -1.0;
                    }
                }
            }
        );

        double errsq = 0.0;
        double avgtime = 0.0;
        double maxtime = 0.0;
        double mintime = 366.0 * 24.0*3600.0; // set the minimum time to a large value;
                                              // one leap year should be enough
        for(boost::uint64_t iter = 0; iter < iterations; ++iter)
        {
            hpx::util::high_resolution_timer t;

            auto range = boost::irange(blocks_start, blocks_end);

            const boost::uint64_t block_size = block_order * block_order;
            for_each(par, boost::begin(range), boost::end(range),
                [&](boost::uint64_t b)
                {
                    transpose_phase(A, B, block_order, b,
                        num_blocks, num_local_blocks, block_size, tile_size
                    ).get();
                });

            double elapsed = t.elapsed();

            if(iter > 0 || iterations == 1) // Skip the first iteration
            {
                avgtime = avgtime + elapsed;
                maxtime = (std::max)(maxtime, elapsed);
                mintime = (std::min)(mintime, elapsed);
            }

            if(root)
                errsq += test_results(order, block_order, B, blocks_start, blocks_end);
        } // end of iter loop

        // Analyze and output results

        double epsilon = 1.e-8;
        if(root)
        {
            if(errsq < epsilon)
            {
                std::cout << "Solution validates\n";
                avgtime = avgtime/static_cast<double>(
                    (std::max)(iterations-1, static_cast<boost::uint64_t>(1)));
                std::cout
                  << "Rate (MB/s): " << 1.e-6 * bytes/mintime << ", "
                  << "Avg time (s): " << avgtime << ", "
                  << "Min time (s): " << mintime << ", "
                  << "Max time (s): " << maxtime << "\n";

                if(verbose)
                    std::cout << "Squared errors: " << errsq << "\n";
            }
            else
            {
                std::cout
                  << "ERROR: Aggregate squared error " << errsq
                  << " exceeds threshold " << epsilon << "\n";
                hpx::terminate();
            }
        }
    }

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    using namespace boost::program_options;

    options_description desc_commandline;
    desc_commandline.add_options()
        ("matrix_size", value<boost::uint64_t>()->default_value(1024),
         "Matrix Size")
        ("iterations", value<boost::uint64_t>()->default_value(10),
         "# iterations")
        ("tile_size", value<boost::uint64_t>(),
         "Number of tiles to divide the individual matrix blocks for improved "
         "cache and TLB performance")
        ("num_blocks", value<boost::uint64_t>()->default_value(1),
         "Number of blocks to divide the individual matrix blocks for "
         "improved cache and TLB performance")
        ( "verbose", "Verbose output")
    ;

    // Initialize and run HPX, this example requires to run hpx_main on all
    // localities
    std::vector<std::string> cfg;
    cfg.push_back("hpx.run_hpx_main!=1");

    return hpx::init(desc_commandline, argc, argv, cfg);
}

void transpose(sub_block const A, sub_block B,
    boost::uint64_t block_order, boost::uint64_t tile_size)
{
    if(tile_size < block_order)
    {
        for(boost::uint64_t i = 0; i != block_order; i += tile_size)
        {
            for(boost::uint64_t j = 0; j != block_order; j += tile_size)
            {
                boost::uint64_t max_i = (std::min)(block_order, i + tile_size);
                boost::uint64_t max_j = (std::min)(block_order, j + tile_size);

                for(boost::uint64_t it = i; it != max_i; ++it)
                {
                    for(boost::uint64_t jt = j; jt != max_j; ++jt)
                    {
                        B[it + block_order * jt] = A[jt + block_order * it];
                    }
                }
            }
        }
    }
    else
    {
        for(boost::uint64_t i = 0; i != block_order; ++i)
        {
            for(boost::uint64_t j = 0; j != block_order; ++j)
            {
                B[i + block_order * j] = A[j + block_order * i];
            }
        }
    }
}

double test_results(boost::uint64_t order, boost::uint64_t block_order,
    std::vector<block> & trans, boost::uint64_t blocks_start,
    boost::uint64_t blocks_end)
{
    using hpx::parallel::transform_reduce;
    using hpx::parallel::par;

    // Fill the original matrix, set transpose to known garbage value.
    auto range = boost::irange(blocks_start, blocks_end);
    double errsq =
        transform_reduce(par, boost::begin(range), boost::end(range),
            [&](boost::uint64_t b) -> double
            {
                sub_block trans_block =
                    trans[b].get_sub_block(0, order * block_order).get();
                double errsq = 0.0;
                for(boost::uint64_t i = 0; i < order; ++i)
                {
                    double col_val = COL_SHIFT * i;
                    for(boost::uint64_t j = 0; j < block_order; ++j)
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
