////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2012 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/util/high_resolution_timer.hpp>
//#include <hpx/components/dataflow/dataflow.hpp>

#include <algorithm>
#include <cmath>
#include <ctime>
#include <limits>
#include <vector>

#include <boost/shared_ptr.hpp>
#include <boost/format.hpp>
#include <boost/cstdint.hpp>
#include <boost/assert.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>

/// Base case.
template <
    typename Head
>
Head scalar_multiply(
    Head head
    )
{
    return head;
}

template <
    typename Head
  , typename... Tail
>
Head scalar_multiply(
    Head head
  , Tail... tail
    )
{
    if (sizeof...(tail))
        head *= scalar_multiply(tail...);
    return head;
}

/// Base case.
template <
    typename Head
>
Head compute_index(
    std::vector<std::size_t> const& extent
  , std::vector<std::size_t> const& offsets
  , std::size_t n
  , Head head
    )
{
    head += offsets[n];

    //std::cout << "(s{" << n << "} + " << offsets[n] << ")";

    for (std::size_t i = 0; i < n; ++i)
    {
        //std::cout << "(d{" << i << "})";  
        head *= extent[i];
    }

    //std::cout << "\n";

    return head;
}

/// Recursive case. 
template <
    typename Head
  , typename... Tail
>
Head compute_index(
    std::vector<std::size_t> const& extent
  , std::vector<std::size_t> const& offsets
  , std::size_t n
  , Head head
  , Tail... tail
    )
{
    if (sizeof...(tail))
    {
        head += offsets[n];

        //std::cout << "(s{" << n << "} + " << offsets[n] << ")";

        for (std::size_t i = 0; i < n; ++i)
        {
            //std::cout << "(d{" << i << "})";  
            head *= extent[i];
        }

        //std::cout << " + ";

        head += compute_index(extent, offsets, n + 1, tail...);
    }

    return head;
}

/// Let A by a multidimensional array with dimensions d{0}, d{1}, ..., d{n}. For
/// any index s{0}, s{1}, ..., s{n}, the offset of the index is:
///
///       (s{n}     - 1)(d{n - 1})(d{n - 2}) ... (d{0})
///     + (s{n - 1} - 1)(d{n - 2})(d{n - 3}) ... (d{0})
///     + ...
///     + (s{1}      -1)(d{0})
///     +  s{0}
template <
    typename Head
  , typename... Tail
>
Head index(
    std::vector<std::size_t> const& extent
  , Head head
  , Tail... tail
    )
{
    BOOST_ASSERT(extent.size() != 0);
    std::vector<std::size_t> const offsets(extent.size(), 0);
    return compute_index(extent, offsets, 0, head, tail...);
}

template <
    typename Head
  , typename... Tail
>
Head index(
    std::vector<std::size_t> const& extent
  , std::vector<std::size_t> const& offsets
  , Head head
  , Tail... tail
    )
{
    BOOST_ASSERT(extent.size() != 0);
    return compute_index(extent, offsets, 0, head, tail...);
}

struct matrix_mutex 
{
  private:
    boost::shared_ptr<hpx::lcos::local::mutex> mtx_;
  
  public:
    matrix_mutex()
      : mtx_(new hpx::lcos::local::mutex)
    {}

    matrix_mutex(
        matrix_mutex const& other
        )
      : mtx_(other.mtx_)
    {}

    matrix_mutex& operator=(
        matrix_mutex const& other
        )
    {
        mtx_->lock();
        mtx_ = other.mtx_;
        return *this;
    } 

    void lock() const
    {
        mtx_->lock();
    }

    bool try_lock() const
    {
        return mtx_->try_lock();
    }

    void unlock() const
    {
        mtx_->unlock();
    }

    typedef boost::unique_lock<matrix_mutex const> scoped_lock;
    typedef boost::detail::try_lock_wrapper<matrix_mutex const> scoped_try_lock;

    /// Dummy serialization function.
    template <
        typename Archive
    >
    void serialize(
        Archive&
      , unsigned
        )
    {}
};

template <
    typename T
>
struct orthotope
{
  private:
    boost::shared_ptr<std::vector<T> > matrix_;
    std::vector<std::size_t> extent_;
    std::vector<std::size_t> bounds_;
    std::vector<std::size_t> offsets_;

  public:
    orthotope() {}

    orthotope(
        std::vector<std::size_t> const& extent
        )
      : matrix_(new std::vector<T>())
      , extent_(extent)
      , bounds_(extent)
      , offsets_(bounds_.size(), 0)
    {
        BOOST_ASSERT(bounds_.size() > 0);

        std::size_t space = 1;

        for (std::size_t i = 0; i < bounds_.size(); ++i)
            space *= bounds_[i];

        matrix_->resize(space);
    }

    orthotope(
        orthotope const& other
        )
      : matrix_(other.matrix_)
      , extent_(other.extent_)
      , bounds_(other.bounds_)
      , offsets_(other.offsets_)
    {}

    orthotope(
        orthotope const& other
      , std::vector<std::size_t> const& extent
      , std::vector<std::size_t> const& offsets
        )
      : matrix_(other.matrix_)
      , extent_(extent)
      , bounds_(other.bounds_)
      , offsets_(offsets)
    {
        BOOST_ASSERT(bounds_.size() > 0);
        BOOST_ASSERT(extent_.size() > 0);
        BOOST_ASSERT(bounds_.size() == extent_.size());
    }

    orthotope& operator=(
        orthotope const& other
        )
    {
        matrix_ = other.matrix_;
        extent_ = other.extent_;
        bounds_ = other.bounds_;
        offsets_ = other.offsets_;
        return *this;
    }

    orthotope copy() const
    {
        orthotope o;
        o.matrix_.reset(new std::vector<T>(*matrix_));
        o.extent_ = extent_;
        o.bounds_ = bounds_;
        o.offsets_ = offsets_;
        return *this; 
    }

    std::size_t order() const
    {
        return extent_.size();
    }

    std::size_t extent(
        std::size_t i
        ) const
    {
        return extent_[i];
    }

    bool hypercube() const
    {
        std::size_t const n = extent_[0];

        for (std::size_t i = 1; i < extent_.size(); ++i)
            if (n != extent_[i])
                return false;

        return true; 
    }

    template <
        typename... Indices
    >
    T& operator()(
        Indices... i
        ) const
    {
        return (*matrix_)[index(bounds_, offsets_, i...)];
    }

    template <
        typename... Values
    >
    void row(
        std::size_t row
      , Values... v
        ) const 
    {
        set_row(row, 0, v...); 
    }

    template <
        typename Head
      , typename... Tail
    >
    void set_row(
        std::size_t row
      , std::size_t column
      , Head head
      , Tail... tail
        ) const 
    {
        (*this)(row, column) = head;
        set_row(row, column + 1, tail...);
    }

    template <
        typename Head
    >
    void set_row(
        std::size_t row
      , std::size_t column
      , Head head
        ) const 
    {
        (*this)(row, column) = head;
    }

    /// Dummy serialization function.
    template <
        typename Archive
    >
    void serialize(
        Archive&
      , unsigned
        )
    {}
};

template <
    typename T
>
inline bool compare_floating(
    T const& x
  , T const& y
    )
{
    T const epsilon = std::numeric_limits<T>::epsilon();
 
    if ((x + epsilon >= y) && (x - epsilon <= y))
        return true;
    else
        return false;
}

template <
    typename T
>
inline bool compare_floating(
    T const& x
  , T const& y
  , T const& epsilon
    )
{
    if ((x + epsilon >= y) && (x - epsilon <= y))
        return true;
    else
        return false;
}

template <
    typename T
>
inline void output_float(
    T const& A
    )
{
    T const v = (compare_floating(0.0, A, 1e-6) ? 0.0 : A);
    std::cout << (boost::format("%10.10s ")
               % (boost::str(boost::format("%10.5g") % v))); 
}

template <
    typename T
>
inline void print(
    T const& A
  , std::string const& name
    )
{
    std::cout << (boost::format("%- 8s = ") % name) << A << "\n\n";
}

template <
    typename T
>
inline void print(
    orthotope<T> const& A
  , std::string const& name
    )
{
    for (std::size_t i = 0; i < A.extent(0); ++i)
    {
        if (i == 0)
            std::cout << (boost::format("%- 8s = [ ") % name);
        else
            std::cout << (boost::format("%|11T |[ "));

        for (std::size_t j = 0; j < A.extent(1); ++j)
            output_float(A(i, j));

        std::cout << "]\n";
    }

    std::cout << "\n";
}

template <
    typename T
>
inline bool matrix_equal(
    orthotope<T> const& A
  , orthotope<T> const& B
    )
{
    // Are A and B both 2 dimensional?
    if ((2 != A.order()) || (2 != B.order()))
        return false;

    // Do A and B have the same dimensions?
    if ((A.extent(0) != B.extent(0)) || (A.extent(1) != B.extent(1)))
        return false;

    std::size_t const n = A.extent(0);
    std::size_t const m = A.extent(1);

    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < m; ++j)
            if (!compare_floating(A(i, j), B(i, j)))
                return false;

    return true;
}

template <
    typename T
>
inline orthotope<T> matrix_multiply(
    orthotope<T> const& A
  , orthotope<T> const& B
    )
{
    BOOST_ASSERT(A.order() == 2);
    BOOST_ASSERT(B.order() == 2);
    BOOST_ASSERT(A.extent(1) == B.extent(0));

    // (n x m) * (m x p) = (n x p)
    std::size_t const n = A.extent(0);
    std::size_t const m = A.extent(1);
    std::size_t const p = B.extent(1);

    orthotope<T> C({n, p});

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < p; ++j)
        {
            T sum = T();

            for (std::size_t l = 0; l < m; ++l)
                sum += A(i, l) * B(l, j);

            C(i, j) = sum;
        }
    }

    return C;
}

template <
    typename T
>
inline void matrix_add(
    orthotope<T> const& A
  , orthotope<T> const& B
    )
{
    BOOST_ASSERT(2 == A.order());
    BOOST_ASSERT(2 == B.order());
    BOOST_ASSERT(A.hypercube());
    BOOST_ASSERT(B.hypercube());

    std::size_t const n = A.extent(0);

    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) += B(i, j);
}

/// C_sub += A_sub * B_sub 
template <
    typename T
>
inline void multiply_and_add(
    orthotope<T> const& C_sub
  , orthotope<T> const& A_sub
  , orthotope<T> const& B_sub
  , matrix_mutex const& mtx
    )
{
    orthotope<T> AB_sub = matrix_multiply(A_sub, B_sub);

    {
        matrix_mutex::scoped_lock l(mtx);
        matrix_add(C_sub, AB_sub);
    }
}

typedef hpx::actions::plain_action4<
    // Arguments.
    orthotope<double> const& // C_sub
  , orthotope<double> const& // A_sub
  , orthotope<double> const& // B_sub
  , matrix_mutex const&
    // Bound function.
  , &multiply_and_add<double>
> multiply_and_add_action;

HPX_REGISTER_PLAIN_ACTION(multiply_and_add_action);

//typedef hpx::lcos::dataflow<multiply_and_add_action> multiply_and_add_dataflow;

template <
    typename T
>
inline orthotope<T> block_matrix_multiply(
    orthotope<T> const& A
  , orthotope<T> const& B
  , std::size_t block_size
    )
{
    BOOST_ASSERT(2 == A.order());
    BOOST_ASSERT(2 == B.order());
    BOOST_ASSERT(A.hypercube());
    BOOST_ASSERT(B.hypercube());
    BOOST_ASSERT(0 != block_size);
    BOOST_ASSERT(0 == (A.extent(0) % block_size));

    std::size_t const n = A.extent(0);

    orthotope<T> C({n, n});

    // TODO: Figure out how large this will be and do a reserve.
    std::vector<hpx::lcos::future<void> > stop_list;

    for (std::size_t i = 0; i < n; i += block_size)
    {
        for (std::size_t j = 0; j < n; j += block_size)
        {
            orthotope<T> C_sub(C, {block_size, block_size}, {i, j});
            matrix_mutex mtx;

            for (std::size_t l = 0; l < n; l += block_size)
            {
                orthotope<T> A_sub(A, {block_size, block_size}, {i, l})
                           , B_sub(B, {block_size, block_size}, {l, j});

                stop_list.push_back(
                    hpx::async<multiply_and_add_action>(
                        hpx::find_here(), C_sub, A_sub, B_sub, mtx));
            } 
        }
    }

    hpx::lcos::wait(stop_list);

    return C;
}

template <
    typename T
>
inline void random_matrix(
    orthotope<T>& A
  , std::size_t seed
    )
{
    BOOST_ASSERT(2 == A.order());

    std::size_t const n = A.extent(0);
    std::size_t const m = A.extent(1);

    boost::random::mt19937_64 rng(std::time(0));
    boost::random::uniform_int_distribution<> dist(-100, 100);

    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < m; ++j)
            A(i, j) = T(dist(rng));
}

int hpx_main(boost::program_options::variables_map& vm)
{
    {
        orthotope<double> A({100, 100}), B({100, 100});

        boost::uint64_t const seed = std::time(0);

        std::cout << "seed: " << seed << "\n\n";

        random_matrix(A, seed);
        random_matrix(B, seed);

        hpx::util::high_resolution_timer t;

        orthotope<double> C = block_matrix_multiply(A, B, 20);

        std::cout << "elapsed time: " << t.elapsed() << "\n";

        if (!matrix_equal(matrix_multiply(A, B), C))
            std::cout << "matrices are not equal!\n";
    }

    return hpx::finalize(); 
}

int main(int argc, char* argv[])
{
    // Configure application-specific options.
    boost::program_options::options_description
        cmdline("Usage: " HPX_APPLICATION_STRING " [options]");

    // Initialize and run HPX.
    return hpx::init(cmdline, argc, argv);
}

