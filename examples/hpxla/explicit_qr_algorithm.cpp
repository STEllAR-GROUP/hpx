////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2012 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
//#include <hpx/util/high_resolution_timer.hpp>
//#include <hpx/components/dataflow/dataflow.hpp>

#include <complex>
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
    BOOST_ASSERT(2 == A.order());

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
inline orthotope<T> blocked_matrix_multiply(
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
                    hpx::lcos::async<multiply_and_add_action>(
                            hpx::find_here(), C_sub, A_sub, B_sub, mtx
                        ).get_future());
            } 
        }
    }

    hpx::lcos::wait(stop_list);

    return C;
}

template <
    typename T
>
inline T euclidean_norm(
    orthotope<T> const& w
    )
{
    BOOST_ASSERT(1 == w.order());

    T sum = T();

    for (std::size_t i = 0; i < w.extent(0); ++i)
        sum += (w(i) * w(i));

    return std::sqrt(sum);
}

template <
    typename T
>
inline T compute_sigma(
    orthotope<T> const& R
  , std::size_t n
  , std::size_t l
    )
{
    BOOST_ASSERT(2 == R.order());

    T sum = T();

    for (std::size_t i = l; i < n; ++i)
        sum += (R(i, l) * R(i, l));

    return std::sqrt(sum);
}

template <
    typename T
>
inline boost::int16_t compute_sign(
    T const& x
    )
{
    T const epsilon = std::numeric_limits<T>::epsilon();
 
    if (std::abs(x) < epsilon)
        return 1;
    else if (x < epsilon)
        return -1;
    else
        return 1;
}

template <
    typename T
>
inline orthotope<T> compute_H(
    orthotope<T> const& w
    )
{
    BOOST_ASSERT(1 == w.order());

    std::size_t const n = w.extent(0);

    orthotope<T> H({n, n});

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            if (i == j)
                H(i, j) = 1 - 2 * (w(i) * w(j));
            else
                H(i, j) = 0 - 2 * (w(i) * w(j)); 
        }
    }

    return H;
}

template <
    typename T
>
void check_QR(
    orthotope<T> const& A
  , orthotope<T> const& Q
  , orthotope<T> const& R
    )
{ // {{{
    BOOST_ASSERT(2 == A.order());
    BOOST_ASSERT(2 == Q.order());
    BOOST_ASSERT(2 == R.order());
    BOOST_ASSERT(A.hypercube());
    BOOST_ASSERT(Q.hypercube());
    BOOST_ASSERT(R.hypercube());

    std::size_t const n = A.extent(0);

    BOOST_ASSERT(n == Q.extent(0));
    BOOST_ASSERT(n == R.extent(0));

    ///////////////////////////////////////////////////////////////////////////
    /// Make sure Q * R equals A.
    orthotope<T> QR = matrix_multiply(Q, R);

    for (std::size_t l = 0; l < n; ++l)
    {
        for (std::size_t i = 0; i < n; ++i)
        {
            if (!compare_floating(A(l, i), QR(l, i), 1e-6)) 
                std::cout << "WARNING: QR[" << l << "][" << i << "] (value "
                          << QR(l, i) << ") is not equal to A[" << l << "]["
                          << i << "] (value " << A(l, i) << ")\n";
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    /// Make sure R is an upper triangular matrix. 
    for (std::size_t l = 0; l < (n - 1); ++l)
    {
        for (std::size_t i = l + 1; i < n; ++i)
        {
            if (!compare_floating(0.0, R(i, l), 1e-6))
                std::cout << "WARNING: R[" << i << "][" << l << "] is not 0 "
                             "(value is " << R(i, l) << "), R is not an upper "
                             "triangular matrix\n";
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    /// Make sure Q is orthogonal. A matrix is orthogonal if its transpose is
    /// equal to its inverse:
    ///
    ///     Q^T = Q^-1
    ///
    /// This implies that:
    ///
    ///     Q^T * Q = Q * Q^T = I
    /// 
    /// We use the above formula to verify Q's orthogonality. 
    orthotope<T> QT = Q.copy();

    // Transpose QT.
    for (std::size_t l = 0; l < (n - 1); ++l)
        for (std::size_t i = l + 1; i < n; ++i)
            std::swap(QT(l, i), QT(i, l));

    // Compute Q^T * Q and store the result in QT.
    QT = matrix_multiply(Q, QT);

    for (std::size_t l = 0; l < n; ++l)
    {
        for (std::size_t i = 0; i < n; ++i)
        {
            // Diagonals should be 1. 
            if (l == i)
            {
                if (!compare_floating(1.0, QT(l, i), 1e-6)) 
                    std::cout << "WARNING: (Q^T * Q)[" << l << "][" << i << "] "
                                 "is not 1 (value is " << QT(l, i) << "), Q is "
                                 "not an orthogonal matrix\n";
            }

            // All other entries should be 0.
            else
            {
                if (!compare_floating(0.0, QT(l, i), 1e-6)) 
                    std::cout << "WARNING: (Q^T * Q)[" << l << "][" << i << "] "
                                 "is not 0 (value is " << QT(l, i) << "), Q is "
                                 "not an orthogonal matrix\n";
            }
        }
    }
} // }}}

template <
    typename T
>
void write_octave_file(
    orthotope<T> const& A
  , std::string const& name
    )
{
    BOOST_ASSERT(2 == A.order());

    std::size_t const rows = A.extent(0);
    std::size_t const cols = A.extent(1);

    std::ofstream file(name + ".mat");

    BOOST_ASSERT(file.is_open());

    file << "# name: " << name << "\n"
         << "# type: matrix\n"
         << "# rows: " << rows << "\n"
         << "# columns: " << cols << "\n";

    for (std::size_t i = 0; i < rows; ++i)
    {
        for (std::size_t j = 0; j < cols; ++j)
        {
            T const v = (compare_floating(0.0, A(i, j), 1e-6) ? 0.0 : A(i, j));
            file << " " << v; 
        }

        file << "\n";
    }

    file.close();
}

template <
    typename T
>
void householders(
    orthotope<T> const& A
  , orthotope<T>& Q
  , orthotope<T>& R
    )
{
    BOOST_ASSERT(2 == A.order());
    BOOST_ASSERT(A.hypercube());

    std::size_t const n = A.extent(0);

    R = A.copy();
    Q = orthotope<T>({n, n});

    for (std::size_t l = 0; l < n; ++l)
        Q(l, l) = 1.0;

    for (std::size_t l = 0; l < (n - 1); ++l)
    {
        T const sigma = compute_sigma(R, n, l);
        boost::int16_t const sign = compute_sign(R(l, l));

        #if defined(HPXLA_DEBUG_HOUSEHOLDERS)
            std::cout << (std::string(80, '#') + "\n")
                      << "ROUND " << l << "\n\n";

            print(sigma, "sigma");
            print(sign, "sign");
        #endif

        orthotope<T> w({n});

        w(l) = R(l, l) + sign * sigma;
 
        for (std::size_t i = (l + 1); i < w.extent(0); ++i)
            w(i) = R(i, l);

        #if defined(HPXLA_DEBUG_HOUSEHOLDERS)
            print(w, "u");
        #endif

        T const w_norm = euclidean_norm(w);

        for (std::size_t i = l; i < n; ++i)
            w(i) /= w_norm;

        #if defined(HPXLA_DEBUG_HOUSEHOLDERS)
            print(w, "v");
        #endif

        orthotope<T> H = compute_H(w);

        #if defined(HPXLA_DEBUG_HOUSEHOLDERS)
            print(H, "H");
        #endif

        R = matrix_multiply(H, R);

        Q = matrix_multiply(Q, H);

        for (std::size_t i = l + 1; i < n; ++i)
            R(i, l) = 0;
    }

    #if defined(HPXLA_DEBUG_HOUSEHOLDERS)
        std::cout << std::string(80, '#') << "\n";
    #endif

    #if defined(HPXLA_DEBUG_HOUSEHOLDERS)
        check_QR(A, Q, R);
    #endif
}

template <
    typename T
>
void qr_eigenvalue(
    orthotope<T> const& A
    )
{
    BOOST_ASSERT(2 == A.order());
    BOOST_ASSERT(A.hypercube());

    std::size_t const n = A.extent(0); 
    
    orthotope<T> Ak = A.copy(), R, Q;

    std::size_t iterations = 0;

    while (true)
    {
        ++iterations;

        householders(Ak, Q, R);

        Ak = matrix_multiply(R, Q); 

        bool pseudo_upper_triangular = true;

        for (std::size_t k = 0; k < (n - 1); ++k)
            for (std::size_t i = k + 2; i < n; ++i)
                if (!compare_floating(0.0, Ak(i, k), 1e-6))
                    pseudo_upper_triangular = false;

        if (pseudo_upper_triangular)
            break;

        if (iterations > 10000)
        {
            std::cout << "didn't converge\n";
            write_octave_file(Ak, "Ak");
            return;
        }
    }

    std::cout << "converged in " << iterations << " iterations\n";

    write_octave_file(Ak, "Ak");

    for (std::size_t i = 0; i < n; ++i)
    {
        if (i != n)
        {
            // Check for complex eigenvalues.
            if (!compare_floating(0.0, Ak(i + 1, i), 1e-6))
            {
                /// For a 2x2 matrix:
                ///
                ///     A = [a b]
                ///         [c d]
                ///
                /// The eigenvalues are:
                ///
                ///     e0 = (a + d) / 2 + sqrt(4 * b * c + (a - d) ^ 2) / 2
                ///     e1 = (a + d) / 2 - sqrt(4 * b * c + (a - d) ^ 2) / 2
                std::complex<T> e0, e1
                              , a(Ak(i,     i))
                              , b(Ak(i,     i + 1))
                              , c(Ak(i + 1, i))
                              , d(Ak(i + 1, i + 1));

                std::complex<T> const c2(2.0);
                std::complex<T> const c4(4.0);

                using std::sqrt;

                e0 = (a + d) / c2 + sqrt(c4 * b * c + (a - d) * (a - d)) / c2;
                e1 = (a + d) / c2 - sqrt(c4 * b * c + (a - d) * (a - d)) / c2;

                // Print e0
                std::cout << "ev[" << i << "] = " << e0.real();

                if (!compare_floating(0.0, e0.imag(), 1e-6))
                {
                    if (1 == compute_sign(e0.imag()))
                        std::cout << " + " << e0.imag() << "i\n"; 
                    else
                        std::cout << " - " << std::abs(e0.imag()) << "i\n"; 
                }

                else
                    std::cout << "\n";

                // Print e1
                std::cout << "ev[" << (i + 1) << "] = " << e1.real();

                if (!compare_floating(0.0, e1.imag(), 1e-6))
                {
                    if (1 == compute_sign(e1.imag()))
                        std::cout << " + " << e1.imag() << "i\n"; 
                    else
                        std::cout << " - " << std::abs(e1.imag()) << "i\n"; 
                }

                else
                    std::cout << "\n";
 
                // A(i + 1, i + 1) is also a complex eigenvalue, so we skip it
                // next iteration.
                ++i;

                continue;
            }
        }

        std::cout << "ev[" << i << "] = " << Ak(i, i) << "\n";
    }
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

    boost::random::mt19937_64 rng(seed);
    boost::random::uniform_int_distribution<> dist(-100, 100);

    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < m; ++j)
            A(i, j) = T(dist(rng));
}

template <
    typename T
>
inline void random_symmetric_matrix(
    orthotope<T>& A
  , std::size_t seed
    )
{
    BOOST_ASSERT(2 == A.order());
    BOOST_ASSERT(A.hypercube());

    std::size_t const n = A.extent(0);

    boost::random::mt19937_64 rng(seed);
    boost::random::uniform_int_distribution<> dist(-100, 100);

    for (std::size_t l = 0; l < n; ++l)
        A(l, l) = T(dist(rng));

    for (std::size_t l = 0; l < (n - 1); ++l)
    {
        for (std::size_t i = l + 1; i < n; ++i)
        {
            A(i, l) = T(dist(rng));
            A(l, i) = A(i, l);
        }
    }
}

int hpx_main(boost::program_options::variables_map& vm)
{
    {
        orthotope<double> A({4, 4});

/*
        A.row(0,  4,   -2,    2,    8   );
        A.row(1, -2,    6,    2,    4   );
        A.row(2,  2,    2,    10,  -6   );
        A.row(3,  8,    4,   -6,    12  );
*/

        A.row(0, -93,  -34,   18,  -85  );
        A.row(1, -19,   6,    86,  -35  );
        A.row(2,  13,   11,  -27,   90  );
        A.row(3,  68,   6,   -12,  -75  );

        boost::uint64_t const seed = 1332629047;//std::time(0);

        std::cout << "seed: " << seed << "\n\n";

//        random_matrix(A, seed);

        write_octave_file(A, "A");

        hpx::util::high_resolution_timer t;

        qr_eigenvalue(A);

        std::cout << "elapsed time: " << t.elapsed() << "\n";
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

