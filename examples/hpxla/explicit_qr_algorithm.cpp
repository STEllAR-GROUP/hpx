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
inline bool compare_are(
    T const& x_old
  , T const& x_new
  , T const& tolerance
    )
{
    if (compare_floating(0.0, x_new, 1e-6))
        return tolerance >= (std::abs(x_new - x_old) * 100.0);

    // Absolute relative error = | (x_new - x_old) / x_new | * 100
    T const are = std::abs((x_new - x_old) / x_new) * 100.0;

    return tolerance >= are; 
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

HPX_PLAIN_ACTION(multiply_and_add<double>);

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
                    hpx::async<multiply_and_add_action>(
                            hpx::find_here(), C_sub, A_sub, B_sub, mtx
                        ));
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

/// H = I - 2 * w * w^T
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
  , std::size_t block_size
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
    orthotope<T> QR = blocked_matrix_multiply(Q, R, block_size);

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
    QT = blocked_matrix_multiply(Q, QT, block_size);

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
void write_matrix_to_octave_file(
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
void write_evs_to_octave_file(
    std::vector<std::complex<T> > const& v
  , std::string const& name
    )
{
    std::size_t const rows = v.size(); 

    std::ofstream file(name + ".mat");

    BOOST_ASSERT(file.is_open());

    file << "# name: " << name << "\n"
         << "# type: complex matrix\n"
         << "# rows: " << rows << "\n"
         << "# columns: 1\n";

    for (std::size_t i = 0; i < rows; ++i)
    {
        T const real = (compare_floating(0.0, v[i].real(), 1e-6)
                       ? 0.0 : v[i].real());
        T const imag = (compare_floating(0.0, v[i].imag(), 1e-6)
                       ? 0.0 : v[i].imag());

        file << " (" << real << "," << imag << ")\n"; 
    }

    file.close();
}


template <
    typename T
>
void householders_qr_factor(
    orthotope<T> const& A
  , orthotope<T>& Q
  , orthotope<T>& R
  , std::size_t block_size
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

        R = blocked_matrix_multiply(H, R, block_size);

        Q = blocked_matrix_multiply(Q, H, block_size);

        for (std::size_t i = l + 1; i < n; ++i)
            R(i, l) = 0;
    }

    #if defined(HPXLA_DEBUG_HOUSEHOLDERS)
        std::cout << std::string(80, '#') << "\n";
    #endif

    #if defined(HPXLA_DEBUG_HOUSEHOLDERS)
        check_QR(A, Q, R, block_size);
    #endif
}

template <
    typename T
>
void householders_tri_factor(
    orthotope<T>& A
  , std::size_t block_size
  , T eps = 1e-8
    )
{
    BOOST_ASSERT(2 == A.order());
    BOOST_ASSERT(A.hypercube());

    std::size_t const n = A.extent(0);

    for (std::size_t l = 0; l < (n - 2); ++l)
    {
        boost::int16_t sign = -compute_sign(A(l + 1, l));

        T alpha = 0.0;

        for (std::size_t j = (l + 1); j < n; ++j)
            alpha += (A(j, l) * A(j, l));

        if (alpha < eps)
            continue;

        alpha = sign * std::sqrt(alpha);

        T const r = std::sqrt(0.5 * ((alpha * alpha) - (alpha * A(l + 1, l))));

        orthotope<T> w({n});

        w(l + 1) = (A(l + 1, l) - alpha) / (2.0 * r); 

        for (std::size_t j = (l + 2); j < n; ++j)
            w(j) = A(j, l) / (2.0 * r);

        orthotope<T> H = compute_H(w);

        /// A_l = H * A_l_minus_1 * H
        orthotope<T> H_A_l_minus_1 = blocked_matrix_multiply(H, A, block_size);

        A = blocked_matrix_multiply(H_A_l_minus_1, H, block_size);
    }
}

template <
    typename T
>
std::vector<std::complex<T> > qr_eigenvalue(
    orthotope<T> const& A
  , std::size_t max_iterations
  , std::size_t block_size 
  , T const& tolerance = 1.0
    )
{
    BOOST_ASSERT(2 == A.order());
    BOOST_ASSERT(A.hypercube());

    std::size_t const n = A.extent(0); 

/*
    std::vector<std::complex<T> > evs;
    evs.reserve(n);
*/

    std::complex<T> const nan_(std::numeric_limits<T>::quiet_NaN()
                             , std::numeric_limits<T>::quiet_NaN());

    std::vector<std::complex<T> > evs(n, nan_), old(n, nan_);

    orthotope<T> Ak = A.copy(), R, Q;

    householders_tri_factor(Ak, block_size);

    write_matrix_to_octave_file(Ak, "hess_A0");

    std::size_t iterations = 0;

    while (true)
    {
/*
        T const mu = Ak(n, n);

        if (0 != iterations)
        {
            for (std::size_t i = 0; i < (n - 1); ++i)
                Ak(i, i) -= mu;

            Ak(n, n) = 0.0;
        }
*/

        householders_qr_factor(Ak, Q, R, block_size);

        Ak = blocked_matrix_multiply(R, Q, block_size); 

/*
        if (0 != iterations)
            for (std::size_t i = 0; i < n; ++i)
                Ak(i, i) += mu;
*/

/*
        bool pseudo_upper_triangular = true;

        for (std::size_t j = 0; j < (n - 1); ++j)
        {
            // Make sure we're in Hessenberg form.
            for (std::size_t i = j + 2; i < n; ++i)
            {
                if (!compare_floating(0.0, Ak(i, j), 1e-6))
                    pseudo_upper_triangular = false;
            }

            /// Check for convergence. Either we converge to 2x2 complex
            /// conjugates eigenvalues, which take the form:
            ///
            ///     [ a  b ]
            ///     [ c  a ]
            ///
            /// Where b * c < 0. Or, we converge to real eigenvalues which take
            /// the form:
            ///
            ///     [ a ]
            ///     [ 0 ]
            ///

            // Determine if we've failed to converge to a real eigenvalue. 
            if (!compare_floating(0.0, Ak(j, j + 1), 1e-6))
            {
                // Determine if we've failed to converge to a pair of complex
                // eigenvalues.
                if (!compare_floating(Ak(j, j), Ak(j + 1, j + 1), 1e-6))
                    pseudo_upper_triangular = false;
            }
        }
*/

        bool converged = true;

//        std::cout << "ITERATION " << iterations << "\n";

        for (std::size_t j = 0; j < n; ++j)
        {
            if (j != n)
            {
                // Check for complex eigenvalues.
                if (!compare_floating(0.0, Ak(j + 1, j), 1e-6))
                {
                    T const test0 = 4.0 * Ak(j, j + 1) * Ak(j + 1, j)
                                  + ( (Ak(j, j) - Ak(j + 1, j + 1))
                                    * (Ak(j, j) - Ak(j + 1, j + 1)));

                    // Check if solving the eigenvalues of the 2x2 matrix
                    // will give us a complex answer. This avoids unnecessary
                    // square roots.  
                    if (-1 == compute_sign(test0)) 
                    {
                        std::complex<T> const a(Ak(j,     j)    );
                        std::complex<T> const d(Ak(j + 1, j + 1));

                        std::complex<T> const i2(2.0);

                        std::complex<T> const comp_part
                            = std::sqrt(std::complex<T>(test0));

                        evs[j    ] = (a + d) / i2 + comp_part / i2; 
                        evs[j + 1] = (a + d) / i2 - comp_part / i2; 

/*
                        std::cout << "evs[" << j << "] = " << evs[j] << "\n"
                                  << "old[" << j << "] = " << old[j] << "\n"
                                  << "evs[" << (j + 1) << "] = " << evs[j + 1] << "\n"
                                  << "old[" << (j + 1) << "] = " << old[j + 1] << "\n";
*/

                        if ((old[j] == nan_) || (old[j + 1] == nan_))
                            converged = false;

                        else 
                        {
                            T const evs0_real = evs[j    ].real();
                            T const evs0_imag = evs[j    ].imag();
                            T const evs1_real = evs[j + 1].real();
                            T const evs1_imag = evs[j + 1].imag();

                            T const old0_real = old[j    ].real();
                            T const old0_imag = old[j    ].imag();
                            T const old1_real = old[j + 1].real();
                            T const old1_imag = old[j + 1].imag();

                            bool const test1
                                = compare_are(old0_real, evs0_real, 0.1)
                               && compare_are(old0_imag, evs0_imag, 0.1)
                               && compare_are(old1_real, evs1_real, 0.1)
                               && compare_are(old1_imag, evs1_imag, 0.1);

/*
                            std::cout << "test = " << test1 << "\n";
                            std::cout << "converged = " << converged << "\n";
*/

                            converged = converged && test1;
                        }

/*
                        std::cout << "converged[" << j << "] = " << converged << "\n";
                        std::cout << "converged[" << (j + 1) << "] = " << converged << "\n";
*/

                        // We handled Ak(j + 1, j + 1), so skip the next
                        // iteration.
                        ++j;

                        continue;
                    }
                }
            }

            evs[j] = Ak(j, j);

/*
            std::cout << "evs[" << j << "] = " << evs[j] << "\n"
                      << "old[" << j << "] = " << old[j] << "\n";
*/

            if (old[j] == nan_)
                converged = false;

            else
            {
                T const evs_real = evs[j].real();
                T const evs_imag = evs[j].imag();

                T const old_real = old[j].real();
                T const old_imag = old[j].imag();

                bool const test = compare_are(old_real, evs_real, 0.1)
                                && compare_are(old_imag, evs_imag, 0.1);

/*
                std::cout << "test = " << test << "\n";
                std::cout << "converged = " << converged << "\n";
*/

                converged = converged && test;
            }

/*
            std::cout << "converged[" << j << "] = " << converged << "\n";
*/
        }

        old = evs;

        ++iterations;

        if (converged)
            break;

/*
        if (pseudo_upper_triangular)
            break;
*/

        if (iterations >= max_iterations)
        {
            std::cout << "Didn't converge in " << max_iterations
                      << " iterations\n";
            write_matrix_to_octave_file(Ak, "best_Ak");
            return std::vector<std::complex<T> >();
        }
    }

    std::cout << "Converged in " << iterations << " iterations\n";

    write_matrix_to_octave_file(Ak, "Ak");

/*
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
                ///
                std::complex<T> e0, e1;

                std::complex<T> const a(Ak(i,     i)    );
                std::complex<T> const b(Ak(i,     i + 1));
                std::complex<T> const c(Ak(i + 1, i)    );
                std::complex<T> const d(Ak(i + 1, i + 1));

                std::complex<T> const i2(2.0);
                std::complex<T> const i4(4.0);

                using std::sqrt;

                e0 = (a + d) / i2 + sqrt(i4 * b * c + (a - d) * (a - d)) / i2;
                e1 = (a + d) / i2 - sqrt(i4 * b * c + (a - d) * (a - d)) / i2;

                evs.push_back(e0);
                evs.push_back(e1);

                // Ak(i + 1, i + 1) is also a complex eigenvalue, so we skip it
                // next iteration.
                ++i;

                continue;
            }
        }

        evs.push_back(std::complex<T>(Ak(i, i)));
    }
*/

    return evs;
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
    std::size_t const dimensions = vm["dimensions"].as<std::size_t>();

    std::size_t const block_size = vm["block-size"].as<std::size_t>();

    std::size_t const max_iterations = vm["max-iterations"].as<std::size_t>();

    std::size_t seed = vm["seed"].as<std::size_t>();

    if (!seed)
        seed = std::size_t(std::time(0));

    std::cout << "Seed: " << seed << "\n";

    {
        orthotope<double> A0({dimensions, dimensions});

        random_matrix(A0, seed);

        write_matrix_to_octave_file(A0, "A0");

        hpx::util::high_resolution_timer t;

        std::vector<std::complex<double> > evs
            = qr_eigenvalue(A0, max_iterations, block_size);

        std::cout << "Elapsed Time: " << t.elapsed() << " [s]\n";

        write_evs_to_octave_file(evs, "evs");
    }

    return hpx::finalize(); 
}

int main(int argc, char* argv[])
{
    // Configure application-specific options.
    boost::program_options::options_description
        cmdline("Usage: " HPX_APPLICATION_STRING " [options]");

    using boost::program_options::value;

    cmdline.add_options()
        ("dimensions", value<std::size_t>()->default_value(24),
            "dimensions of randomly generated square input matrix")
        ("block-size", value<std::size_t>()->default_value(6),
            "sub-block size for matrix multiplication")
        ("max-iterations", value<std::size_t>()->default_value(10000),
            "maximum number of iterations before admitting failure to converge")
        ("seed", value<std::size_t>()->default_value(0),
            "the seed for the pseudo random number generator (if 0, a seed "
            "is chosen based on the current system time)")
    ;

    // Initialize and run HPX.
    return hpx::init(cmdline, argc, argv);
}

