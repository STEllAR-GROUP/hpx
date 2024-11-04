//Reproducible Floating Point Accumulations via Binned Floating Point
//Adapted to C++ by Richard Barnes from ReproBLAS v2.1.0.
//ReproBLAS by Peter Ahrens, Hong Diep Nguyen, and James Demmel.
//
//The code accomplishes several objectives:
//
//1. Reproducible summation, independent of summation order, assuming only a
//   subset of the IEEE 754 Floating Point Standard
//
//2. Has accuracy at least as good as conventional summation, and tunable
//
//3. Handles overflow, underflow, and other exceptions reproducibly.
//
//4. Makes only one read-only pass over the summands.
//
//5. Requires only one parallel reduction.
//
//6. Uses minimal memory (6 doubles per accumulator with fold=3).
//
//7. Relatively easy to use

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

namespace RFA {
    template <typename Real>
inline constexpr Real ldexp_impl(Real arg, int exp) noexcept
{
    while(exp > 0)
    {
        arg *= 2;
        --exp;
    }
    while(exp < 0)
    {
        arg /= 2;
        ++exp;
    }

    return arg;
}

    ///Class to hold a reproducible summation of the numbers passed to it
    ///
    ///@param ftype Floating-point data type; either `float` or `double`
    ///@param FOLD  The fold; use 3 as a default unless you understand it.
    template <class ftype, int FOLD = 3,
        typename std::enable_if<std::is_floating_point<ftype>::value>::type* =
            nullptr>
    class ReproducibleFloatingAccumulator
    {
    private:
        std::array<ftype, 2 * FOLD> data = {0};

        ///Floating-point precision bin width
        static constexpr auto BIN_WIDTH =
            std::is_same<ftype, double>::value ? 40 : 13;
        static constexpr auto MIN_EXP =
            std::numeric_limits<ftype>::min_exponent;
        static constexpr auto MAX_EXP =
            std::numeric_limits<ftype>::max_exponent;
        static constexpr auto MANT_DIG = std::numeric_limits<ftype>::digits;
        ///Binned floating-point maximum index
        static constexpr auto MAXINDEX =
            ((MAX_EXP - MIN_EXP + MANT_DIG - 1) / BIN_WIDTH) - 1;
        //The maximum floating-point fold supported by the library
        static constexpr auto MAXFOLD = MAXINDEX + 1;
        ///Binned floating-point compression factor
        ///This factor is used to scale down inputs before deposition into the bin of
        ///highest index
        static constexpr auto COMPRESSION =
            1.0 / (1 << (MANT_DIG - BIN_WIDTH + 1));
        ///Binned double precision expansion factor
        ///This factor is used to scale up inputs after deposition into the bin of
        ///highest index
        static constexpr auto EXPANSION =
            1.0 * (1 << (MANT_DIG - BIN_WIDTH + 1));
        static constexpr auto EXP_BIAS = MAX_EXP - 2;
        static constexpr auto EPSILON = std::numeric_limits<ftype>::epsilon();
        ///Binned floating-point deposit endurance
        ///The number of deposits that can be performed before a renorm is necessary.
        ///Applies also to binned complex double precision.
        static constexpr auto ENDURANCE = 1 << (MANT_DIG - BIN_WIDTH - 2);

        ///Generates binned floating-point reference bins
        static constexpr std::array<ftype, MAXINDEX + MAXFOLD> initialize_bins()
        {
            std::array<ftype, MAXINDEX + MAXFOLD> bins{0};

            if (std::is_same<ftype, float>::value)
            {
                bins[0] = ldexp_impl(0.75, MAX_EXP);
            }
            else
            {
                bins[0] = 2.0 * ldexp_impl(0.75, MAX_EXP - 1);
            }

            for (int index = 1; index <= MAXINDEX; index++)
            {
                bins[index] = ldexp_impl(0.75,
                    MAX_EXP + MANT_DIG - BIN_WIDTH + 1 - index * BIN_WIDTH);
            }
            for (int index = MAXINDEX + 1; index < MAXINDEX + MAXFOLD; index++)
            {
                bins[index] = bins[index - 1];
            }

            return bins;
        }

        ///The binned floating-point reference bins
        static constexpr auto bins = initialize_bins();

        ///Return a binned floating-point reference bin
        static inline constexpr const ftype* binned_bins(const int x)
        {
            return &bins[x];
        }

        ///Get the bit representation of a float
        static inline uint32_t& get_bits(float& x)
        {
            return *reinterpret_cast<uint32_t*>(&x);
        }
        ///Get the bit representation of a double
        static inline uint64_t& get_bits(double& x)
        {
            return *reinterpret_cast<uint64_t*>(&x);
        }
        ///Get the bit representation of a const float
        static inline uint32_t get_bits(const float& x)
        {
            return *reinterpret_cast<const uint32_t*>(&x);
        }
        ///Get the bit representation of a const double
        static inline uint64_t get_bits(const double& x)
        {
            return *reinterpret_cast<const uint64_t*>(&x);
        }

        ///Return a pointer to the primary vector
        inline ftype* pvec()
        {
            return &data[0];
        }
        ///Return a pointer to the carry vector
        inline ftype* cvec()
        {
            return &data[FOLD];
        }
        ///Return a const pointer to the primary vector
        inline const ftype* pvec() const
        {
            return &data[0];
        }
        ///Return a const pointer to the carry vector
        inline const ftype* cvec() const
        {
            return &data[FOLD];
        }

        static inline constexpr int ISNANINF(const ftype x)
        {
            const auto bits = get_bits(x);
            return (bits & ((2ull * MAX_EXP - 1) << (MANT_DIG - 1))) ==
                ((2ull * MAX_EXP - 1) << (MANT_DIG - 1));
        }

        static inline constexpr int EXP(const ftype x)
        {
            const auto bits = get_bits(x);
            return (bits >> (MANT_DIG - 1)) & (2 * MAX_EXP - 1);
        }

        ///Get index of float-point precision
        ///The index of a non-binned type is the smallest index a binned type would
        ///need to have to sum it reproducibly. Higher indicies correspond to smaller
        ///bins.
        static inline constexpr int binned_dindex(const ftype x)
        {
            int exp = EXP(x);
            if (exp == 0)
            {
                if (x == 0.0)
                {
                    return MAXINDEX;
                }
                else
                {
                    frexp(x, &exp);
                    return std::min((MAX_EXP - exp) / BIN_WIDTH, MAXINDEX);
                }
            }
            return ((MAX_EXP + EXP_BIAS) - exp) / BIN_WIDTH;
        }

        ///Get index of manually specified binned double precision
        ///The index of a binned type is the bin that it corresponds to. Higher
        ///indicies correspond to smaller bins.
        inline int binned_index() const
        {
            return ((MAX_EXP + MANT_DIG - BIN_WIDTH + 1 + EXP_BIAS) -
                       EXP(pvec()[0])) /
                BIN_WIDTH;
        }

        ///Check if index of manually specified binned floating-point is 0
        ///A quick check to determine if the index is 0
        inline bool binned_index0() const
        {
            return EXP(pvec()[0]) == MAX_EXP + EXP_BIAS;
        }

        ///Update manually specified binned fp with a scalar (X -> Y)
        ///
        ///This method updates the binned fp to an index suitable for adding numbers
        ///with absolute value less than @p max_abs_val
        ///
        ///@param incpriY stride within Y's primary vector (use every incpriY'th element)
        ///@param inccarY stride within Y's carry vector (use every inccarY'th element)
        void binned_dmdupdate(
            const ftype max_abs_val, const int incpriY, const int inccarY)
        {
            int i;
            int j;
            int X_index;
            int shift;
            auto* const priY = pvec();
            auto* const carY = cvec();

            if (ISNANINF(priY[0]))
            {
                return;
            }

            X_index = binned_dindex(max_abs_val);
            if (priY[0] == 0.0)
            {
                const ftype* const bins = binned_bins(X_index);
                for (i = 0; i < FOLD; i++)
                {
                    priY[i * incpriY] = bins[i];
                    carY[i * inccarY] = 0.0;
                }
            }
            else
            {
                shift = binned_index() - X_index;
                if (shift > 0)
                {
                    for (i = FOLD - 1; i >= shift; i--)
                    {
                        priY[i * incpriY] = priY[(i - shift) * incpriY];
                        carY[i * inccarY] = carY[(i - shift) * inccarY];
                    }
                    const ftype* const bins = binned_bins(X_index);
                    for (j = 0; j < i + 1; j++)
                    {
                        priY[j * incpriY] = bins[j];
                        carY[j * inccarY] = 0.0;
                    }
                }
            }
        }

        ///Add scalar @p X to suitably binned manually specified binned fp (Y += X)
        ///
        ///Performs the operation Y += X on an binned type Y where the index of Y is
        ///larger than the index of @p X
        ///
        ///@param incpriY stride within Y's primary vector (use every incpriY'th element)
        void binned_dmddeposit(const ftype X, const int incpriY)
        {
            ftype M;
            int i;
            ftype x = X;
            auto* const priY = pvec();

            if (ISNANINF(x) || ISNANINF(priY[0]))
            {
                priY[0] += x;
                return;
            }

            if (binned_index0())
            {
                M = priY[0];
                ftype qd = x * COMPRESSION;
                auto& ql = get_bits(qd);
                ql |= 1;
                qd += M;
                priY[0] = qd;
                M -= qd;
                M *= EXPANSION * 0.5;
                x += M;
                x += M;
                for (i = 1; i < FOLD - 1; i++)
                {
                    M = priY[i * incpriY];
                    qd = x;
                    ql |= 1;
                    qd += M;
                    priY[i * incpriY] = qd;
                    M -= qd;
                    x += M;
                }
                qd = x;
                ql |= 1;
                priY[i * incpriY] += qd;
            }
            else
            {
                ftype qd = x;
                auto& ql = get_bits(qd);
                for (i = 0; i < FOLD - 1; i++)
                {
                    M = priY[i * incpriY];
                    qd = x;
                    ql |= 1;
                    qd += M;
                    priY[i * incpriY] = qd;
                    M -= qd;
                    x += M;
                }
                qd = x;
                ql |= 1;
                priY[i * incpriY] += qd;
            }
        }

        ///Renormalize manually specified binned double precision
        ///
        ///Renormalization keeps the primary vector within the necessary bins by
        ///shifting over to the carry vector
        ///
        ///@param incpriX stride within X's primary vector (use every incpriX'th element)
        ///@param inccarX stride within X's carry vector (use every inccarX'th element)
        void binned_dmrenorm(const int incpriX, const int inccarX)
        {
            auto* priX = pvec();
            auto* carX = cvec();

            if (priX[0] == 0.0 || ISNANINF(priX[0]))
            {
                return;
            }

            for (int i = 0; i < FOLD; i++, priX += incpriX, carX += inccarX)
            {
                auto tmp_renormd = priX[0];
                auto& tmp_renorml = get_bits(tmp_renormd);

                carX[0] += (int) ((tmp_renorml >> (MANT_DIG - 3)) & 3) - 2;

                tmp_renorml &= ~(1ull << (MANT_DIG - 3));
                tmp_renorml |= 1ull << (MANT_DIG - 2);
                priX[0] = tmp_renormd;
            }
        }

        ///Add scalar to manually specified binned fp (Y += X)
        ///
        ///Performs the operation Y += X on an binned type Y
        ///
        ///@param incpriY stride within Y's primary vector (use every incpriY'th element)
        ///@param inccarY stride within Y's carry vector (use every inccarY'th element)
        void binned_dmdadd(const ftype X, const int incpriY, const int inccarY)
        {
            binned_dmdupdate(X, incpriY, inccarY);
            binned_dmddeposit(X, incpriY);
            binned_dmrenorm(incpriY, inccarY);
        }

        ///Convert manually specified binned fp to native double-precision (X -> Y)
        ///
        ///@param incpriX stride within X's primary vector (use every incpriX'th element)
        ///@param inccarX stride within X's carry vector (use every inccarX'th element)
        double binned_conv_double(const int incpriX, const int inccarX) const
        {
            int i = 0;

            const auto* const priX = pvec();
            const auto* const carX = cvec();

            if (ISNANINF(priX[0]))
            {
                return priX[0];
            }

            if (priX[0] == 0.0)
            {
                return 0.0;
            }

            double Y = 0.0;
            double scale_down;
            double scale_up;
            int scaled;
            const auto X_index = binned_index();
            const auto* const bins = binned_bins(X_index);
            if (X_index <= (3 * MANT_DIG) / BIN_WIDTH)
            {
                scale_down = ldexp_impl(0.5, 1 - (2 * MANT_DIG - BIN_WIDTH));
                scale_up = ldexp_impl(0.5, 1 + (2 * MANT_DIG - BIN_WIDTH));
                scaled = std::max(
                    std::min(FOLD, (3 * MANT_DIG) / BIN_WIDTH - X_index), 0);
                if (X_index == 0)
                {
                    Y += carX[0] * ((bins[0] / 6.0) * scale_down * EXPANSION);
                    Y += carX[inccarX] * ((bins[1] / 6.0) * scale_down);
                    Y += (priX[0] - bins[0]) * scale_down * EXPANSION;
                    i = 2;
                }
                else
                {
                    Y += carX[0] * ((bins[0] / 6.0) * scale_down);
                    i = 1;
                }
                for (; i < scaled; i++)
                {
                    Y += carX[i * inccarX] * ((bins[i] / 6.0) * scale_down);
                    Y += (priX[(i - 1) * incpriX] - bins[i - 1]) * scale_down;
                }
                if (i == FOLD)
                {
                    Y += (priX[(FOLD - 1) * incpriX] - bins[FOLD - 1]) *
                        scale_down;
                    return Y * scale_up;
                }
                if (std::isinf(Y * scale_up))
                {
                    return Y * scale_up;
                }
                Y *= scale_up;
                for (; i < FOLD; i++)
                {
                    Y += carX[i * inccarX] * (bins[i] / 6.0);
                    Y += priX[(i - 1) * incpriX] - bins[i - 1];
                }
                Y += priX[(FOLD - 1) * incpriX] - bins[FOLD - 1];
            }
            else
            {
                Y += carX[0] * (bins[0] / 6.0);
                for (i = 1; i < FOLD; i++)
                {
                    Y += carX[i * inccarX] * (bins[i] / 6.0);
                    Y += (priX[(i - 1) * incpriX] - bins[i - 1]);
                }
                Y += (priX[(FOLD - 1) * incpriX] - bins[FOLD - 1]);
            }
            return Y;
        }

        ///Convert manually specified binned fp to native single-precision (X -> Y)
        ///
        ///@param incpriX stride within X's primary vector (use every incpriX'th element)
        ///@param inccarX stride within X's carry vector (use every inccarX'th element)
        float binned_conv_single(const int incpriX, const int inccarX) const
        {
            int i = 0;
            double Y = 0.0;
            const auto* const priX = pvec();
            const auto* const carX = cvec();

            if (ISNANINF(priX[0]))
            {
                return priX[0];
            }

            if (priX[0] == 0.0)
            {
                return 0.0;
            }

            //Note that the following order of summation is in order of decreasing
            //exponent. The following code is specific to SBWIDTH=13, FLT_MANT_DIG=24, and
            //the number of carries equal to 1.
            const auto X_index = binned_index();
            const auto* const bins = binned_bins(X_index);
            if (X_index == 0)
            {
                Y += (double) carX[0] * (double) (bins[0] / 6.0) *
                    (double) EXPANSION;
                Y += (double) carX[inccarX] * (double) (bins[1] / 6.0);
                Y += (double) (priX[0] - bins[0]) * (double) EXPANSION;
                i = 2;
            }
            else
            {
                Y += (double) carX[0] * (double) (bins[0] / 6.0);
                i = 1;
            }
            for (; i < FOLD; i++)
            {
                Y += (double) carX[i * inccarX] * (double) (bins[i] / 6.0);
                Y += (double) (priX[(i - 1) * incpriX] - bins[i - 1]);
            }
            Y += (double) (priX[(FOLD - 1) * incpriX] - bins[FOLD - 1]);

            return (float) Y;
        }

        ///Add two manually specified binned fp (Y += X)
        ///Performs the operation Y += X
        ///
        ///@param other   Another binned fp of the same type
        ///@param incpriX stride within X's primary vector (use every incpriX'th element)
        ///@param inccarX stride within X's carry vector (use every inccarX'th element)
        ///@param incpriY stride within Y's primary vector (use every incpriY'th element)
        ///@param inccarY stride within Y's carry vector (use every inccarY'th element)
        void binned_dmdmadd(const ReproducibleFloatingAccumulator& other,
            const int incpriX, const int inccarX, const int incpriY,
            const int inccarY)
        {
            auto* const priX = pvec();
            auto* const carX = cvec();
            auto* const priY = other.pvec();
            auto* const carY = other.cvec();

            if (priX[0] == 0.0)
                return;

            if (priY[0] == 0.0)
            {
                for (int i = 0; i < FOLD; i++)
                {
                    priY[i * incpriY] = priX[i * incpriX];
                    carY[i * inccarY] = carX[i * inccarX];
                }
                return;
            }

            if (ISNANINF(priX[0]) || ISNANINF(priY[0]))
            {
                priY[0] += priX[0];
                return;
            }

            const auto X_index = binned_index(priX);
            const auto Y_index = binned_index(priY);
            const auto shift = Y_index - X_index;
            if (shift > 0)
            {
                const auto* const bins = binned_bins(Y_index);
                //shift Y upwards and add X to Y
                for (int i = FOLD - 1; i >= shift; i--)
                {
                    priY[i * incpriY] = priX[i * incpriX] +
                        (priY[(i - shift) * incpriY] - bins[i - shift]);
                    carY[i * inccarY] =
                        carX[i * inccarX] + carY[(i - shift) * inccarY];
                }
                for (int i = 0; i < shift && i < FOLD; i++)
                {
                    priY[i * incpriY] = priX[i * incpriX];
                    carY[i * inccarY] = carX[i * inccarX];
                }
            }
            else
            {
                const auto* const bins = binned_bins(X_index);
                //shift X upwards and add X to Y
                for (int i = 0 - shift; i < FOLD; i++)
                {
                    priY[i * incpriY] +=
                        priX[(i + shift) * incpriX] - bins[i + shift];
                    carY[i * inccarY] += carX[(i + shift) * inccarX];
                }
            }

            binned_dmrenorm(incpriY, inccarY);
        }

        ///Add two manually specified binned fp (Y += X)
        ///Performs the operation Y += X
        void binned_dbdbadd(const ReproducibleFloatingAccumulator& other)
        {
            binned_dmdmadd(other, 1, 1, 1, 1);
        }

    public:
        ///Set the binned fp to zero
        void zero()
        {
            data = {0};
        }

        ///Return the fold of the binned fp
        int fold() const
        {
            return FOLD;
        }

        ///Returns the number of reference bins. Used for judging memory usage.
        static constexpr size_t number_of_reference_bins()
        {
            return bins.size();
        }

        ///Accumulate an arithmetic @p x into the binned fp.
        ///NOTE: Casts @p x to the type of the binned fp
        template <typename U,
            typename std::enable_if<std::is_arithmetic<U>::value>::type* =
                nullptr>
        ReproducibleFloatingAccumulator& operator+=(const U x)
        {
            binned_dmdadd(static_cast<ftype>(x), 1, 1);
            return *this;
        }

        ///Accumulate-subtract an arithmetic @p x into the binned fp.
        ///NOTE: Casts @p x to the type of the binned fp
        template <typename U,
            typename std::enable_if<std::is_arithmetic<U>::value>::type* =
                nullptr>
        ReproducibleFloatingAccumulator& operator-=(const U x)
        {
            binned_dmdadd(-static_cast<ftype>(x), 1, 1);
            return *this;
        }

        ///Accumulate a binned fp @p x into the binned fp.
        ReproducibleFloatingAccumulator& operator+=(
            const ReproducibleFloatingAccumulator& other)
        {
            binned_dbdbadd(other);
            return *this;
        }

        ///Accumulate-subtract a binned fp @p x into the binned fp.
        ///NOTE: Makes a copy and performs arithmetic; slow.
        ReproducibleFloatingAccumulator& operator-=(
            const ReproducibleFloatingAccumulator& other)
        {
            const auto temp = -other;
            binned_dbdbadd(temp);
        }

        ///Determines if two binned fp are equal
        bool operator==(const ReproducibleFloatingAccumulator& other) const
        {
            return data == other.data;
        }

        ///Determines if two binned fp are not equal
        bool operator!=(const ReproducibleFloatingAccumulator& other) const
        {
            return !operator==(other);
        }

        ///Sets this binned fp equal to the arithmetic value @p x
        ///NOTE: Casts @p x to the type of the binned fp
        template <typename U,
            typename std::enable_if<std::is_arithmetic<U>::value>::type* =
                nullptr>
        ReproducibleFloatingAccumulator& operator=(const U x)
        {
            zero();
            binned_dmdadd(static_cast<ftype>(x), 1, 1);
            return *this;
        }

        ///Sets this binned fp equal to another binned fp
        ReproducibleFloatingAccumulator& operator=(
            const ReproducibleFloatingAccumulator<ftype, FOLD>& o)
        {
            data = o.data;
            return *this;
        }

        ///Returns the negative of this binned fp
        ///NOTE: Makes a copy and performs arithmetic; slow.
        ReproducibleFloatingAccumulator operator-()
        {
            constexpr int incpriX = 1;
            constexpr int inccarX = 1;
            ReproducibleFloatingAccumulator temp = *this;
            if (pvec()[0] != 0.0)
            {
                const auto* const bins = binned_bins(binned_index());
                for (int i = 0; i < FOLD; i++)
                {
                    temp.pvec()[i * incpriX] =
                        bins[i] - (pvec()[i * incpriX] - bins[i]);
                    temp.cvec()[i * inccarX] = -cvec()[i * inccarX];
                }
            }
            return temp;
        }

        ///Convert this binned fp into its native floating-point representation
        ftype conv() const
        {
            if (std::is_same<ftype, float>::value)
            {
                return binned_conv_single(1, 1);
            }
            else
            {
                return binned_conv_double(1, 1);
            }
        }

        ///@brief Get binned fp summation error bound
        ///
        ///This is a bound on the absolute error of a summation using binned types
        ///
        ///@param N           The number of single precision floating point summands
        ///@param max_abs_val The summand of maximum absolute value
        ///@param binned_sum  The value of the sum computed using binned types
        ///@return            The absolute error bound
        static constexpr ftype error_bound(
            const uint64_t N, const ftype max_abs_val, const ftype binned_sum)
        {
            const double X = std::abs(max_abs_val);
            const double S = std::abs(binned_sum);
            return static_cast<ftype>(std::max(X, ldexp_impl(0.5, MIN_EXP - 1)) *
                    ldexp_impl(0.5, (1 - FOLD) * BIN_WIDTH + 1) * N +
                ((7.0 * EPSILON) /
                    (1.0 - 6.0 * std::sqrt(static_cast<double>(EPSILON)) -
                        7.0 * EPSILON)) *
                    S);
        }

        ///Add @p x to the binned fp
        void add(const ftype x)
        {
            binned_dmdadd(x, 1, 1);
        }

        ///Add arithmetics in the range [first, last) to the binned fp
        ///
        ///@param first       Start of range
        ///@param last        End of range
        ///@param max_abs_val Maximum absolute value of any member of the range
        template <typename InputIt>
        void add(InputIt first, InputIt last, const ftype max_abs_val)
        {
            binned_dmdupdate(std::abs(max_abs_val), 1, 1);
            size_t count = 0;
            for (; first != last; first++, count++)
            {
                binned_dmddeposit(static_cast<ftype>(*first), 1);
                if (count == ENDURANCE)
                {
                    binned_dmrenorm(1, 1);
                    count = 0;
                }
            }
        }

        ///Add arithmetics in the range [first, last) to the binned fp
        ///
        ///NOTE: A maximum absolute value is calculated, so two passes are made over
        ///      the data
        ///
        ///@param first       Start of range
        ///@param last        End of range
        template <typename InputIt>
        void add(InputIt first, InputIt last)
        {
            const auto max_abs_val = *std::max_element(
                first, last, [](const auto& a, const auto& b) {
                    return std::abs(a) < std::abs(b);
                });
            add(first, last, static_cast<ftype>(max_abs_val));
        }

        ///Add @p N elements starting at @p input to the binned fp: [input, input+N)
        ///
        ///@param input       Start of the range
        ///@param N           Number of elements to add
        ///@param max_abs_val Maximum absolute value of any member of the range
        template <typename T,
            typename std::enable_if<std::is_arithmetic<T>::value>::type* =
                nullptr>
        void add(const T* input, const size_t N, const ftype max_abs_val)
        {
            if (N == 0)
            {
                return;
            }
            add(input, input + N, max_abs_val);
        }

        ///Add @p N elements starting at @p input to the binned fp: [input, input+N)
        ///
        ///NOTE: A maximum absolute value is calculated, so two passes are made over
        ///      the data
        ///
        ///@param input       Start of the range
        ///@param N           Number of elements to add
        template <typename T,
            typename std::enable_if<std::is_arithmetic<T>::value>::type* =
                nullptr>
        void add(const T* input, const size_t N)
        {
            if (N == 0)
            {
                return;
            }
            T max_abs_val = input[0];
            for (size_t i = 0; i < N; i++)
            {
                max_abs_val = std::max(max_abs_val, std::abs(input[i]));
            }
            add(input, N, max_abs_val);
        }

        //////////////////////////////////////
        //MANUAL OPERATIONS; USE WISELY
        //////////////////////////////////////

        ///Rebins for repeated accumulation of scalars with magnitude <= @p mav
        ///
        ///Once rebinned, `ENDURANCE` values <= @p mav can be added to the accumulator
        ///with `unsafe_add` after which `renorm()` must be called. See the source of
        ///`add()` for an example
        template <typename T,
            typename std::enable_if<std::is_arithmetic<T>::value>::type* =
                nullptr>
        void set_max_abs_val(const T mav)
        {
            binned_dmdupdate(std::abs(mav), 1, 1);
        }

        ///Add @p x to the binned fp
        ///
        ///This is intended to be used after a call to `set_max_abs_val()`
        void unsafe_add(const ftype x)
        {
            binned_dmddeposit(x, 1);
        }

        ///Renormalizes the binned fp
        ///
        ///This is intended to be used after a call to `set_max_abs_val()` and one or
        ///more calls to `unsafe_add()`
        void renorm()
        {
            binned_dmrenorm(1, 1);
        }
    };
}    // namespace RFA
