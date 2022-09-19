//  Copyright (c) 2022 A Kishore Kumar
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <hpx/functional/detail/invoke.hpp>
#include <hpx/functional/invoke_result.hpp>
#include <cstdint>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace util { namespace detail {
    ///////////////////////////////////////////////////////////////////////////
    // TODO: Make it detect template even for implicit convertible operations
    template <typename T, typename Operation>
    using is_arithmetic_plus_reduction = std::integral_constant<bool,
        std::is_arithmetic_v<T> and std::is_same_v<Operation, std::plus<T>>>;

    template <typename T, typename Operation>
    using is_arithmetic_minus_reduction = std::integral_constant<bool,
        std::is_arithmetic_v<T> and std::is_same_v<Operation, std::minus<T>>>;

    template <typename T, typename Operation>
    using is_arithmetic_multiplies_reduction = std::integral_constant<bool,
        std::is_arithmetic_v<T> and
            std::is_same_v<Operation, std::multiplies<T>>>;

    template <typename T, typename Operation>
    using is_arithmetic_bit_and_reduction = std::integral_constant<bool,
        std::is_arithmetic_v<T> and std::is_same_v<Operation, std::bit_and<T>>>;

    template <typename T, typename Operation>
    using is_arithmetic_bit_or_reduction = std::integral_constant<bool,
        std::is_arithmetic_v<T> and std::is_same_v<Operation, std::bit_or<T>>>;

    template <typename T, typename Operation>
    using is_arithmetic_bit_xor_reduction = std::integral_constant<bool,
        std::is_arithmetic_v<T> and std::is_same_v<Operation, std::bit_xor<T>>>;

    template <typename T, typename Operation>
    using is_arithmetic_logical_and_reduction = std::integral_constant<bool,
        std::is_arithmetic_v<T> and
            std::is_same_v<Operation, std::logical_and<T>>>;

    template <typename T, typename Operation>
    using is_arithmetic_logical_or_reduction = std::integral_constant<bool,
        std::is_arithmetic_v<T> and
            std::is_same_v<Operation, std::logical_or<T>>>;

    template <typename T, typename Operation>
    using is_not_omp_reduction = std::integral_constant<bool,
        !is_arithmetic_plus_reduction<T, Operation>::value and
            !is_arithmetic_minus_reduction<T, Operation>::value and
            !is_arithmetic_multiplies_reduction<T, Operation>::value and
            !is_arithmetic_bit_and_reduction<T, Operation>::value and
            !is_arithmetic_bit_or_reduction<T, Operation>::value and
            !is_arithmetic_bit_xor_reduction<T, Operation>::value and
            !is_arithmetic_logical_and_reduction<T, Operation>::value and
            !is_arithmetic_logical_or_reduction<T, Operation>::value>;

    ///////////////////////////////////////////////////////////////////////////
    struct unseq_reduce_n
    {
        /* Will only be called when the iterators are all random access */
        template <typename Iter1, typename T, typename Convert, typename Reduce>
        HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
            is_arithmetic_plus_reduction<T, Reduce>::value, T>::type
        reduce(Iter1 it, std::size_t count, T init, Reduce /* */,
            Convert conv)
        {
            HPX_VECTOR_REDUCTION(+ : init)
            for (std::size_t i = 0; i < count; i++)
            {
                init += HPX_INVOKE(conv, *it);
                ++it;
            }
            return init;
        }

        template <typename Iter1, typename T, typename Convert, typename Reduce>
        HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
            is_arithmetic_minus_reduction<T, Reduce>::value, T>::type
        reduce(Iter1 it, std::size_t count, T init, Reduce /* */,
            Convert conv)
        {
            HPX_VECTOR_REDUCTION(- : init)
            for (std::size_t i = 0; i < count; i++)
            {
                init -= HPX_INVOKE(conv, *it);
                ++it;
            }
            return init;
        }

        template <typename Iter1, typename T, typename Convert, typename Reduce>
        HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
            is_arithmetic_multiplies_reduction<T, Reduce>::value, T>::type
        reduce(Iter1 it, std::size_t count, T init, Reduce /* */,
            Convert conv)
        {
            HPX_VECTOR_REDUCTION(* : init)
            for (std::size_t i = 0; i < count; i++)
            {
                init *= HPX_INVOKE(conv, *it);
                ++it;
            }
            return init;
        }

        template <typename Iter1, typename T, typename Convert, typename Reduce>
        HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
            is_arithmetic_bit_and_reduction<T, Reduce>::value, T>::type
        reduce(Iter1 it, std::size_t count, T init, Reduce /* */,
            Convert conv)
        {
            HPX_VECTOR_REDUCTION(& : init)
            for (std::size_t i = 0; i < count; i++)
            {
                init &= HPX_INVOKE(conv, *it);
                ++it;
            }
            return init;
        }

        template <typename Iter1, typename T, typename Convert, typename Reduce>
        HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
            is_arithmetic_bit_or_reduction<T, Reduce>::value, T>::type
        reduce(Iter1 it, std::size_t count, T init, Reduce /* */,
            Convert conv)
        {
            HPX_VECTOR_REDUCTION(| : init)
            for (std::size_t i = 0; i < count; i++)
            {
                init |= HPX_INVOKE(conv, *it);
                ++it;
            }
            return init;
        }

        template <typename Iter1, typename T, typename Convert, typename Reduce>
        HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
            is_arithmetic_bit_xor_reduction<T, Reduce>::value, T>::type
        reduce(Iter1 it, std::size_t count, T init, Reduce /* */,
            Convert conv)
        {
            HPX_VECTOR_REDUCTION(^ : init)
            for (std::size_t i = 0; i < count; i++)
            {
                init ^= HPX_INVOKE(conv, *it);
                ++it;
            }
            return init;
        }

        template <typename Iter1, typename T, typename Convert, typename Reduce>
        HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
            is_arithmetic_logical_and_reduction<T, Reduce>::value, T>::type
        reduce(Iter1 it, std::size_t count, T init, Reduce /* */,
            Convert conv)
        {
            HPX_VECTOR_REDUCTION(&& : init)
            for (std::size_t i = 0; i < count; i++)
            {
                init = HPX_INVOKE(conv, *it) && init;
                ++it;
            }
            return init;
        }

        template <typename Iter1, typename T, typename Convert, typename Reduce>
        HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
            is_arithmetic_logical_or_reduction<T, Reduce>::value, T>::type
        reduce(Iter1 it, std::size_t count, T init, Reduce /* */,
            Convert conv)
        {
            HPX_VECTOR_REDUCTION(|| : init)
            for (std::size_t i = 0; i < count; i++)
            {
                init = HPX_INVOKE(conv, *it) || init;
                ++it;
            }
            return init;
        }

        template <typename Iter1, typename T, typename Convert, typename Reduce>
        HPX_HOST_DEVICE HPX_FORCEINLINE static
            typename std::enable_if<is_not_omp_reduction<T, Reduce>::value,
                T>::type
            reduce(Iter1 it, std::size_t count, T init, Reduce r,
                Convert conv)
        {
            const std::size_t block_size = HPX_LANE_SIZE / sizeof(T);

            // To small, just run sequential
            if (count <= (block_size << 1))
            {
                for (std::size_t i = 0; i < count; i++)
                {
                    init = HPX_INVOKE(r, init, HPX_INVOKE(conv, *it));
                    ++it;
                }
            }
            else
            {
                alignas(HPX_LANE_SIZE) std::uint8_t block[HPX_LANE_SIZE];
                T* tblock = reinterpret_cast<T*>(block);
                /* Initialize block[i] = r(f(2*i), f(2*i + 1)) */
                for (std::size_t i = 0; i < (block_size << 1); i += 2)
                {
                    ::new (tblock + i / 2) T(HPX_INVOKE(
                        r, HPX_INVOKE(conv, *it), HPX_INVOKE(conv, *(it + 1))));
                    it += 2;
                }
                /* Vectorized loop */
                const std::size_t limit = block_size * (count / block_size);
                for (std::size_t i = (block_size << 1); i < limit;
                     i += block_size)
                {
                    HPX_VECTORIZE
                    for (std::size_t j = 0; j < block_size; j++)
                    {
                        tblock[j] = HPX_INVOKE(
                            r, tblock[j], HPX_INVOKE(conv, *(it + j)));
                    }
                    it += block_size;
                }
                /* Remainder */
                HPX_VECTORIZE
                for (std::size_t i = 0; i < count - limit; i++)
                {
                    tblock[i] = HPX_INVOKE(r, tblock[i], HPX_INVOKE(conv, *it));
                    ++it;
                }
                /* Merge */
                for (std::size_t i = 0; i < block_size; i++)
                {
                    init = HPX_INVOKE(r, init, tblock[i]);
                }
                /* Cleanup resources */
                HPX_VECTORIZE
                for (std::size_t i = 0; i < block_size; i++)
                {
                    tblock[i].~T();
                }
            }
            return init;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    struct unseq_binary_reduce_n
    {
        /* Will only be called when the iterators are all random access */
        template <typename Iter1, typename Iter2, typename T, typename Convert,
            typename Reduce>
        HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
            is_arithmetic_plus_reduction<T, Reduce>::value, T>::type
        reduce(Iter1 it1, Iter2 it2,
            std::size_t count, T init, Reduce /* */, Convert conv)
        {
            HPX_VECTOR_REDUCTION(+ : init)
            for (std::size_t i = 0; i < count; i++)
            {
                init += HPX_INVOKE(conv, *it1, *it2);
                ++it1, ++it2;
            }
            return init;
        }

        template <typename Iter1, typename Iter2, typename T, typename Convert,
            typename Reduce>
        HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
            is_arithmetic_minus_reduction<T, Reduce>::value, T>::type
        reduce(Iter1 it1, Iter2 it2,
            std::size_t count, T init, Reduce /* */, Convert conv)
        {
            HPX_VECTOR_REDUCTION(- : init)
            for (std::size_t i = 0; i < count; i++)
            {
                init -= HPX_INVOKE(conv, *it1, *it2);
                ++it1, ++it2;
            }
            return init;
        }

        template <typename Iter1, typename Iter2, typename T, typename Convert,
            typename Reduce>
        HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
            is_arithmetic_multiplies_reduction<T, Reduce>::value, T>::type
        reduce(Iter1 it1, Iter2 it2,
            std::size_t count, T init, Reduce /* */, Convert conv)
        {
            HPX_VECTOR_REDUCTION(* : init)
            for (std::size_t i = 0; i < count; i++)
            {
                init *= HPX_INVOKE(conv, *it1, *it2);
                ++it1, ++it2;
            }
            return init;
        }

        template <typename Iter1, typename Iter2, typename T, typename Convert,
            typename Reduce>
        HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
            is_arithmetic_bit_and_reduction<T, Reduce>::value, T>::type
        reduce(Iter1 it1, Iter2 it2,
            std::size_t count, T init, Reduce /* */, Convert conv)
        {
            HPX_VECTOR_REDUCTION(& : init)
            for (std::size_t i = 0; i < count; i++)
            {
                init &= HPX_INVOKE(conv, *it1, *it2);
                ++it1, ++it2;
            }
            return init;
        }

        template <typename Iter1, typename Iter2, typename T, typename Convert,
            typename Reduce>
        HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
            is_arithmetic_bit_or_reduction<T, Reduce>::value, T>::type
        reduce(Iter1 it1, Iter2 it2,
            std::size_t count, T init, Reduce /* */, Convert conv)
        {
            HPX_VECTOR_REDUCTION(| : init)
            for (std::size_t i = 0; i < count; i++)
            {
                init |= HPX_INVOKE(conv, *it1, *it2);
                ++it1, ++it2;
            }
            return init;
        }

        template <typename Iter1, typename Iter2, typename T, typename Convert,
            typename Reduce>
        HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
            is_arithmetic_bit_xor_reduction<T, Reduce>::value, T>::type
        reduce(Iter1 it1, Iter2 it2,
            std::size_t count, T init, Reduce /* */, Convert conv)
        {
            HPX_VECTOR_REDUCTION(^ : init)
            for (std::size_t i = 0; i < count; i++)
            {
                init ^= HPX_INVOKE(conv, *it1, *it2);
                ++it1, ++it2;
            }
            return init;
        }

        template <typename Iter1, typename Iter2, typename T, typename Convert,
            typename Reduce>
        HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
            is_arithmetic_logical_and_reduction<T, Reduce>::value, T>::type
        reduce(Iter1 it1, Iter2 it2,
            std::size_t count, T init, Reduce /* */, Convert conv)
        {
            HPX_VECTOR_REDUCTION(&& : init)
            for (std::size_t i = 0; i < count; i++)
            {
                init = HPX_INVOKE(conv, *it1, *it2) && init;
                ++it1, ++it2;
            }
            return init;
        }

        template <typename Iter1, typename Iter2, typename T, typename Convert,
            typename Reduce>
        HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
            is_arithmetic_logical_or_reduction<T, Reduce>::value, T>::type
        reduce(Iter1 it1, Iter2 it2,
            std::size_t count, T init, Reduce /* */, Convert conv)
        {
            HPX_VECTOR_REDUCTION(|| : init)
            for (std::size_t i = 0; i < count; i++)
            {
                init = HPX_INVOKE(conv, *it1, *it2) || init;
                ++it1, ++it2;
            }
            return init;
        }

        template <typename Iter1, typename Iter2, typename T, typename Convert,
            typename Reduce>
        HPX_HOST_DEVICE HPX_FORCEINLINE static
            typename std::enable_if<is_not_omp_reduction<T, Reduce>::value,
                T>::type
            reduce(Iter1 it1, Iter2 it2,
                std::size_t count, T init, Reduce r, Convert conv)
        {
            const std::size_t block_size = HPX_LANE_SIZE / (sizeof(T) * 8);

            // To small, just run sequential
            if (count <= (block_size << 1))
            {
                for (std::size_t i = 0; i < count; i++)
                {
                    init = HPX_INVOKE(r, init, HPX_INVOKE(conv, *it1, *it2));
                    ++it1, ++it2;
                }
            }
            else
            {
                alignas(HPX_LANE_SIZE) std::uint8_t block[HPX_LANE_SIZE];
                T* tblock = reinterpret_cast<T*>(block);
                /* Initialize block[i] = r(f(2*i), f(2*i + 1)) */
                for (std::size_t i = 0; i < block_size; i++)
                {
                    ::new (tblock + i)
                        T(HPX_INVOKE(r, HPX_INVOKE(conv, *it1, *it2),
                            HPX_INVOKE(conv, *(it1 + 1), *(it2 + 1))));
                    it1 += 2;
                    it2 += 2;
                }
                /* Vectorized loop */
                const std::size_t limit = block_size * (count / block_size);
                for (std::size_t i = (block_size << 1); i < limit;
                     i += block_size)
                {
                    HPX_VECTORIZE
                    for (std::size_t j = 0; j < block_size; j++)
                    {
                        tblock[j] = HPX_INVOKE(r, tblock[j],
                            HPX_INVOKE(conv, *(it1 + j), *(it2 + j)));
                    }
                    it1 += block_size;
                    it2 += block_size;
                }
                /* Remainder */
                HPX_VECTORIZE
                for (std::size_t i = 0; i < count - limit; i++)
                {
                    tblock[i] =
                        HPX_INVOKE(r, tblock[i], HPX_INVOKE(conv, *it1, *it2));
                    ++it1, ++it2;
                }
                /* Merge */
                for (std::size_t i = 0; i < block_size; i++)
                {
                    init = HPX_INVOKE(r, init, tblock[i]);
                }
                /* Cleanup resources */
                HPX_VECTORIZE
                for (std::size_t i = 0; i < block_size; i++)
                {
                    tblock[i].~T();
                }
            }
            return init;
        }
    };

}}}}    // namespace hpx::parallel::util::detail