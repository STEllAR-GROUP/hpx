//  Copyright (c) 2016 Hartmut Kaiser
//  Copyright (c) 2016 Matthias Kretz
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_TRAITS_VECTOR_PACK_LOAD_VC_SEP_26_2016_0719PM)
#define HPX_PARALLEL_TRAITS_VECTOR_PACK_LOAD_VC_SEP_26_2016_0719PM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR_VC)

#include <cstddef>
#include <iterator>
#include <memory>

#include <Vc/Vc>

#if Vc_IS_VERSION_1

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Abi, typename NewT>
    struct rebind_pack<Vc::Vector<T, Abi>, NewT>
    {
        typedef Vc::Vector<NewT, Abi> type;
    };

    template <typename T, std::size_t N, typename V, std::size_t W, typename NewT>
    struct rebind_pack<Vc::SimdArray<T, N, V, W>, NewT>
    {
        typedef Vc::SimdArray<NewT, N, V, W> type;
    };

    template <typename T, typename NewT>
    struct rebind_pack<Vc::Scalar::Vector<T>, NewT>
    {
        typedef Vc::Scalar::Vector<NewT> type;
    };

    // don't wrap types twice
    template <typename T, typename Abi1, typename NewT, typename Abi2>
    struct rebind_pack<Vc::Vector<T, Abi1>, Vc::Vector<NewT, Abi2> >
    {
        typedef Vc::Vector<NewT, Abi2> type;
    };

    template <typename T, std::size_t N1, typename V1, std::size_t W1,
        typename NewT, std::size_t N2, typename V2, std::size_t W2>
    struct rebind_pack<Vc::SimdArray<T, N1, V1, W1>,
        Vc::SimdArray<NewT, N2, V2, W2> >
    {
        typedef Vc::SimdArray<NewT, N2, V2, W2> type;
    };

    template <typename T, typename NewT>
    struct rebind_pack<Vc::Scalar::Vector<T>, Vc::Scalar::Vector<NewT> >
    {
        typedef Vc::Scalar::Vector<NewT> type;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename V, typename ValueType, typename Enable>
    struct vector_pack_load
    {
        template <typename Iter>
        static typename rebind_pack<V, ValueType>::type
        aligned(Iter const& iter)
        {
            typedef typename rebind_pack<V, ValueType>::type vector_pack_type;
            return vector_pack_type(std::addressof(*iter), Vc::Aligned);
        }

        template <typename Iter>
        static typename rebind_pack<V, ValueType>::type
        unaligned(Iter const& iter)
        {
            typedef typename rebind_pack<V, ValueType>::type vector_pack_type;
            return vector_pack_type(std::addressof(*iter), Vc::Unaligned);
        }
    };

    template <typename V, typename T, typename Abi>
    struct vector_pack_load<V, Vc::Vector<T, Abi> >
    {
        template <typename Iter>
        static typename rebind_pack<V, Vc::Vector<T, Abi> >::type
        aligned(Iter const& iter)
        {
            return *iter;
        }

        template <typename Iter>
        static typename rebind_pack<V, Vc::Vector<T, Abi> >::type
        unaligned(Iter const& iter)
        {
            return *iter;
        }
    };

    template <typename Value, typename T, std::size_t N, typename V, std::size_t W>
    struct vector_pack_load<Value, Vc::SimdArray<T, N, V, W> >
    {
        template <typename Iter>
        static typename rebind_pack<Value, Vc::SimdArray<T, N, V, W> >::type
        aligned(Iter const& iter)
        {
            return *iter;
        }

        template <typename Iter>
        static typename rebind_pack<Value, Vc::SimdArray<T, N, V, W> >::type
        unaligned(Iter const& iter)
        {
            return *iter;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename V, typename ValueType, typename Enable>
    struct vector_pack_store
    {
        template <typename Iter>
        static void aligned(V const& value, Iter const& iter)
        {
            value.store(std::addressof(*iter), Vc::Aligned);
        }

        template <typename Iter>
        static void unaligned(V const& value, Iter const& iter)
        {
            value.store(std::addressof(*iter), Vc::Unaligned);
        }
    };

    template <typename V, typename T, typename Abi>
    struct vector_pack_store<V, Vc::Vector<T, Abi> >
    {
        template <typename Iter>
        static void aligned(V const& value, Iter const& iter)
        {
            *iter = value;
        }

        template <typename Iter>
        static void unaligned(V const& value, Iter const& iter)
        {
            *iter = value;
        }
    };

    template <typename Value, typename T, std::size_t N, typename V, std::size_t W>
    struct vector_pack_store<Value, Vc::SimdArray<T, N, V, W> >
    {
        template <typename Iter>
        static void aligned(Value const& value, Iter const& iter)
        {
            *iter = value;
        }

        template <typename Iter>
        static void unaligned(Value const& value, Iter const& iter)
        {
            *iter = value;
        }
    };
}}}

#else

#include <Vc/datapar>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Abi, typename NewT>
    struct rebind_pack<Vc::datapar<T, Abi>, NewT>
    {
        typedef Vc::datapar<NewT, Abi> type;
    };

    // don't wrap types twice
    template <typename T, typename Abi1, typename NewT, typename Abi2>
    struct rebind_pack<Vc::datapar<T, Abi1>, Vc::datapar<NewT, Abi2> >
    {
        typedef Vc::datapar<NewT, Abi2> type;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename V, typename ValueType, typename Enable>
    struct vector_pack_load
    {
        template <typename Iter>
        static typename rebind_pack<V, ValueType>::type
        aligned(Iter const& iter)
        {
            typedef typename rebind_pack<V, ValueType>::type vector_pack_type;
            return vector_pack_type(std::addressof(*iter), Vc::flags::vector_aligned_tag);
        }

        template <typename Iter>
        static typename rebind_pack<V, ValueType>::type
        unaligned(Iter const& iter)
        {
            typedef typename rebind_pack<V, ValueType>::type vector_pack_type;
            return vector_pack_type(std::addressof(*iter), Vc::flags::element_aligned_tag);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename V, typename ValueType, typename Enable>
    struct vector_pack_store
    {
        template <typename Iter>
        static void aligned(V const& value, Iter const& iter)
        {
            value.copy_to(std::addressof(*iter), Vc::flags::vector_aligned_tag);
        }

        template <typename Iter>
        static void unaligned(V const& value, Iter const& iter)
        {
            value.copy_to(std::addressof(*iter), Vc::flags::element_aligned_tag);
        }
    };
}}}

#endif  // Vc_IS_VERSION_1

#endif
#endif
