//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_DATAPAR_ITERATOR_HELPERS_SEP_29_2016_0111PM)
#define HPX_PARALLEL_DATAPAR_ITERATOR_HELPERS_SEP_29_2016_0111PM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR)

#include <hpx/parallel/datapar/detail/vc/iterator_helpers.hpp>
#include <hpx/parallel/datapar/detail/boost_simd/iterator_helpers.hpp>
#include <hpx/parallel/traits/vector_pack_alignment_size.hpp>
#include <hpx/parallel/traits/vector_pack_load_store.hpp>

#include <cstddef>
#include <type_traits>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace util { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Iter>
    struct data_alignment_impl
    {
        template <typename Iter>
        static HPX_FORCEINLINE std::size_t call(Iter const& it)
        {
            typedef typename std::iterator_traits<Iter>::value_type value_type;
            return reinterpret_cast<std::uintptr_t>(std::addressof(*it)) &
                (traits::vector_pack_alignment<Iter, value_type>::value - 1);
        }
    };

    template <typename Iter>
    HPX_FORCEINLINE std::size_t data_alignment(Iter const& it)
    {
        return data_alignment_impl<Iter>::call(it);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iter1, typename Iter2>
    struct iterators_datapar_compatible_impl
    {
        typedef typename hpx::util::decay<Iter1>::type iterator1_type;
        typedef typename hpx::util::decay<Iter2>::type iterator2_type;

        typedef typename std::iterator_traits<iterator1_type>::value_type
            value1_type;
        typedef typename std::iterator_traits<iterator2_type>::value_type
            value2_type;

        typedef std::integral_constant<bool,
                traits::vector_pack_size<Iter1, value1_type>::value ==
                    traits::vector_pack_size<Iter2, value2_type>::value &&
                traits::vector_pack_alignment<Iter1, value1_type>::value ==
                    traits::vector_pack_alignment<Iter2, value2_type>::value
            > type;
    };

    template <typename Iter1, typename Iter2>
    struct iterators_datapar_compatible
      : iterators_datapar_compatible_impl<Iter1, Iter2>::type
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iter, typename Enable = void>
    struct iterator_datapar_compatible_impl
      : std::is_arithmetic<typename std::iterator_traits<Iter>::value_type>
    {};

    template <typename Iter, typename Enable = void>
    struct iterator_datapar_compatible
      : std::false_type
    {};

    template <typename Iter>
    struct iterator_datapar_compatible<Iter,
            typename std::enable_if<
                hpx::traits::is_random_access_iterator<Iter>::value
            >::type>
      : iterator_datapar_compatible_impl<
            typename hpx::util::decay<Iter>::type
        >::type
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iter, typename V, typename Enable = void>
    struct store_on_exit
    {
        typedef typename traits::vector_pack_load<Iter, V>::vector_pack_type
            pack_type;

        store_on_exit(Iter const& iter)
          : value_(traits::vector_pack_load<Iter, V>::aligned(iter)),
            iter_(iter)
        {
        }
        ~store_on_exit()
        {
            traits::vector_pack_store<Iter>::aligned(value_, iter_);
        }

        pack_type* operator&() { return &value_; }
        pack_type const* operator&() const { return &value_; }

        pack_type value_;
        Iter iter_;
    };

    template <typename Iter, typename V>
    struct store_on_exit<Iter, V,
        typename std::enable_if<
            std::is_const<
                typename std::iterator_traits<Iter>::value_type
            >::value
        >::type>
    {
        typedef typename traits::vector_pack_load<Iter, V>::vector_pack_type
            pack_type;

        store_on_exit(Iter const& iter)
          : value_(traits::vector_pack_load<Iter, V>::aligned(iter))
        {
        }

        pack_type* operator&() { return &value_; }
        pack_type const* operator&() const { return &value_; }

        pack_type value_;
    };
}}}}

#endif
#endif

