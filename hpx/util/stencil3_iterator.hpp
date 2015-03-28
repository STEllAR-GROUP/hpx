//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_STENCIL_ITERATOR_MAR_23_2015_1123AM)
#define HPX_UTIL_STENCIL_ITERATOR_MAR_23_2015_1123AM

#include <hpx/util/transform_iterator.hpp>
#include <hpx/util/zip_iterator.hpp>

#include <type_traits>
#include <iterator>

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Iterator,
        typename IterBegin = Iterator, typename IterValueBegin = Iterator,
        typename IterEnd = IterBegin, typename IterValueEnd = IterValueBegin>
    class stencil3_iterator;

    template <typename Iterator>
    class stencil3_iterator_nocheck;

    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename IteratorBase, typename IteratorValue>
        struct previous_transformer
        {
            template <typename T>
            struct result;

            template <typename This, typename Iterator>
            struct result<This(Iterator)>
            {
                typedef typename std::iterator_traits<Iterator>::reference type;
            };

            previous_transformer() {}

            // at position 'begin' it will dereference 'value', otherwise 'it-1'
            previous_transformer(IteratorBase const& begin, IteratorValue const& value)
              : begin_(begin), value_(value)
            {}

            template <typename Iterator>
            typename std::iterator_traits<Iterator>::reference
            operator()(Iterator const& it) const
            {
                if (it == begin_)
                    return *value_;
                return *(it - 1);
            }

        private:
            IteratorBase begin_;
            IteratorValue value_;
        };

        template <typename IteratorBase, typename IteratorValue>
        inline previous_transformer<IteratorBase, IteratorValue>
        make_previous_transformer(
            IteratorBase const& base, IteratorValue const& value)
        {
            return previous_transformer<IteratorBase, IteratorValue>(base, value);
        }

        // the user handles the left boundary case explicitly
        struct previous_transformer_nocheck
        {
            template <typename T>
            struct result;

            template <typename This, typename Iterator>
            struct result<This(Iterator)>
            {
                typedef typename std::iterator_traits<Iterator>::reference type;
            };

            // this transformer will always dereference 'it-1'
            template <typename Iterator>
            typename std::iterator_traits<Iterator>::reference
            operator()(Iterator const& it) const
            {
                return *(it - 1);
            }
        };

        inline previous_transformer_nocheck
        make_previous_transformer()
        {
            return previous_transformer_nocheck();
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename IteratorBase, typename IteratorValue>
        struct next_transformer
        {
            template <typename T>
            struct result;

            template <typename This, typename Iterator>
            struct result<This(Iterator)>
            {
                typedef typename std::iterator_traits<Iterator>::reference type;
            };

            next_transformer() {}

            // at position 'end' it will dereference 'value', otherwise 'it+1'
            next_transformer(IteratorBase const& end, IteratorValue const& value)
              : end_(end), value_(value)
            {}

            template <typename Iterator>
            typename std::iterator_traits<Iterator>::reference
            operator()(Iterator const& it) const
            {
                if (it == end_)
                    return *value_;
                return *(it + 1);
            }

        private:
            IteratorBase end_;
            IteratorValue value_;
        };

        template <typename IteratorBase, typename IteratorValue>
        inline next_transformer<IteratorBase, IteratorValue>
        make_next_transformer(
            IteratorBase const& base, IteratorValue const& value)
        {
            return next_transformer<IteratorBase, IteratorValue>(base, value);
        }

        // the user handles the right boundary case explicitly
        struct next_transformer_nocheck
        {
            template <typename T>
            struct result;

            template <typename This, typename Iterator>
            struct result<This(Iterator)>
            {
                typedef typename std::iterator_traits<Iterator>::reference type;
            };

            // this transformer will always dereference 'it+1'
            template <typename Iterator>
            typename std::iterator_traits<Iterator>::reference
            operator()(Iterator const& it) const
            {
                return *(it + 1);
            }
        };

        inline next_transformer_nocheck
        make_next_transformer()
        {
            return next_transformer_nocheck();
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename Iterator, typename IterBegin, typename IterValueBegin,
            typename IterEnd, typename IterValueEnd>
        struct stencil3_iterator_base
        {
            typedef previous_transformer<IterBegin, IterValueBegin> left_transformer;
            typedef next_transformer<IterEnd, IterValueEnd> right_transformer;

            typedef transform_iterator<Iterator, left_transformer> left_iterator;
            typedef transform_iterator<Iterator, right_transformer> right_iterator;

            typedef zip_iterator_base<
                    tuple<left_iterator, Iterator, right_iterator>,
                    stencil3_iterator<
                        Iterator, IterBegin, IterValueBegin, IterEnd, IterValueEnd
                    >
                > type;

            static type create(Iterator const& it,
                IterBegin const& begin, IterValueBegin const& begin_val,
                IterEnd const& end, IterValueEnd const& end_val)
            {
                auto prev = make_previous_transformer(begin, begin_val);
                auto next = make_next_transformer(end, end_val);

                return type(util::make_tuple(
                    make_transform_iterator(it, prev), it,
                    make_transform_iterator(it, next)));
            }

            static type create(Iterator const& it)
            {
                return type(util::make_tuple(
                    make_transform_iterator(it, left_transformer()), it,
                    make_transform_iterator(it, right_transformer())));
            }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iterator, typename IterBegin, typename IterValueBegin,
        typename IterEnd, typename IterValueEnd>
    class stencil3_iterator
      : public detail::stencil3_iterator_base<
            Iterator, IterBegin, IterValueBegin, IterEnd, IterValueEnd
        >::type
    {
    private:
        typedef detail::stencil3_iterator_base<
                Iterator, IterBegin, IterValueBegin, IterEnd, IterValueEnd
            > base_maker;
        typedef typename base_maker::type base_type;

    public:
        stencil3_iterator() {}

        stencil3_iterator(Iterator const& it,
                IterBegin const& begin, IterValueBegin const& begin_val,
                IterEnd const& end, IterValueEnd const& end_val)
          : base_type(base_maker::create(it, begin, begin_val, end, end_val))
        {}

        explicit stencil3_iterator(Iterator const& it)
          : base_type(base_maker::create(it))
        {}

        template <typename IterLeft, typename Iter, typename IterRight>
        stencil3_iterator(tuple<IterLeft, Iter, IterRight> const& t)
          : base_type(t)
        {}

    private:
        friend class boost::iterator_core_access;

        bool equal(stencil3_iterator const& other) const
        {
            return util::get<1>(this->get_iterator_tuple()) ==
                   util::get<1>(other.get_iterator_tuple());
        }
    };

    template <typename Iterator, typename IterBegin, typename IterValueBegin,
        typename IterEnd, typename IterValueEnd>
    inline stencil3_iterator<
        Iterator, IterBegin, IterValueBegin, IterEnd, IterValueEnd
    >
    make_stencil3_iterator(Iterator const& it,
        IterBegin const& begin, IterValueBegin const& begin_val,
        IterEnd const& end, IterValueEnd const& end_val)
    {
        typedef stencil3_iterator<
                Iterator, IterBegin, IterValueBegin, IterEnd, IterValueEnd
            > result_type;
        return result_type(it, begin, begin_val, end, end_val);
    }

    template <typename StencilIterator, typename Iterator>
    inline StencilIterator
    make_stencil3_iterator(Iterator const& it)
    {
        return StencilIterator(it);
    }

    template <typename Iterator, typename IterValue>
    inline std::pair<
        stencil3_iterator<Iterator, Iterator, IterValue, Iterator, IterValue>,
        stencil3_iterator<Iterator, Iterator, IterValue, Iterator, IterValue>
    >
    make_stencil3_range(Iterator const& begin, Iterator const& end,
        IterValue const& begin_val, IterValue const& end_val)
    {
        auto b = make_stencil3_iterator(begin, begin, begin_val, end-1, end_val);
        return std::make_pair(b, make_stencil3_iterator<decltype(b)>(end));
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename Iterator>
        struct stencil3_iterator_nocheck_base
        {
            typedef previous_transformer_nocheck left_transformer;
            typedef next_transformer_nocheck right_transformer;

            typedef transform_iterator<Iterator, left_transformer> left_iterator;
            typedef transform_iterator<Iterator, right_transformer> right_iterator;

            typedef zip_iterator_base<
                    tuple<left_iterator, Iterator, right_iterator>,
                    stencil3_iterator_nocheck<Iterator>
                > type;

            static type create(Iterator const& it)
            {
                return type(util::make_tuple(
                    make_transform_iterator(it, make_previous_transformer()), it,
                    make_transform_iterator(it, make_next_transformer())));
            }
        };
    }

    template <typename Iterator>
    class stencil3_iterator_nocheck
      : public detail::stencil3_iterator_nocheck_base<Iterator>::type
    {
    private:
        typedef detail::stencil3_iterator_nocheck_base<Iterator> base_maker;
        typedef typename base_maker::type base_type;

    public:
        stencil3_iterator_nocheck() {}

        explicit stencil3_iterator_nocheck(Iterator const& it)
          : base_type(base_maker::create(it))
        {}

        stencil3_iterator_nocheck(tuple<Iterator, Iterator, Iterator> const& t)
          : base_type(t)
        {}

    private:
        friend class boost::iterator_core_access;

        bool equal(stencil3_iterator_nocheck const& other) const
        {
            return util::get<1>(this->get_iterator_tuple()) ==
                   util::get<1>(other.get_iterator_tuple());
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iterator>
    inline stencil3_iterator_nocheck<Iterator>
    make_stencil3_iterator_nocheck(Iterator const& it)
    {
        return stencil3_iterator_nocheck<Iterator>(it);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iterator>
    inline std::pair<
        stencil3_iterator_nocheck<Iterator>,
        stencil3_iterator_nocheck<Iterator>
    >
    make_stencil3_range(Iterator const& begin, Iterator const& end)
    {
        return std::make_pair(
            make_stencil3_iterator_nocheck(begin),
            make_stencil3_iterator_nocheck(end));
    }
}}

#endif

