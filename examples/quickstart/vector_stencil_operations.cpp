//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>

#include <hpx/include/parallel_for_each.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/vector.hpp>

#include <hpx/util/transform_iterator.hpp>

///////////////////////////////////////////////////////////////////////////////
// Define the vector types to be used.
HPX_REGISTER_VECTOR(double);

///////////////////////////////////////////////////////////////////////////////
template <typename T>
struct identity
{
    template <typename F>
    struct result;

    template <typename This, typename Iterator>
    struct result<This(Iterator)>
    {
        typedef T& type;
    };

    template <typename Iterator>
    T& operator()(Iterator it) const
    {
        return *it;
    }
};

///////////////////////////////////////////////////////////////////////////////
template <typename Iterator>
struct previous
{
    typedef Iterator iterator;
    typedef typename std::iterator_traits<Iterator>::reference reference;

    template <typename T>
    struct result;

    template <typename This, typename Iterator>
    struct result<This(Iterator)>
    {
        typedef reference type;
    };

    previous() {}

    previous(Iterator begin, Iterator value)
      : begin_(begin), value_(value)
    {}

    template <typename Iterator>
    reference operator()(Iterator const& it) const
    {
        if (it == begin_)
            return *value_;
        return *(it - 1);
    }

    Iterator base() const { return begin_; }
    Iterator value() const { return value_; }

private:
    friend class boost::serialization::access;

    template <typename Archive>
    void serialize(Archive& ar, unsigned)
    {
        ar & begin_ & value_;
    }

private:
    Iterator begin_;
    Iterator value_;
};

// namespace hpx { namespace traits
// {
//     template <typename Iterator>
//     struct transform_iterator_transformer_traits<
//         previous<Iterator>,
//         typename boost::enable_if<
//             typename segmented_iterator_traits<Iterator>::is_segmented_iterator
//         >::type>
//     {
//         template <typename T>
//         struct result;
//
//         template <typename This, typename Transformer, typename F>
//         struct result<This(Transformer, F)>
//         {
//             typedef typename F::template apply<Iterator>::type iterator;
//             typedef previous<iterator> type;
//         };
//
//         template <typename Transformer, typename F>
//         typename result<
//                 transform_iterator_transformer_traits(Transformer, F)
//             >::type
//         operator()(Transformer const& tr, F const&) const
//         {
//             typedef typename result<
//                     transform_iterator_transformer_traits(Transformer, F)
//                 >::type transformer;
//
//             typedef typename Transformer::iterator iterator;
//             typename F::template apply<iterator> f;
//             return transformer(f(tr.base()), f(tr.value()));
//         }
//     };
// }}

///////////////////////////////////////////////////////////////////////////////
template <typename Iterator>
struct next
{
    typedef Iterator iterator;
    typedef typename std::iterator_traits<Iterator>::reference reference;

    template <typename T>
    struct result;

    template <typename This, typename Iterator>
    struct result<This(Iterator)>
    {
        typedef reference type;
    };

    next() {}

    next(Iterator end, Iterator value)
      : value_(value), end_(end)
    {}

    template <typename Iterator>
    reference operator()(Iterator const& it) const
    {
        if (it == (end_ - 1))
            return *value_;
        return *(it + 1);
    }

    Iterator base() const { return end_; }
    Iterator value() const { return value_; }

private:
    friend class boost::serialization::access;

    template <typename Archive>
    void serialize(Archive& ar, unsigned)
    {
        ar & value_ & end_;
    }

private:
    Iterator value_;
    Iterator end_;
};

// namespace hpx { namespace traits
// {
//     template <typename Iterator>
//     struct transform_iterator_transformer_traits<
//         next<Iterator>,
//         typename boost::enable_if<
//             typename segmented_iterator_traits<Iterator>::is_segmented_iterator
//         >::type>
//     {
//         template <typename T>
//         struct result;
//
//         template <typename This, typename Transformer, typename F>
//         struct result<This(Transformer, F)>
//         {
//             typedef typename F::template apply<Iterator>::type iterator;
//             typedef next<iterator> type;
//         };
//
//         template <typename Transformer, typename F>
//         typename result<
//                 transform_iterator_transformer_traits(Transformer, F)
//             >::type
//         operator()(Transformer const& tr, F const& f) const
//         {
//             typedef typename result<
//                     transform_iterator_transformer_traits(Transformer, F)
//                 >::type transformer;
//
//             return transformer(tr.base(), tr.value());
//         }
//     };
// }}

///////////////////////////////////////////////////////////////////////////////
// print element to the left of current
template <typename Container>
void print_left_element(Container const& values)
{
    typedef typename Container::const_iterator base_iterator;
    typedef typename Container::value_type value_type;

    typedef previous<base_iterator> transformer_type;

    typedef hpx::util::transform_iterator<
        base_iterator, previous<base_iterator>
    > iterator;

    transformer_type transformer = transformer_type(
        std::begin(values), std::end(values));

    hpx::parallel::for_each(
        hpx::parallel::seq,
        iterator(std::begin(values), transformer),
        iterator(std::end(values)),
        [](value_type d)
        {
            hpx::cout << d << " ";
        });
    hpx::cout << std::endl;
}

// print element to the right of current
template <typename Container>
void print_right_element(Container const& values)
{
    typedef typename Container::const_iterator base_iterator;
    typedef typename Container::value_type value_type;

    typedef next<base_iterator> transformer_type;

    typedef hpx::util::transform_iterator<
        base_iterator, transformer_type
    > iterator;

    transformer_type transformer = transformer_type(
        std::begin(values), std::end(values));

    hpx::parallel::for_each(
        hpx::parallel::seq,
        iterator(std::begin(values), transformer),
        iterator(std::end(values)),
        [](value_type d)
        {
            hpx::cout << d << " ";
        });
    hpx::cout << std::endl;
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    // demonstrate use of std::vector
    {
        std::vector<double> values(10);
        std::iota(std::begin(values), std::end(values), 1.0);

        print_left_element(values);
        print_right_element(values);
    }

    // demonstrate use of hpx::vector
    {
        hpx::vector<double> values(10);
        std::iota(std::begin(values), std::end(values), 1.0);

        print_left_element(values);
        print_right_element(values);
    }

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // Initialize and run HPX
    return hpx::init(argc, argv);
}
