// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


        
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename T0>
        struct dereference_iterator<tuple<
            T0
        > >
        {
            typedef typename zip_iterator_reference<tuple<
                T0
            > >::type result_type;
            static result_type call(
                tuple<T0> const& iterators)
            {
                return util::forward_as_tuple(
                    *util::get< 0>(iterators));
            }
        };
    }
    template <typename T0>
    class zip_iterator<T0>
      : public detail::zip_iterator_base<tuple<
            T0
        > >
    {
        typedef detail::zip_iterator_base<tuple<
            T0
        > > base_type;
    public:
        zip_iterator() : base_type() {}
        explicit zip_iterator(
            T0 const& v0
        ) : base_type(util::tie(v0))
        {}
    };
    template <typename T0>
    zip_iterator<typename decay<T0>::type>
    make_zip_iterator(T0 && v0)
    {
        typedef zip_iterator<
            typename decay<T0>::type
        > result_type;
        return result_type(std::forward<T0>( v0 ));
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename T0 , typename T1>
        struct dereference_iterator<tuple<
            T0 , T1
        > >
        {
            typedef typename zip_iterator_reference<tuple<
                T0 , T1
            > >::type result_type;
            static result_type call(
                tuple<T0 , T1> const& iterators)
            {
                return util::forward_as_tuple(
                    *util::get< 0>(iterators) , *util::get< 1>(iterators));
            }
        };
    }
    template <typename T0 , typename T1>
    class zip_iterator<T0 , T1>
      : public detail::zip_iterator_base<tuple<
            T0 , T1
        > >
    {
        typedef detail::zip_iterator_base<tuple<
            T0 , T1
        > > base_type;
    public:
        zip_iterator() : base_type() {}
        explicit zip_iterator(
            T0 const& v0 , T1 const& v1
        ) : base_type(util::tie(v0 , v1))
        {}
    };
    template <typename T0 , typename T1>
    zip_iterator<typename decay<T0>::type , typename decay<T1>::type>
    make_zip_iterator(T0 && v0 , T1 && v1)
    {
        typedef zip_iterator<
            typename decay<T0>::type , typename decay<T1>::type
        > result_type;
        return result_type(std::forward<T0>( v0 ) , std::forward<T1>( v1 ));
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename T0 , typename T1 , typename T2>
        struct dereference_iterator<tuple<
            T0 , T1 , T2
        > >
        {
            typedef typename zip_iterator_reference<tuple<
                T0 , T1 , T2
            > >::type result_type;
            static result_type call(
                tuple<T0 , T1 , T2> const& iterators)
            {
                return util::forward_as_tuple(
                    *util::get< 0>(iterators) , *util::get< 1>(iterators) , *util::get< 2>(iterators));
            }
        };
    }
    template <typename T0 , typename T1 , typename T2>
    class zip_iterator<T0 , T1 , T2>
      : public detail::zip_iterator_base<tuple<
            T0 , T1 , T2
        > >
    {
        typedef detail::zip_iterator_base<tuple<
            T0 , T1 , T2
        > > base_type;
    public:
        zip_iterator() : base_type() {}
        explicit zip_iterator(
            T0 const& v0 , T1 const& v1 , T2 const& v2
        ) : base_type(util::tie(v0 , v1 , v2))
        {}
    };
    template <typename T0 , typename T1 , typename T2>
    zip_iterator<typename decay<T0>::type , typename decay<T1>::type , typename decay<T2>::type>
    make_zip_iterator(T0 && v0 , T1 && v1 , T2 && v2)
    {
        typedef zip_iterator<
            typename decay<T0>::type , typename decay<T1>::type , typename decay<T2>::type
        > result_type;
        return result_type(std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ));
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename T0 , typename T1 , typename T2 , typename T3>
        struct dereference_iterator<tuple<
            T0 , T1 , T2 , T3
        > >
        {
            typedef typename zip_iterator_reference<tuple<
                T0 , T1 , T2 , T3
            > >::type result_type;
            static result_type call(
                tuple<T0 , T1 , T2 , T3> const& iterators)
            {
                return util::forward_as_tuple(
                    *util::get< 0>(iterators) , *util::get< 1>(iterators) , *util::get< 2>(iterators) , *util::get< 3>(iterators));
            }
        };
    }
    template <typename T0 , typename T1 , typename T2 , typename T3>
    class zip_iterator<T0 , T1 , T2 , T3>
      : public detail::zip_iterator_base<tuple<
            T0 , T1 , T2 , T3
        > >
    {
        typedef detail::zip_iterator_base<tuple<
            T0 , T1 , T2 , T3
        > > base_type;
    public:
        zip_iterator() : base_type() {}
        explicit zip_iterator(
            T0 const& v0 , T1 const& v1 , T2 const& v2 , T3 const& v3
        ) : base_type(util::tie(v0 , v1 , v2 , v3))
        {}
    };
    template <typename T0 , typename T1 , typename T2 , typename T3>
    zip_iterator<typename decay<T0>::type , typename decay<T1>::type , typename decay<T2>::type , typename decay<T3>::type>
    make_zip_iterator(T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3)
    {
        typedef zip_iterator<
            typename decay<T0>::type , typename decay<T1>::type , typename decay<T2>::type , typename decay<T3>::type
        > result_type;
        return result_type(std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ));
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
        struct dereference_iterator<tuple<
            T0 , T1 , T2 , T3 , T4
        > >
        {
            typedef typename zip_iterator_reference<tuple<
                T0 , T1 , T2 , T3 , T4
            > >::type result_type;
            static result_type call(
                tuple<T0 , T1 , T2 , T3 , T4> const& iterators)
            {
                return util::forward_as_tuple(
                    *util::get< 0>(iterators) , *util::get< 1>(iterators) , *util::get< 2>(iterators) , *util::get< 3>(iterators) , *util::get< 4>(iterators));
            }
        };
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
    class zip_iterator<T0 , T1 , T2 , T3 , T4>
      : public detail::zip_iterator_base<tuple<
            T0 , T1 , T2 , T3 , T4
        > >
    {
        typedef detail::zip_iterator_base<tuple<
            T0 , T1 , T2 , T3 , T4
        > > base_type;
    public:
        zip_iterator() : base_type() {}
        explicit zip_iterator(
            T0 const& v0 , T1 const& v1 , T2 const& v2 , T3 const& v3 , T4 const& v4
        ) : base_type(util::tie(v0 , v1 , v2 , v3 , v4))
        {}
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
    zip_iterator<typename decay<T0>::type , typename decay<T1>::type , typename decay<T2>::type , typename decay<T3>::type , typename decay<T4>::type>
    make_zip_iterator(T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4)
    {
        typedef zip_iterator<
            typename decay<T0>::type , typename decay<T1>::type , typename decay<T2>::type , typename decay<T3>::type , typename decay<T4>::type
        > result_type;
        return result_type(std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ));
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
        struct dereference_iterator<tuple<
            T0 , T1 , T2 , T3 , T4 , T5
        > >
        {
            typedef typename zip_iterator_reference<tuple<
                T0 , T1 , T2 , T3 , T4 , T5
            > >::type result_type;
            static result_type call(
                tuple<T0 , T1 , T2 , T3 , T4 , T5> const& iterators)
            {
                return util::forward_as_tuple(
                    *util::get< 0>(iterators) , *util::get< 1>(iterators) , *util::get< 2>(iterators) , *util::get< 3>(iterators) , *util::get< 4>(iterators) , *util::get< 5>(iterators));
            }
        };
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
    class zip_iterator<T0 , T1 , T2 , T3 , T4 , T5>
      : public detail::zip_iterator_base<tuple<
            T0 , T1 , T2 , T3 , T4 , T5
        > >
    {
        typedef detail::zip_iterator_base<tuple<
            T0 , T1 , T2 , T3 , T4 , T5
        > > base_type;
    public:
        zip_iterator() : base_type() {}
        explicit zip_iterator(
            T0 const& v0 , T1 const& v1 , T2 const& v2 , T3 const& v3 , T4 const& v4 , T5 const& v5
        ) : base_type(util::tie(v0 , v1 , v2 , v3 , v4 , v5))
        {}
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
    zip_iterator<typename decay<T0>::type , typename decay<T1>::type , typename decay<T2>::type , typename decay<T3>::type , typename decay<T4>::type , typename decay<T5>::type>
    make_zip_iterator(T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5)
    {
        typedef zip_iterator<
            typename decay<T0>::type , typename decay<T1>::type , typename decay<T2>::type , typename decay<T3>::type , typename decay<T4>::type , typename decay<T5>::type
        > result_type;
        return result_type(std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ));
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
        struct dereference_iterator<tuple<
            T0 , T1 , T2 , T3 , T4 , T5 , T6
        > >
        {
            typedef typename zip_iterator_reference<tuple<
                T0 , T1 , T2 , T3 , T4 , T5 , T6
            > >::type result_type;
            static result_type call(
                tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6> const& iterators)
            {
                return util::forward_as_tuple(
                    *util::get< 0>(iterators) , *util::get< 1>(iterators) , *util::get< 2>(iterators) , *util::get< 3>(iterators) , *util::get< 4>(iterators) , *util::get< 5>(iterators) , *util::get< 6>(iterators));
            }
        };
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
    class zip_iterator<T0 , T1 , T2 , T3 , T4 , T5 , T6>
      : public detail::zip_iterator_base<tuple<
            T0 , T1 , T2 , T3 , T4 , T5 , T6
        > >
    {
        typedef detail::zip_iterator_base<tuple<
            T0 , T1 , T2 , T3 , T4 , T5 , T6
        > > base_type;
    public:
        zip_iterator() : base_type() {}
        explicit zip_iterator(
            T0 const& v0 , T1 const& v1 , T2 const& v2 , T3 const& v3 , T4 const& v4 , T5 const& v5 , T6 const& v6
        ) : base_type(util::tie(v0 , v1 , v2 , v3 , v4 , v5 , v6))
        {}
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
    zip_iterator<typename decay<T0>::type , typename decay<T1>::type , typename decay<T2>::type , typename decay<T3>::type , typename decay<T4>::type , typename decay<T5>::type , typename decay<T6>::type>
    make_zip_iterator(T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6)
    {
        typedef zip_iterator<
            typename decay<T0>::type , typename decay<T1>::type , typename decay<T2>::type , typename decay<T3>::type , typename decay<T4>::type , typename decay<T5>::type , typename decay<T6>::type
        > result_type;
        return result_type(std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ));
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
        struct dereference_iterator<tuple<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7
        > >
        {
            typedef typename zip_iterator_reference<tuple<
                T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7
            > >::type result_type;
            static result_type call(
                tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7> const& iterators)
            {
                return util::forward_as_tuple(
                    *util::get< 0>(iterators) , *util::get< 1>(iterators) , *util::get< 2>(iterators) , *util::get< 3>(iterators) , *util::get< 4>(iterators) , *util::get< 5>(iterators) , *util::get< 6>(iterators) , *util::get< 7>(iterators));
            }
        };
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
    class zip_iterator
      : public detail::zip_iterator_base<tuple<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7
        > >
    {
        typedef detail::zip_iterator_base<tuple<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7
        > > base_type;
    public:
        zip_iterator() : base_type() {}
        explicit zip_iterator(
            T0 const& v0 , T1 const& v1 , T2 const& v2 , T3 const& v3 , T4 const& v4 , T5 const& v5 , T6 const& v6 , T7 const& v7
        ) : base_type(util::tie(v0 , v1 , v2 , v3 , v4 , v5 , v6 , v7))
        {}
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
    zip_iterator<typename decay<T0>::type , typename decay<T1>::type , typename decay<T2>::type , typename decay<T3>::type , typename decay<T4>::type , typename decay<T5>::type , typename decay<T6>::type , typename decay<T7>::type>
    make_zip_iterator(T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7)
    {
        typedef zip_iterator<
            typename decay<T0>::type , typename decay<T1>::type , typename decay<T2>::type , typename decay<T3>::type , typename decay<T4>::type , typename decay<T5>::type , typename decay<T6>::type , typename decay<T7>::type
        > result_type;
        return result_type(std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ));
    }
}}
