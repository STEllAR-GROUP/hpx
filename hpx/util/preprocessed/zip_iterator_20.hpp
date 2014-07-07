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
    class zip_iterator<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7>
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
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8>
        struct dereference_iterator<tuple<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8
        > >
        {
            typedef typename zip_iterator_reference<tuple<
                T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8
            > >::type result_type;
            static result_type call(
                tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8> const& iterators)
            {
                return util::forward_as_tuple(
                    *util::get< 0>(iterators) , *util::get< 1>(iterators) , *util::get< 2>(iterators) , *util::get< 3>(iterators) , *util::get< 4>(iterators) , *util::get< 5>(iterators) , *util::get< 6>(iterators) , *util::get< 7>(iterators) , *util::get< 8>(iterators));
            }
        };
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8>
    class zip_iterator<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8>
      : public detail::zip_iterator_base<tuple<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8
        > >
    {
        typedef detail::zip_iterator_base<tuple<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8
        > > base_type;
    public:
        zip_iterator() : base_type() {}
        explicit zip_iterator(
            T0 const& v0 , T1 const& v1 , T2 const& v2 , T3 const& v3 , T4 const& v4 , T5 const& v5 , T6 const& v6 , T7 const& v7 , T8 const& v8
        ) : base_type(util::tie(v0 , v1 , v2 , v3 , v4 , v5 , v6 , v7 , v8))
        {}
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8>
    zip_iterator<typename decay<T0>::type , typename decay<T1>::type , typename decay<T2>::type , typename decay<T3>::type , typename decay<T4>::type , typename decay<T5>::type , typename decay<T6>::type , typename decay<T7>::type , typename decay<T8>::type>
    make_zip_iterator(T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8)
    {
        typedef zip_iterator<
            typename decay<T0>::type , typename decay<T1>::type , typename decay<T2>::type , typename decay<T3>::type , typename decay<T4>::type , typename decay<T5>::type , typename decay<T6>::type , typename decay<T7>::type , typename decay<T8>::type
        > result_type;
        return result_type(std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ));
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9>
        struct dereference_iterator<tuple<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9
        > >
        {
            typedef typename zip_iterator_reference<tuple<
                T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9
            > >::type result_type;
            static result_type call(
                tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9> const& iterators)
            {
                return util::forward_as_tuple(
                    *util::get< 0>(iterators) , *util::get< 1>(iterators) , *util::get< 2>(iterators) , *util::get< 3>(iterators) , *util::get< 4>(iterators) , *util::get< 5>(iterators) , *util::get< 6>(iterators) , *util::get< 7>(iterators) , *util::get< 8>(iterators) , *util::get< 9>(iterators));
            }
        };
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9>
    class zip_iterator<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9>
      : public detail::zip_iterator_base<tuple<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9
        > >
    {
        typedef detail::zip_iterator_base<tuple<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9
        > > base_type;
    public:
        zip_iterator() : base_type() {}
        explicit zip_iterator(
            T0 const& v0 , T1 const& v1 , T2 const& v2 , T3 const& v3 , T4 const& v4 , T5 const& v5 , T6 const& v6 , T7 const& v7 , T8 const& v8 , T9 const& v9
        ) : base_type(util::tie(v0 , v1 , v2 , v3 , v4 , v5 , v6 , v7 , v8 , v9))
        {}
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9>
    zip_iterator<typename decay<T0>::type , typename decay<T1>::type , typename decay<T2>::type , typename decay<T3>::type , typename decay<T4>::type , typename decay<T5>::type , typename decay<T6>::type , typename decay<T7>::type , typename decay<T8>::type , typename decay<T9>::type>
    make_zip_iterator(T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9)
    {
        typedef zip_iterator<
            typename decay<T0>::type , typename decay<T1>::type , typename decay<T2>::type , typename decay<T3>::type , typename decay<T4>::type , typename decay<T5>::type , typename decay<T6>::type , typename decay<T7>::type , typename decay<T8>::type , typename decay<T9>::type
        > result_type;
        return result_type(std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ));
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10>
        struct dereference_iterator<tuple<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10
        > >
        {
            typedef typename zip_iterator_reference<tuple<
                T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10
            > >::type result_type;
            static result_type call(
                tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10> const& iterators)
            {
                return util::forward_as_tuple(
                    *util::get< 0>(iterators) , *util::get< 1>(iterators) , *util::get< 2>(iterators) , *util::get< 3>(iterators) , *util::get< 4>(iterators) , *util::get< 5>(iterators) , *util::get< 6>(iterators) , *util::get< 7>(iterators) , *util::get< 8>(iterators) , *util::get< 9>(iterators) , *util::get< 10>(iterators));
            }
        };
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10>
    class zip_iterator<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10>
      : public detail::zip_iterator_base<tuple<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10
        > >
    {
        typedef detail::zip_iterator_base<tuple<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10
        > > base_type;
    public:
        zip_iterator() : base_type() {}
        explicit zip_iterator(
            T0 const& v0 , T1 const& v1 , T2 const& v2 , T3 const& v3 , T4 const& v4 , T5 const& v5 , T6 const& v6 , T7 const& v7 , T8 const& v8 , T9 const& v9 , T10 const& v10
        ) : base_type(util::tie(v0 , v1 , v2 , v3 , v4 , v5 , v6 , v7 , v8 , v9 , v10))
        {}
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10>
    zip_iterator<typename decay<T0>::type , typename decay<T1>::type , typename decay<T2>::type , typename decay<T3>::type , typename decay<T4>::type , typename decay<T5>::type , typename decay<T6>::type , typename decay<T7>::type , typename decay<T8>::type , typename decay<T9>::type , typename decay<T10>::type>
    make_zip_iterator(T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10)
    {
        typedef zip_iterator<
            typename decay<T0>::type , typename decay<T1>::type , typename decay<T2>::type , typename decay<T3>::type , typename decay<T4>::type , typename decay<T5>::type , typename decay<T6>::type , typename decay<T7>::type , typename decay<T8>::type , typename decay<T9>::type , typename decay<T10>::type
        > result_type;
        return result_type(std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ));
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11>
        struct dereference_iterator<tuple<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11
        > >
        {
            typedef typename zip_iterator_reference<tuple<
                T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11
            > >::type result_type;
            static result_type call(
                tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11> const& iterators)
            {
                return util::forward_as_tuple(
                    *util::get< 0>(iterators) , *util::get< 1>(iterators) , *util::get< 2>(iterators) , *util::get< 3>(iterators) , *util::get< 4>(iterators) , *util::get< 5>(iterators) , *util::get< 6>(iterators) , *util::get< 7>(iterators) , *util::get< 8>(iterators) , *util::get< 9>(iterators) , *util::get< 10>(iterators) , *util::get< 11>(iterators));
            }
        };
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11>
    class zip_iterator<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11>
      : public detail::zip_iterator_base<tuple<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11
        > >
    {
        typedef detail::zip_iterator_base<tuple<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11
        > > base_type;
    public:
        zip_iterator() : base_type() {}
        explicit zip_iterator(
            T0 const& v0 , T1 const& v1 , T2 const& v2 , T3 const& v3 , T4 const& v4 , T5 const& v5 , T6 const& v6 , T7 const& v7 , T8 const& v8 , T9 const& v9 , T10 const& v10 , T11 const& v11
        ) : base_type(util::tie(v0 , v1 , v2 , v3 , v4 , v5 , v6 , v7 , v8 , v9 , v10 , v11))
        {}
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11>
    zip_iterator<typename decay<T0>::type , typename decay<T1>::type , typename decay<T2>::type , typename decay<T3>::type , typename decay<T4>::type , typename decay<T5>::type , typename decay<T6>::type , typename decay<T7>::type , typename decay<T8>::type , typename decay<T9>::type , typename decay<T10>::type , typename decay<T11>::type>
    make_zip_iterator(T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10 , T11 && v11)
    {
        typedef zip_iterator<
            typename decay<T0>::type , typename decay<T1>::type , typename decay<T2>::type , typename decay<T3>::type , typename decay<T4>::type , typename decay<T5>::type , typename decay<T6>::type , typename decay<T7>::type , typename decay<T8>::type , typename decay<T9>::type , typename decay<T10>::type , typename decay<T11>::type
        > result_type;
        return result_type(std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ) , std::forward<T11>( v11 ));
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12>
        struct dereference_iterator<tuple<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12
        > >
        {
            typedef typename zip_iterator_reference<tuple<
                T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12
            > >::type result_type;
            static result_type call(
                tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12> const& iterators)
            {
                return util::forward_as_tuple(
                    *util::get< 0>(iterators) , *util::get< 1>(iterators) , *util::get< 2>(iterators) , *util::get< 3>(iterators) , *util::get< 4>(iterators) , *util::get< 5>(iterators) , *util::get< 6>(iterators) , *util::get< 7>(iterators) , *util::get< 8>(iterators) , *util::get< 9>(iterators) , *util::get< 10>(iterators) , *util::get< 11>(iterators) , *util::get< 12>(iterators));
            }
        };
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12>
    class zip_iterator<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12>
      : public detail::zip_iterator_base<tuple<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12
        > >
    {
        typedef detail::zip_iterator_base<tuple<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12
        > > base_type;
    public:
        zip_iterator() : base_type() {}
        explicit zip_iterator(
            T0 const& v0 , T1 const& v1 , T2 const& v2 , T3 const& v3 , T4 const& v4 , T5 const& v5 , T6 const& v6 , T7 const& v7 , T8 const& v8 , T9 const& v9 , T10 const& v10 , T11 const& v11 , T12 const& v12
        ) : base_type(util::tie(v0 , v1 , v2 , v3 , v4 , v5 , v6 , v7 , v8 , v9 , v10 , v11 , v12))
        {}
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12>
    zip_iterator<typename decay<T0>::type , typename decay<T1>::type , typename decay<T2>::type , typename decay<T3>::type , typename decay<T4>::type , typename decay<T5>::type , typename decay<T6>::type , typename decay<T7>::type , typename decay<T8>::type , typename decay<T9>::type , typename decay<T10>::type , typename decay<T11>::type , typename decay<T12>::type>
    make_zip_iterator(T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10 , T11 && v11 , T12 && v12)
    {
        typedef zip_iterator<
            typename decay<T0>::type , typename decay<T1>::type , typename decay<T2>::type , typename decay<T3>::type , typename decay<T4>::type , typename decay<T5>::type , typename decay<T6>::type , typename decay<T7>::type , typename decay<T8>::type , typename decay<T9>::type , typename decay<T10>::type , typename decay<T11>::type , typename decay<T12>::type
        > result_type;
        return result_type(std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ) , std::forward<T11>( v11 ) , std::forward<T12>( v12 ));
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13>
        struct dereference_iterator<tuple<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13
        > >
        {
            typedef typename zip_iterator_reference<tuple<
                T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13
            > >::type result_type;
            static result_type call(
                tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13> const& iterators)
            {
                return util::forward_as_tuple(
                    *util::get< 0>(iterators) , *util::get< 1>(iterators) , *util::get< 2>(iterators) , *util::get< 3>(iterators) , *util::get< 4>(iterators) , *util::get< 5>(iterators) , *util::get< 6>(iterators) , *util::get< 7>(iterators) , *util::get< 8>(iterators) , *util::get< 9>(iterators) , *util::get< 10>(iterators) , *util::get< 11>(iterators) , *util::get< 12>(iterators) , *util::get< 13>(iterators));
            }
        };
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13>
    class zip_iterator<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13>
      : public detail::zip_iterator_base<tuple<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13
        > >
    {
        typedef detail::zip_iterator_base<tuple<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13
        > > base_type;
    public:
        zip_iterator() : base_type() {}
        explicit zip_iterator(
            T0 const& v0 , T1 const& v1 , T2 const& v2 , T3 const& v3 , T4 const& v4 , T5 const& v5 , T6 const& v6 , T7 const& v7 , T8 const& v8 , T9 const& v9 , T10 const& v10 , T11 const& v11 , T12 const& v12 , T13 const& v13
        ) : base_type(util::tie(v0 , v1 , v2 , v3 , v4 , v5 , v6 , v7 , v8 , v9 , v10 , v11 , v12 , v13))
        {}
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13>
    zip_iterator<typename decay<T0>::type , typename decay<T1>::type , typename decay<T2>::type , typename decay<T3>::type , typename decay<T4>::type , typename decay<T5>::type , typename decay<T6>::type , typename decay<T7>::type , typename decay<T8>::type , typename decay<T9>::type , typename decay<T10>::type , typename decay<T11>::type , typename decay<T12>::type , typename decay<T13>::type>
    make_zip_iterator(T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10 , T11 && v11 , T12 && v12 , T13 && v13)
    {
        typedef zip_iterator<
            typename decay<T0>::type , typename decay<T1>::type , typename decay<T2>::type , typename decay<T3>::type , typename decay<T4>::type , typename decay<T5>::type , typename decay<T6>::type , typename decay<T7>::type , typename decay<T8>::type , typename decay<T9>::type , typename decay<T10>::type , typename decay<T11>::type , typename decay<T12>::type , typename decay<T13>::type
        > result_type;
        return result_type(std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ) , std::forward<T11>( v11 ) , std::forward<T12>( v12 ) , std::forward<T13>( v13 ));
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14>
        struct dereference_iterator<tuple<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14
        > >
        {
            typedef typename zip_iterator_reference<tuple<
                T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14
            > >::type result_type;
            static result_type call(
                tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14> const& iterators)
            {
                return util::forward_as_tuple(
                    *util::get< 0>(iterators) , *util::get< 1>(iterators) , *util::get< 2>(iterators) , *util::get< 3>(iterators) , *util::get< 4>(iterators) , *util::get< 5>(iterators) , *util::get< 6>(iterators) , *util::get< 7>(iterators) , *util::get< 8>(iterators) , *util::get< 9>(iterators) , *util::get< 10>(iterators) , *util::get< 11>(iterators) , *util::get< 12>(iterators) , *util::get< 13>(iterators) , *util::get< 14>(iterators));
            }
        };
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14>
    class zip_iterator<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14>
      : public detail::zip_iterator_base<tuple<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14
        > >
    {
        typedef detail::zip_iterator_base<tuple<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14
        > > base_type;
    public:
        zip_iterator() : base_type() {}
        explicit zip_iterator(
            T0 const& v0 , T1 const& v1 , T2 const& v2 , T3 const& v3 , T4 const& v4 , T5 const& v5 , T6 const& v6 , T7 const& v7 , T8 const& v8 , T9 const& v9 , T10 const& v10 , T11 const& v11 , T12 const& v12 , T13 const& v13 , T14 const& v14
        ) : base_type(util::tie(v0 , v1 , v2 , v3 , v4 , v5 , v6 , v7 , v8 , v9 , v10 , v11 , v12 , v13 , v14))
        {}
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14>
    zip_iterator<typename decay<T0>::type , typename decay<T1>::type , typename decay<T2>::type , typename decay<T3>::type , typename decay<T4>::type , typename decay<T5>::type , typename decay<T6>::type , typename decay<T7>::type , typename decay<T8>::type , typename decay<T9>::type , typename decay<T10>::type , typename decay<T11>::type , typename decay<T12>::type , typename decay<T13>::type , typename decay<T14>::type>
    make_zip_iterator(T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10 , T11 && v11 , T12 && v12 , T13 && v13 , T14 && v14)
    {
        typedef zip_iterator<
            typename decay<T0>::type , typename decay<T1>::type , typename decay<T2>::type , typename decay<T3>::type , typename decay<T4>::type , typename decay<T5>::type , typename decay<T6>::type , typename decay<T7>::type , typename decay<T8>::type , typename decay<T9>::type , typename decay<T10>::type , typename decay<T11>::type , typename decay<T12>::type , typename decay<T13>::type , typename decay<T14>::type
        > result_type;
        return result_type(std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ) , std::forward<T11>( v11 ) , std::forward<T12>( v12 ) , std::forward<T13>( v13 ) , std::forward<T14>( v14 ));
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15>
        struct dereference_iterator<tuple<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15
        > >
        {
            typedef typename zip_iterator_reference<tuple<
                T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15
            > >::type result_type;
            static result_type call(
                tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15> const& iterators)
            {
                return util::forward_as_tuple(
                    *util::get< 0>(iterators) , *util::get< 1>(iterators) , *util::get< 2>(iterators) , *util::get< 3>(iterators) , *util::get< 4>(iterators) , *util::get< 5>(iterators) , *util::get< 6>(iterators) , *util::get< 7>(iterators) , *util::get< 8>(iterators) , *util::get< 9>(iterators) , *util::get< 10>(iterators) , *util::get< 11>(iterators) , *util::get< 12>(iterators) , *util::get< 13>(iterators) , *util::get< 14>(iterators) , *util::get< 15>(iterators));
            }
        };
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15>
    class zip_iterator<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15>
      : public detail::zip_iterator_base<tuple<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15
        > >
    {
        typedef detail::zip_iterator_base<tuple<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15
        > > base_type;
    public:
        zip_iterator() : base_type() {}
        explicit zip_iterator(
            T0 const& v0 , T1 const& v1 , T2 const& v2 , T3 const& v3 , T4 const& v4 , T5 const& v5 , T6 const& v6 , T7 const& v7 , T8 const& v8 , T9 const& v9 , T10 const& v10 , T11 const& v11 , T12 const& v12 , T13 const& v13 , T14 const& v14 , T15 const& v15
        ) : base_type(util::tie(v0 , v1 , v2 , v3 , v4 , v5 , v6 , v7 , v8 , v9 , v10 , v11 , v12 , v13 , v14 , v15))
        {}
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15>
    zip_iterator<typename decay<T0>::type , typename decay<T1>::type , typename decay<T2>::type , typename decay<T3>::type , typename decay<T4>::type , typename decay<T5>::type , typename decay<T6>::type , typename decay<T7>::type , typename decay<T8>::type , typename decay<T9>::type , typename decay<T10>::type , typename decay<T11>::type , typename decay<T12>::type , typename decay<T13>::type , typename decay<T14>::type , typename decay<T15>::type>
    make_zip_iterator(T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10 , T11 && v11 , T12 && v12 , T13 && v13 , T14 && v14 , T15 && v15)
    {
        typedef zip_iterator<
            typename decay<T0>::type , typename decay<T1>::type , typename decay<T2>::type , typename decay<T3>::type , typename decay<T4>::type , typename decay<T5>::type , typename decay<T6>::type , typename decay<T7>::type , typename decay<T8>::type , typename decay<T9>::type , typename decay<T10>::type , typename decay<T11>::type , typename decay<T12>::type , typename decay<T13>::type , typename decay<T14>::type , typename decay<T15>::type
        > result_type;
        return result_type(std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ) , std::forward<T11>( v11 ) , std::forward<T12>( v12 ) , std::forward<T13>( v13 ) , std::forward<T14>( v14 ) , std::forward<T15>( v15 ));
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16>
        struct dereference_iterator<tuple<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16
        > >
        {
            typedef typename zip_iterator_reference<tuple<
                T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16
            > >::type result_type;
            static result_type call(
                tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16> const& iterators)
            {
                return util::forward_as_tuple(
                    *util::get< 0>(iterators) , *util::get< 1>(iterators) , *util::get< 2>(iterators) , *util::get< 3>(iterators) , *util::get< 4>(iterators) , *util::get< 5>(iterators) , *util::get< 6>(iterators) , *util::get< 7>(iterators) , *util::get< 8>(iterators) , *util::get< 9>(iterators) , *util::get< 10>(iterators) , *util::get< 11>(iterators) , *util::get< 12>(iterators) , *util::get< 13>(iterators) , *util::get< 14>(iterators) , *util::get< 15>(iterators) , *util::get< 16>(iterators));
            }
        };
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16>
    class zip_iterator<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16>
      : public detail::zip_iterator_base<tuple<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16
        > >
    {
        typedef detail::zip_iterator_base<tuple<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16
        > > base_type;
    public:
        zip_iterator() : base_type() {}
        explicit zip_iterator(
            T0 const& v0 , T1 const& v1 , T2 const& v2 , T3 const& v3 , T4 const& v4 , T5 const& v5 , T6 const& v6 , T7 const& v7 , T8 const& v8 , T9 const& v9 , T10 const& v10 , T11 const& v11 , T12 const& v12 , T13 const& v13 , T14 const& v14 , T15 const& v15 , T16 const& v16
        ) : base_type(util::tie(v0 , v1 , v2 , v3 , v4 , v5 , v6 , v7 , v8 , v9 , v10 , v11 , v12 , v13 , v14 , v15 , v16))
        {}
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16>
    zip_iterator<typename decay<T0>::type , typename decay<T1>::type , typename decay<T2>::type , typename decay<T3>::type , typename decay<T4>::type , typename decay<T5>::type , typename decay<T6>::type , typename decay<T7>::type , typename decay<T8>::type , typename decay<T9>::type , typename decay<T10>::type , typename decay<T11>::type , typename decay<T12>::type , typename decay<T13>::type , typename decay<T14>::type , typename decay<T15>::type , typename decay<T16>::type>
    make_zip_iterator(T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10 , T11 && v11 , T12 && v12 , T13 && v13 , T14 && v14 , T15 && v15 , T16 && v16)
    {
        typedef zip_iterator<
            typename decay<T0>::type , typename decay<T1>::type , typename decay<T2>::type , typename decay<T3>::type , typename decay<T4>::type , typename decay<T5>::type , typename decay<T6>::type , typename decay<T7>::type , typename decay<T8>::type , typename decay<T9>::type , typename decay<T10>::type , typename decay<T11>::type , typename decay<T12>::type , typename decay<T13>::type , typename decay<T14>::type , typename decay<T15>::type , typename decay<T16>::type
        > result_type;
        return result_type(std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ) , std::forward<T11>( v11 ) , std::forward<T12>( v12 ) , std::forward<T13>( v13 ) , std::forward<T14>( v14 ) , std::forward<T15>( v15 ) , std::forward<T16>( v16 ));
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17>
        struct dereference_iterator<tuple<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17
        > >
        {
            typedef typename zip_iterator_reference<tuple<
                T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17
            > >::type result_type;
            static result_type call(
                tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17> const& iterators)
            {
                return util::forward_as_tuple(
                    *util::get< 0>(iterators) , *util::get< 1>(iterators) , *util::get< 2>(iterators) , *util::get< 3>(iterators) , *util::get< 4>(iterators) , *util::get< 5>(iterators) , *util::get< 6>(iterators) , *util::get< 7>(iterators) , *util::get< 8>(iterators) , *util::get< 9>(iterators) , *util::get< 10>(iterators) , *util::get< 11>(iterators) , *util::get< 12>(iterators) , *util::get< 13>(iterators) , *util::get< 14>(iterators) , *util::get< 15>(iterators) , *util::get< 16>(iterators) , *util::get< 17>(iterators));
            }
        };
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17>
    class zip_iterator<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17>
      : public detail::zip_iterator_base<tuple<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17
        > >
    {
        typedef detail::zip_iterator_base<tuple<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17
        > > base_type;
    public:
        zip_iterator() : base_type() {}
        explicit zip_iterator(
            T0 const& v0 , T1 const& v1 , T2 const& v2 , T3 const& v3 , T4 const& v4 , T5 const& v5 , T6 const& v6 , T7 const& v7 , T8 const& v8 , T9 const& v9 , T10 const& v10 , T11 const& v11 , T12 const& v12 , T13 const& v13 , T14 const& v14 , T15 const& v15 , T16 const& v16 , T17 const& v17
        ) : base_type(util::tie(v0 , v1 , v2 , v3 , v4 , v5 , v6 , v7 , v8 , v9 , v10 , v11 , v12 , v13 , v14 , v15 , v16 , v17))
        {}
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17>
    zip_iterator<typename decay<T0>::type , typename decay<T1>::type , typename decay<T2>::type , typename decay<T3>::type , typename decay<T4>::type , typename decay<T5>::type , typename decay<T6>::type , typename decay<T7>::type , typename decay<T8>::type , typename decay<T9>::type , typename decay<T10>::type , typename decay<T11>::type , typename decay<T12>::type , typename decay<T13>::type , typename decay<T14>::type , typename decay<T15>::type , typename decay<T16>::type , typename decay<T17>::type>
    make_zip_iterator(T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10 , T11 && v11 , T12 && v12 , T13 && v13 , T14 && v14 , T15 && v15 , T16 && v16 , T17 && v17)
    {
        typedef zip_iterator<
            typename decay<T0>::type , typename decay<T1>::type , typename decay<T2>::type , typename decay<T3>::type , typename decay<T4>::type , typename decay<T5>::type , typename decay<T6>::type , typename decay<T7>::type , typename decay<T8>::type , typename decay<T9>::type , typename decay<T10>::type , typename decay<T11>::type , typename decay<T12>::type , typename decay<T13>::type , typename decay<T14>::type , typename decay<T15>::type , typename decay<T16>::type , typename decay<T17>::type
        > result_type;
        return result_type(std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ) , std::forward<T11>( v11 ) , std::forward<T12>( v12 ) , std::forward<T13>( v13 ) , std::forward<T14>( v14 ) , std::forward<T15>( v15 ) , std::forward<T16>( v16 ) , std::forward<T17>( v17 ));
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18>
        struct dereference_iterator<tuple<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18
        > >
        {
            typedef typename zip_iterator_reference<tuple<
                T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18
            > >::type result_type;
            static result_type call(
                tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18> const& iterators)
            {
                return util::forward_as_tuple(
                    *util::get< 0>(iterators) , *util::get< 1>(iterators) , *util::get< 2>(iterators) , *util::get< 3>(iterators) , *util::get< 4>(iterators) , *util::get< 5>(iterators) , *util::get< 6>(iterators) , *util::get< 7>(iterators) , *util::get< 8>(iterators) , *util::get< 9>(iterators) , *util::get< 10>(iterators) , *util::get< 11>(iterators) , *util::get< 12>(iterators) , *util::get< 13>(iterators) , *util::get< 14>(iterators) , *util::get< 15>(iterators) , *util::get< 16>(iterators) , *util::get< 17>(iterators) , *util::get< 18>(iterators));
            }
        };
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18>
    class zip_iterator<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18>
      : public detail::zip_iterator_base<tuple<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18
        > >
    {
        typedef detail::zip_iterator_base<tuple<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18
        > > base_type;
    public:
        zip_iterator() : base_type() {}
        explicit zip_iterator(
            T0 const& v0 , T1 const& v1 , T2 const& v2 , T3 const& v3 , T4 const& v4 , T5 const& v5 , T6 const& v6 , T7 const& v7 , T8 const& v8 , T9 const& v9 , T10 const& v10 , T11 const& v11 , T12 const& v12 , T13 const& v13 , T14 const& v14 , T15 const& v15 , T16 const& v16 , T17 const& v17 , T18 const& v18
        ) : base_type(util::tie(v0 , v1 , v2 , v3 , v4 , v5 , v6 , v7 , v8 , v9 , v10 , v11 , v12 , v13 , v14 , v15 , v16 , v17 , v18))
        {}
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18>
    zip_iterator<typename decay<T0>::type , typename decay<T1>::type , typename decay<T2>::type , typename decay<T3>::type , typename decay<T4>::type , typename decay<T5>::type , typename decay<T6>::type , typename decay<T7>::type , typename decay<T8>::type , typename decay<T9>::type , typename decay<T10>::type , typename decay<T11>::type , typename decay<T12>::type , typename decay<T13>::type , typename decay<T14>::type , typename decay<T15>::type , typename decay<T16>::type , typename decay<T17>::type , typename decay<T18>::type>
    make_zip_iterator(T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10 , T11 && v11 , T12 && v12 , T13 && v13 , T14 && v14 , T15 && v15 , T16 && v16 , T17 && v17 , T18 && v18)
    {
        typedef zip_iterator<
            typename decay<T0>::type , typename decay<T1>::type , typename decay<T2>::type , typename decay<T3>::type , typename decay<T4>::type , typename decay<T5>::type , typename decay<T6>::type , typename decay<T7>::type , typename decay<T8>::type , typename decay<T9>::type , typename decay<T10>::type , typename decay<T11>::type , typename decay<T12>::type , typename decay<T13>::type , typename decay<T14>::type , typename decay<T15>::type , typename decay<T16>::type , typename decay<T17>::type , typename decay<T18>::type
        > result_type;
        return result_type(std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ) , std::forward<T11>( v11 ) , std::forward<T12>( v12 ) , std::forward<T13>( v13 ) , std::forward<T14>( v14 ) , std::forward<T15>( v15 ) , std::forward<T16>( v16 ) , std::forward<T17>( v17 ) , std::forward<T18>( v18 ));
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19>
        struct dereference_iterator<tuple<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19
        > >
        {
            typedef typename zip_iterator_reference<tuple<
                T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19
            > >::type result_type;
            static result_type call(
                tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19> const& iterators)
            {
                return util::forward_as_tuple(
                    *util::get< 0>(iterators) , *util::get< 1>(iterators) , *util::get< 2>(iterators) , *util::get< 3>(iterators) , *util::get< 4>(iterators) , *util::get< 5>(iterators) , *util::get< 6>(iterators) , *util::get< 7>(iterators) , *util::get< 8>(iterators) , *util::get< 9>(iterators) , *util::get< 10>(iterators) , *util::get< 11>(iterators) , *util::get< 12>(iterators) , *util::get< 13>(iterators) , *util::get< 14>(iterators) , *util::get< 15>(iterators) , *util::get< 16>(iterators) , *util::get< 17>(iterators) , *util::get< 18>(iterators) , *util::get< 19>(iterators));
            }
        };
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19>
    class zip_iterator<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19>
      : public detail::zip_iterator_base<tuple<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19
        > >
    {
        typedef detail::zip_iterator_base<tuple<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19
        > > base_type;
    public:
        zip_iterator() : base_type() {}
        explicit zip_iterator(
            T0 const& v0 , T1 const& v1 , T2 const& v2 , T3 const& v3 , T4 const& v4 , T5 const& v5 , T6 const& v6 , T7 const& v7 , T8 const& v8 , T9 const& v9 , T10 const& v10 , T11 const& v11 , T12 const& v12 , T13 const& v13 , T14 const& v14 , T15 const& v15 , T16 const& v16 , T17 const& v17 , T18 const& v18 , T19 const& v19
        ) : base_type(util::tie(v0 , v1 , v2 , v3 , v4 , v5 , v6 , v7 , v8 , v9 , v10 , v11 , v12 , v13 , v14 , v15 , v16 , v17 , v18 , v19))
        {}
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19>
    zip_iterator<typename decay<T0>::type , typename decay<T1>::type , typename decay<T2>::type , typename decay<T3>::type , typename decay<T4>::type , typename decay<T5>::type , typename decay<T6>::type , typename decay<T7>::type , typename decay<T8>::type , typename decay<T9>::type , typename decay<T10>::type , typename decay<T11>::type , typename decay<T12>::type , typename decay<T13>::type , typename decay<T14>::type , typename decay<T15>::type , typename decay<T16>::type , typename decay<T17>::type , typename decay<T18>::type , typename decay<T19>::type>
    make_zip_iterator(T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10 , T11 && v11 , T12 && v12 , T13 && v13 , T14 && v14 , T15 && v15 , T16 && v16 , T17 && v17 , T18 && v18 , T19 && v19)
    {
        typedef zip_iterator<
            typename decay<T0>::type , typename decay<T1>::type , typename decay<T2>::type , typename decay<T3>::type , typename decay<T4>::type , typename decay<T5>::type , typename decay<T6>::type , typename decay<T7>::type , typename decay<T8>::type , typename decay<T9>::type , typename decay<T10>::type , typename decay<T11>::type , typename decay<T12>::type , typename decay<T13>::type , typename decay<T14>::type , typename decay<T15>::type , typename decay<T16>::type , typename decay<T17>::type , typename decay<T18>::type , typename decay<T19>::type
        > result_type;
        return result_type(std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ) , std::forward<T11>( v11 ) , std::forward<T12>( v12 ) , std::forward<T13>( v13 ) , std::forward<T14>( v14 ) , std::forward<T15>( v15 ) , std::forward<T16>( v16 ) , std::forward<T17>( v17 ) , std::forward<T18>( v18 ) , std::forward<T19>( v19 ));
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19 , typename T20>
        struct dereference_iterator<tuple<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19 , T20
        > >
        {
            typedef typename zip_iterator_reference<tuple<
                T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19 , T20
            > >::type result_type;
            static result_type call(
                tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19 , T20> const& iterators)
            {
                return util::forward_as_tuple(
                    *util::get< 0>(iterators) , *util::get< 1>(iterators) , *util::get< 2>(iterators) , *util::get< 3>(iterators) , *util::get< 4>(iterators) , *util::get< 5>(iterators) , *util::get< 6>(iterators) , *util::get< 7>(iterators) , *util::get< 8>(iterators) , *util::get< 9>(iterators) , *util::get< 10>(iterators) , *util::get< 11>(iterators) , *util::get< 12>(iterators) , *util::get< 13>(iterators) , *util::get< 14>(iterators) , *util::get< 15>(iterators) , *util::get< 16>(iterators) , *util::get< 17>(iterators) , *util::get< 18>(iterators) , *util::get< 19>(iterators) , *util::get< 20>(iterators));
            }
        };
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19 , typename T20>
    class zip_iterator<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19 , T20>
      : public detail::zip_iterator_base<tuple<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19 , T20
        > >
    {
        typedef detail::zip_iterator_base<tuple<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19 , T20
        > > base_type;
    public:
        zip_iterator() : base_type() {}
        explicit zip_iterator(
            T0 const& v0 , T1 const& v1 , T2 const& v2 , T3 const& v3 , T4 const& v4 , T5 const& v5 , T6 const& v6 , T7 const& v7 , T8 const& v8 , T9 const& v9 , T10 const& v10 , T11 const& v11 , T12 const& v12 , T13 const& v13 , T14 const& v14 , T15 const& v15 , T16 const& v16 , T17 const& v17 , T18 const& v18 , T19 const& v19 , T20 const& v20
        ) : base_type(util::tie(v0 , v1 , v2 , v3 , v4 , v5 , v6 , v7 , v8 , v9 , v10 , v11 , v12 , v13 , v14 , v15 , v16 , v17 , v18 , v19 , v20))
        {}
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19 , typename T20>
    zip_iterator<typename decay<T0>::type , typename decay<T1>::type , typename decay<T2>::type , typename decay<T3>::type , typename decay<T4>::type , typename decay<T5>::type , typename decay<T6>::type , typename decay<T7>::type , typename decay<T8>::type , typename decay<T9>::type , typename decay<T10>::type , typename decay<T11>::type , typename decay<T12>::type , typename decay<T13>::type , typename decay<T14>::type , typename decay<T15>::type , typename decay<T16>::type , typename decay<T17>::type , typename decay<T18>::type , typename decay<T19>::type , typename decay<T20>::type>
    make_zip_iterator(T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10 , T11 && v11 , T12 && v12 , T13 && v13 , T14 && v14 , T15 && v15 , T16 && v16 , T17 && v17 , T18 && v18 , T19 && v19 , T20 && v20)
    {
        typedef zip_iterator<
            typename decay<T0>::type , typename decay<T1>::type , typename decay<T2>::type , typename decay<T3>::type , typename decay<T4>::type , typename decay<T5>::type , typename decay<T6>::type , typename decay<T7>::type , typename decay<T8>::type , typename decay<T9>::type , typename decay<T10>::type , typename decay<T11>::type , typename decay<T12>::type , typename decay<T13>::type , typename decay<T14>::type , typename decay<T15>::type , typename decay<T16>::type , typename decay<T17>::type , typename decay<T18>::type , typename decay<T19>::type , typename decay<T20>::type
        > result_type;
        return result_type(std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ) , std::forward<T11>( v11 ) , std::forward<T12>( v12 ) , std::forward<T13>( v13 ) , std::forward<T14>( v14 ) , std::forward<T15>( v15 ) , std::forward<T16>( v16 ) , std::forward<T17>( v17 ) , std::forward<T18>( v18 ) , std::forward<T19>( v19 ) , std::forward<T20>( v20 ));
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19 , typename T20 , typename T21>
        struct dereference_iterator<tuple<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19 , T20 , T21
        > >
        {
            typedef typename zip_iterator_reference<tuple<
                T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19 , T20 , T21
            > >::type result_type;
            static result_type call(
                tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19 , T20 , T21> const& iterators)
            {
                return util::forward_as_tuple(
                    *util::get< 0>(iterators) , *util::get< 1>(iterators) , *util::get< 2>(iterators) , *util::get< 3>(iterators) , *util::get< 4>(iterators) , *util::get< 5>(iterators) , *util::get< 6>(iterators) , *util::get< 7>(iterators) , *util::get< 8>(iterators) , *util::get< 9>(iterators) , *util::get< 10>(iterators) , *util::get< 11>(iterators) , *util::get< 12>(iterators) , *util::get< 13>(iterators) , *util::get< 14>(iterators) , *util::get< 15>(iterators) , *util::get< 16>(iterators) , *util::get< 17>(iterators) , *util::get< 18>(iterators) , *util::get< 19>(iterators) , *util::get< 20>(iterators) , *util::get< 21>(iterators));
            }
        };
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19 , typename T20 , typename T21>
    class zip_iterator<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19 , T20 , T21>
      : public detail::zip_iterator_base<tuple<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19 , T20 , T21
        > >
    {
        typedef detail::zip_iterator_base<tuple<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19 , T20 , T21
        > > base_type;
    public:
        zip_iterator() : base_type() {}
        explicit zip_iterator(
            T0 const& v0 , T1 const& v1 , T2 const& v2 , T3 const& v3 , T4 const& v4 , T5 const& v5 , T6 const& v6 , T7 const& v7 , T8 const& v8 , T9 const& v9 , T10 const& v10 , T11 const& v11 , T12 const& v12 , T13 const& v13 , T14 const& v14 , T15 const& v15 , T16 const& v16 , T17 const& v17 , T18 const& v18 , T19 const& v19 , T20 const& v20 , T21 const& v21
        ) : base_type(util::tie(v0 , v1 , v2 , v3 , v4 , v5 , v6 , v7 , v8 , v9 , v10 , v11 , v12 , v13 , v14 , v15 , v16 , v17 , v18 , v19 , v20 , v21))
        {}
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19 , typename T20 , typename T21>
    zip_iterator<typename decay<T0>::type , typename decay<T1>::type , typename decay<T2>::type , typename decay<T3>::type , typename decay<T4>::type , typename decay<T5>::type , typename decay<T6>::type , typename decay<T7>::type , typename decay<T8>::type , typename decay<T9>::type , typename decay<T10>::type , typename decay<T11>::type , typename decay<T12>::type , typename decay<T13>::type , typename decay<T14>::type , typename decay<T15>::type , typename decay<T16>::type , typename decay<T17>::type , typename decay<T18>::type , typename decay<T19>::type , typename decay<T20>::type , typename decay<T21>::type>
    make_zip_iterator(T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10 , T11 && v11 , T12 && v12 , T13 && v13 , T14 && v14 , T15 && v15 , T16 && v16 , T17 && v17 , T18 && v18 , T19 && v19 , T20 && v20 , T21 && v21)
    {
        typedef zip_iterator<
            typename decay<T0>::type , typename decay<T1>::type , typename decay<T2>::type , typename decay<T3>::type , typename decay<T4>::type , typename decay<T5>::type , typename decay<T6>::type , typename decay<T7>::type , typename decay<T8>::type , typename decay<T9>::type , typename decay<T10>::type , typename decay<T11>::type , typename decay<T12>::type , typename decay<T13>::type , typename decay<T14>::type , typename decay<T15>::type , typename decay<T16>::type , typename decay<T17>::type , typename decay<T18>::type , typename decay<T19>::type , typename decay<T20>::type , typename decay<T21>::type
        > result_type;
        return result_type(std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ) , std::forward<T11>( v11 ) , std::forward<T12>( v12 ) , std::forward<T13>( v13 ) , std::forward<T14>( v14 ) , std::forward<T15>( v15 ) , std::forward<T16>( v16 ) , std::forward<T17>( v17 ) , std::forward<T18>( v18 ) , std::forward<T19>( v19 ) , std::forward<T20>( v20 ) , std::forward<T21>( v21 ));
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19 , typename T20 , typename T21 , typename T22>
        struct dereference_iterator<tuple<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19 , T20 , T21 , T22
        > >
        {
            typedef typename zip_iterator_reference<tuple<
                T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19 , T20 , T21 , T22
            > >::type result_type;
            static result_type call(
                tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19 , T20 , T21 , T22> const& iterators)
            {
                return util::forward_as_tuple(
                    *util::get< 0>(iterators) , *util::get< 1>(iterators) , *util::get< 2>(iterators) , *util::get< 3>(iterators) , *util::get< 4>(iterators) , *util::get< 5>(iterators) , *util::get< 6>(iterators) , *util::get< 7>(iterators) , *util::get< 8>(iterators) , *util::get< 9>(iterators) , *util::get< 10>(iterators) , *util::get< 11>(iterators) , *util::get< 12>(iterators) , *util::get< 13>(iterators) , *util::get< 14>(iterators) , *util::get< 15>(iterators) , *util::get< 16>(iterators) , *util::get< 17>(iterators) , *util::get< 18>(iterators) , *util::get< 19>(iterators) , *util::get< 20>(iterators) , *util::get< 21>(iterators) , *util::get< 22>(iterators));
            }
        };
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19 , typename T20 , typename T21 , typename T22>
    class zip_iterator
      : public detail::zip_iterator_base<tuple<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19 , T20 , T21 , T22
        > >
    {
        typedef detail::zip_iterator_base<tuple<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19 , T20 , T21 , T22
        > > base_type;
    public:
        zip_iterator() : base_type() {}
        explicit zip_iterator(
            T0 const& v0 , T1 const& v1 , T2 const& v2 , T3 const& v3 , T4 const& v4 , T5 const& v5 , T6 const& v6 , T7 const& v7 , T8 const& v8 , T9 const& v9 , T10 const& v10 , T11 const& v11 , T12 const& v12 , T13 const& v13 , T14 const& v14 , T15 const& v15 , T16 const& v16 , T17 const& v17 , T18 const& v18 , T19 const& v19 , T20 const& v20 , T21 const& v21 , T22 const& v22
        ) : base_type(util::tie(v0 , v1 , v2 , v3 , v4 , v5 , v6 , v7 , v8 , v9 , v10 , v11 , v12 , v13 , v14 , v15 , v16 , v17 , v18 , v19 , v20 , v21 , v22))
        {}
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19 , typename T20 , typename T21 , typename T22>
    zip_iterator<typename decay<T0>::type , typename decay<T1>::type , typename decay<T2>::type , typename decay<T3>::type , typename decay<T4>::type , typename decay<T5>::type , typename decay<T6>::type , typename decay<T7>::type , typename decay<T8>::type , typename decay<T9>::type , typename decay<T10>::type , typename decay<T11>::type , typename decay<T12>::type , typename decay<T13>::type , typename decay<T14>::type , typename decay<T15>::type , typename decay<T16>::type , typename decay<T17>::type , typename decay<T18>::type , typename decay<T19>::type , typename decay<T20>::type , typename decay<T21>::type , typename decay<T22>::type>
    make_zip_iterator(T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10 , T11 && v11 , T12 && v12 , T13 && v13 , T14 && v14 , T15 && v15 , T16 && v16 , T17 && v17 , T18 && v18 , T19 && v19 , T20 && v20 , T21 && v21 , T22 && v22)
    {
        typedef zip_iterator<
            typename decay<T0>::type , typename decay<T1>::type , typename decay<T2>::type , typename decay<T3>::type , typename decay<T4>::type , typename decay<T5>::type , typename decay<T6>::type , typename decay<T7>::type , typename decay<T8>::type , typename decay<T9>::type , typename decay<T10>::type , typename decay<T11>::type , typename decay<T12>::type , typename decay<T13>::type , typename decay<T14>::type , typename decay<T15>::type , typename decay<T16>::type , typename decay<T17>::type , typename decay<T18>::type , typename decay<T19>::type , typename decay<T20>::type , typename decay<T21>::type , typename decay<T22>::type
        > result_type;
        return result_type(std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ) , std::forward<T11>( v11 ) , std::forward<T12>( v12 ) , std::forward<T13>( v13 ) , std::forward<T14>( v14 ) , std::forward<T15>( v15 ) , std::forward<T16>( v16 ) , std::forward<T17>( v17 ) , std::forward<T18>( v18 ) , std::forward<T19>( v19 ) , std::forward<T20>( v20 ) , std::forward<T21>( v21 ) , std::forward<T22>( v22 ));
    }
}}
