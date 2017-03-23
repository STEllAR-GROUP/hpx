//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_DETAIL_COLOCATED_HELPERS_FEB_04_2014_0828PM)
#define HPX_UTIL_DETAIL_COLOCATED_HELPERS_FEB_04_2014_0828PM

#include <hpx/config.hpp>
#include <hpx/runtime/agas/gva.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/serialization/unique_ptr.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/result_of.hpp>
#include <hpx/util/unused.hpp>

#include <memory>
#include <utility>

#include <boost/format.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util { namespace functional
{
    ///////////////////////////////////////////////////////////////////////////
    struct extract_locality
    {
        extract_locality() {}

        naming::id_type operator()(naming::id_type const& locality_id,
            naming::id_type const& id) const
        {
            if(locality_id == naming::invalid_id)
            {
                HPX_THROW_EXCEPTION(hpx::no_success,
                    "extract_locality::operator()",
                    boost::str(boost::format(
                        "could not resolve colocated locality for id(%1%)"
                    ) % id));
                return naming::invalid_id;
            }
            return locality_id;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename Bound, typename Continuation>
        struct apply_continuation_impl
        {
            HPX_MOVABLE_ONLY(apply_continuation_impl);
        public:
            typedef typename util::decay<Bound>::type bound_type;
            typedef typename util::decay<Continuation>::type continuation_type;

            apply_continuation_impl()
              : bound_(), cont_() {}

            template <typename Bound_, typename Continuation_>
            explicit apply_continuation_impl(
                    Bound_ && bound, Continuation_ && c)
              : bound_(std::forward<Bound_>(bound)),
                cont_(std::forward<Continuation_>(c))
            {}

            apply_continuation_impl(apply_continuation_impl && o)
              : bound_(std::move(o.bound_))
              , cont_(std::move(o.cont_))
            {}

            apply_continuation_impl &operator=(apply_continuation_impl && o)
            {
                bound_ = std::move(o.bound_);
                cont_ = std::move(o.cont_);
                return *this;
            }

            template <typename T>
            typename util::result_of<bound_type(naming::id_type, T)>::type
            operator()(naming::id_type lco, T && t)
            {
                typedef typename util::result_of<
                    bound_type(naming::id_type, T)
                >::type result_type;

                bound_.apply_c(std::move(cont_), lco, std::forward<T>(t));
                return result_type();
            }

        private:
            // serialization support
            friend class hpx::serialization::access;

            template <typename Archive>
            HPX_FORCEINLINE void save(Archive& ar, unsigned int const) const
            {
                ar & bound_ & cont_;
            }

            template <typename Archive>
            HPX_FORCEINLINE void load(Archive& ar, unsigned int const)
            {
                ar & bound_ & cont_;
            }

            HPX_SERIALIZATION_SPLIT_MEMBER();

            bound_type bound_;
            continuation_type cont_;
        };

        template <typename Bound>
        struct apply_continuation_impl<Bound, hpx::util::unused_type>
        {
            HPX_MOVABLE_ONLY(apply_continuation_impl);
        public:
            typedef typename util::decay<Bound>::type bound_type;

            apply_continuation_impl()
              : bound_() {}

            template <typename Bound_>
            explicit apply_continuation_impl(Bound_ && bound)
              : bound_(std::forward<Bound_>(bound))
            {}

            apply_continuation_impl(apply_continuation_impl && o)
              : bound_(std::move(o.bound_))
            {}

            apply_continuation_impl &operator=(apply_continuation_impl && o)
            {
                bound_ = std::move(o.bound_);
                return *this;
            }

            template <typename T>
            typename util::result_of<bound_type(naming::id_type, T)>::type
            operator()(naming::id_type lco, T && t)
            {
                typedef typename util::result_of<
                    bound_type(naming::id_type, T)
                >::type result_type;

                bound_.apply(lco, std::forward<T>(t));
                return result_type();
            }

        private:
            // serialization support
            friend class hpx::serialization::access;

            template <typename Archive>
            HPX_FORCEINLINE void save(Archive& ar, unsigned int const) const
            {
                ar & bound_;
            }

            template <typename Archive>
            HPX_FORCEINLINE void load(Archive& ar, unsigned int const)
            {
                ar & bound_;
            }

            HPX_SERIALIZATION_SPLIT_MEMBER();

            bound_type bound_;
        };
    }

    template <typename Bound>
    functional::detail::apply_continuation_impl<Bound, hpx::util::unused_type>
    apply_continuation(Bound && bound)
    {
        return functional::detail::apply_continuation_impl<Bound, hpx::util::unused_type>(
            std::forward<Bound>(bound));
    }

    template <typename Bound, typename Continuation>
    functional::detail::apply_continuation_impl<Bound, Continuation>
    apply_continuation(Bound && bound, Continuation && c)
    {
        return functional::detail::apply_continuation_impl<Bound, Continuation>(
                std::forward<Bound>(bound), std::forward<Continuation>(c));
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename Bound, typename Continuation>
        struct async_continuation_impl
        {
            HPX_MOVABLE_ONLY(async_continuation_impl);
        public:
            typedef typename util::decay<Bound>::type bound_type;
            typedef typename util::decay<Continuation>::type continuation_type;

            async_continuation_impl()
              : bound_(), cont_()
            {}

            template <typename Bound_, typename Continuation_>
            explicit async_continuation_impl(
                    Bound_ && bound, Continuation_ && c)
              : bound_(std::forward<Bound_>(bound)),
                cont_(std::forward<Continuation_>(c))
            {}

            async_continuation_impl(async_continuation_impl && o)
              : bound_(std::move(o.bound_))
              , cont_(std::move(o.cont_))
            {}

            async_continuation_impl &operator=(async_continuation_impl && o)
            {
                bound_ = std::move(o.bound_);
                cont_ = std::move(o.cont_);
                return *this;
            }

            template <typename T>
            typename util::result_of<bound_type(naming::id_type, T)>::type
            operator()(naming::id_type lco, T && t)
            {
                typedef typename util::result_of<
                    bound_type(naming::id_type, T)
                >::type result_type;

                bound_.apply_c(std::move(cont_), lco, std::forward<T>(t));
                return result_type();
            }

        private:
            // serialization support
            friend class hpx::serialization::access;

            template <typename Archive>
            HPX_FORCEINLINE void save(Archive& ar, unsigned int const) const
            {
                ar & bound_ & cont_;
            }

            template <typename Archive>
            HPX_FORCEINLINE void load(Archive& ar, unsigned int const)
            {
                ar & bound_ & cont_;
            }

            HPX_SERIALIZATION_SPLIT_MEMBER();

            bound_type bound_;
            continuation_type cont_;
        };

        template <typename Bound>
        struct async_continuation_impl<Bound, hpx::util::unused_type>
        {
            HPX_MOVABLE_ONLY(async_continuation_impl);
        public:
            typedef typename util::decay<Bound>::type bound_type;

            async_continuation_impl()
              : bound_()
            {}

            template <typename Bound_>
            explicit async_continuation_impl(Bound_ && bound)
              : bound_(std::forward<Bound_>(bound))
            {}

            async_continuation_impl(async_continuation_impl && o)
              : bound_(std::move(o.bound_))
            {}

            async_continuation_impl &operator=(async_continuation_impl && o)
            {
                bound_ = std::move(o.bound_);
                return *this;
            }

            template <typename T>
            typename util::result_of<bound_type(naming::id_type, T)>::type
            operator()(naming::id_type lco, T && t)
            {
                typedef typename util::result_of<
                    bound_type(naming::id_type, T)
                >::type result_type;

                bound_.apply_c(lco, lco, std::forward<T>(t));
                return result_type();
            }

        private:
            // serialization support
            friend class hpx::serialization::access;

            template <typename Archive>
            HPX_FORCEINLINE void save(Archive& ar, unsigned int const) const
            {
                ar & bound_;
            }

            template <typename Archive>
            HPX_FORCEINLINE void load(Archive& ar, unsigned int const)
            {
                ar & bound_;
            }

            HPX_SERIALIZATION_SPLIT_MEMBER();

            bound_type bound_;
        };
    }

    template <typename Bound>
    functional::detail::async_continuation_impl<Bound, hpx::util::unused_type>
    async_continuation(Bound && bound)
    {
        return functional::detail::async_continuation_impl<Bound, hpx::util::unused_type>(
            std::forward<Bound>(bound));
    }

    template <typename Bound, typename Continuation>
    functional::detail::async_continuation_impl<Bound, Continuation>
    async_continuation(Bound && bound, Continuation && c)
    {
        return functional::detail::async_continuation_impl<Bound, Continuation>(
            std::forward<Bound>(bound), std::forward<Continuation>(c));
    }
}}}

#endif
