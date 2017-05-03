//  Copyright (c)      2014 Thomas Heller
//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2007 Richard D. Guidry Jr.
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARCELSET_LOCALITY_HPP)
#define HPX_PARCELSET_LOCALITY_HPP

#include <hpx/config.hpp>

#include <hpx/exception.hpp>
#include <hpx/runtime/parcelset_fwd.hpp>
#include <hpx/runtime/serialization/serialization_fwd.hpp>
#include <hpx/runtime/serialization/map.hpp>
#include <hpx/traits/is_iterator.hpp>

#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset
{
    class HPX_EXPORT locality
    {
        template <typename Impl>
        class impl;

        class impl_base
        {
        public:
            virtual ~impl_base() {}

            virtual bool equal(impl_base const & rhs) const = 0;
            virtual bool less_than(impl_base const & rhs) const = 0;
            virtual bool valid() const = 0;
            virtual const char *type() const = 0;
            virtual std::ostream & print(std::ostream & os) const = 0;
            virtual void save(serialization::output_archive & ar) const = 0;
            virtual void load(serialization::input_archive & ar) = 0;
            virtual impl_base * clone() const = 0;
            virtual impl_base * move() = 0;

            template <typename Impl>
            Impl & get()
            {
                HPX_ASSERT(Impl::type() == type());
                return static_cast<impl<Impl>*>(this)->impl_;
            }

            template <typename Impl>
            Impl const & get() const
            {
                HPX_ASSERT(Impl::type() == type());
                return static_cast<const impl<Impl>*>(this)->impl_;
            }
        };

    public:
        locality()
        {}

        template <typename Impl, typename Enable =
            typename std::enable_if<
               !std::is_same<locality, typename util::decay<Impl>::type>::value &&
               !traits::is_iterator<Impl>::value
            >::type
        >
        locality(Impl && i)
          : impl_(new impl<typename util::decay<Impl>::type>(std::forward<Impl>(i)))
        {}

        locality(locality const & other)
          : impl_(other.impl_ ? other.impl_->clone() : nullptr)
        {
        }

        locality(locality && other)
          : impl_(other.impl_ ? other.impl_->move() : nullptr)
        {
        }

        locality & operator=(locality const & other)
        {
            if(this != &other)
            {
                if(other.impl_)
                {
                    impl_.reset(other.impl_->clone());
                }
                else
                {
                    impl_.reset();
                }
            }
            return *this;
        }

        locality & operator=(locality && other)
        {
            if(this != &other)
            {
                if(other.impl_)
                {
                    impl_.reset(other.impl_->move());
                }
                else
                {
                    impl_.reset();
                }
            }
            return *this;
        }

        ///////////////////////////////////////////////////////////////////////
        explicit operator bool() const HPX_NOEXCEPT
        {
            return impl_ ? impl_->valid(): false;
        }

        const char *type() const
        {
            return impl_ ? impl_->type() : "";
        }

        template <typename Impl>
        Impl & get()
        {
            HPX_ASSERT(impl_);
            return impl_->get<Impl>();
        }

        template <typename Impl>
        Impl const & get() const
        {
            HPX_ASSERT(impl_);
            return impl_->get<Impl>();
        }

    private:
        friend bool operator==(locality const& lhs, locality const& rhs)
        {
            if(lhs.impl_ == rhs.impl_) return true;
            if(!lhs.impl_ || !rhs.impl_) return false;
            return lhs.impl_->equal(*rhs.impl_);
        }

        friend bool operator!=(locality const& lhs, locality const& rhs)
        {
            return !(lhs == rhs);
        }

        friend bool operator< (locality const& lhs, locality const& rhs)
        {
            if(lhs.impl_ == rhs.impl_) return false;
            if(!lhs.impl_ || ! rhs.impl_) return false;
            return lhs.impl_->less_than(*rhs.impl_);
        }

        friend bool operator> (locality const& lhs, locality const& rhs)
        {
            if(lhs.impl_ == rhs.impl_) return false;
            if(!lhs.impl_ || ! rhs.impl_) return false;
            return !(lhs < rhs) && !(lhs == rhs);
        }

        friend std::ostream& operator<< (std::ostream& os, locality const& l)
        {
            if(!l.impl_) return os;
            return l.impl_->print(os);
        }

        // serialization support
        friend class hpx::serialization::access;

        void save(serialization::output_archive& ar, const unsigned int version) const;

        void load(serialization::input_archive& ar, const unsigned int version);

        HPX_SERIALIZATION_SPLIT_MEMBER();

        std::unique_ptr<impl_base> impl_;

        template <typename Impl>
        class impl : public impl_base
        {
        public:
            explicit impl(Impl && i) : impl_(std::move(i)) {}
            explicit impl(Impl const & i) : impl_(i) {}

            bool equal(impl_base const & rhs) const
            {
                if(type() != rhs.type()) return false;
                return impl_ == rhs.get<Impl>();
            }

            bool less_than(impl_base const & rhs) const
            {
                return type() < rhs.type() ||
                    (type() == rhs.type() && impl_ < rhs.get<Impl>());
            }

            bool valid() const
            {
                return !!impl_;
            }

            const char *type() const
            {
                return Impl::type();
            }

            std::ostream & print(std::ostream & os) const
            {
                os << impl_;
                return os;
            }

            void save(serialization::output_archive & ar) const
            {
                impl_.save(ar);
            }

            void load(serialization::input_archive & ar)
            {
                impl_.load(ar);
            }

            impl_base * clone() const
            {
                return new impl<Impl>(impl_);
            }

            impl_base * move()
            {
                return new impl<Impl>(std::move(impl_));
            }

            Impl impl_;
        };
    };

    typedef std::map<std::string, locality> endpoints_type;

    std::ostream& operator<< (std::ostream& os, endpoints_type const& endpoints);
}}

#include <hpx/config/warnings_suffix.hpp>

#endif
