//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_VALUE_OR_ERROR_FEB_28_2012_1220PM)
#define HPX_UTIL_VALUE_OR_ERROR_FEB_28_2012_1220PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/move.hpp>

#include <boost/exception_ptr.hpp>

#include <boost/mpl/vector.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/sizeof.hpp>
#include <boost/mpl/placeholders.hpp>
#include <boost/mpl/max_element.hpp>
#include <boost/mpl/deref.hpp>

#include <boost/math/common_factor_ct.hpp>
#include <boost/aligned_storage.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        struct max_alignment
        {
            template <typename State, typename Item>
            struct apply : boost::mpl::size_t<
                boost::math::static_lcm<
                    State::value, boost::alignment_of<Item>::value
                >::value>
            {};
        };

        template <typename Sequence, typename F>
        struct max_value
        {
            typedef typename boost::mpl::transform1<Sequence, F>::type transformed_;
            typedef typename boost::mpl::max_element<transformed_>::type max_it;

            typedef typename boost::mpl::deref<max_it>::type type;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    class value_or_error
    {
    protected:
        typedef T value_type;
        typedef boost::exception_ptr error_type;
        typedef boost::mpl::vector2<value_type, error_type> types;

        enum { has_value = true, has_error = false };

    public:
        // constructors
        value_or_error()
          : has_value_(has_value)
        {
            ::new (get_value_address()) value_type;
        }

        value_or_error(value_or_error const& rhs)
          : has_value_(rhs.has_value_)
        {
            if (rhs.stores_value()) {
                construct_value(rhs.get_value());
            }
            else {
                construct_error(rhs.get_error());
            }
        }

        value_or_error(BOOST_RV_REF(value_or_error) rhs)
          : has_value_(rhs.has_value_)
        {
            if (rhs.stores_value()) {
                construct_value(rhs.move_value());
            }
            else {
                construct_error(rhs.get_error());
            }
        }

        explicit value_or_error(BOOST_RV_REF(value_type) t)
          : has_value_(has_value)
        {
            construct_value(boost::move(t));
        }

        explicit value_or_error(error_type const& e)
          : has_value_(has_error)
        {
            construct_error(e);
        }

        ~value_or_error()
        {
            if (stores_value())
                destruct_value();
            else
                destruct_error();
        }

        // assignment from another value_or_error instance
        value_or_error& operator=(BOOST_COPY_ASSIGN_REF(value_or_error) rhs)
        {
            if (this != &rhs) {
                if (rhs.stores_value() != stores_value()) {
                    if (stores_value()) {
                        destruct_value();
                        construct_error(rhs.get_error());
                    }
                    else {
                        destruct_error();
                        construct_value(rhs.get_value());
                    }
                    has_value_ = rhs.stores_value();
                }
                else if (rhs.stores_value()) {
                    assign_value(rhs.get_value());
                }
                else {
                    assign_error(rhs.get_error());
                }
            }
            return *this;
        }
        value_or_error& operator=(BOOST_RV_REF(value_or_error) rhs)
        {
            if (this != &rhs) {
                if (rhs.stores_value() != stores_value()) {
                    if (stores_value()) {
                        destruct_value();
                        construct_error(rhs.get_error());
                    }
                    else {
                        destruct_error();
                        construct_value(rhs.move_value());
                    }
                    has_value_ = rhs.stores_value();
                }
                else if (rhs.stores_value()) {
                    assign_value(rhs.move_value());
                }
                else {
                    assign_error(rhs.get_error());
                }
            }
            return *this;
        }

        // assign from value or error type
        value_or_error& operator=(BOOST_COPY_ASSIGN_REF(value_type) t)
        {
            if (!stores_value()) {
                destruct_error();
                construct_value(t);
                has_value_ = has_value;
            }
            else {
                assign_value(t);
            }
            return *this;
        }
        value_or_error& operator=(BOOST_RV_REF(value_type) t)
        {
            if (!stores_value()) {
                destruct_error();
                construct_value(boost::move(t));
                has_value_ = has_value;
            }
            else {
                assign_value(boost::move(t));
            }
            return *this;
        }

        value_or_error& operator=(error_type const& e)
        {
            if (stores_value()) {
                destruct_value();
                construct_error(e);
                has_value_ = has_error;
            }
            else {
                assign_error(e);
            }
            return *this;
        }

        // what is currently stored
        bool stores_value() const
        {
            return has_value_;
        }

        // access stored data
#if !defined(BOOST_NO_RVALUE_REFERENCES)
#if __GNUC__ == 4 && __GNUC_MINOR__ == 4
        value_type move_value()
        {
            if (!stores_value()) {
                HPX_THROW_EXCEPTION(invalid_status,
                    "value_or_error::get_value",
                    "unexpected retrieval of value")
            }
            return boost::move(*get_value_address());
        }
#else
        value_type&& move_value()
        {
            if (!stores_value()) {
                HPX_THROW_EXCEPTION(invalid_status,
                    "value_or_error::get_value",
                    "unexpected retrieval of value")
            }
            return boost::move(*get_value_address());
        }
#endif
#else
        ::boost::rv<value_type>& move_value()
        {
            if (!stores_value()) {
                HPX_THROW_EXCEPTION(invalid_status,
                    "value_or_error::get_value",
                    "unexpected retrieval of value")
            }
            return boost::move(*get_value_address());
        }
#endif

        value_type& get_value()
        {
            if (!stores_value()) {
                HPX_THROW_EXCEPTION(invalid_status,
                    "value_or_error::get_value",
                    "unexpected retrieval of value")
            }
            return *get_value_address();
        }

        value_type const& get_value() const
        {
            if (!stores_value()) {
                HPX_THROW_EXCEPTION(invalid_status,
                    "value_or_error::get_value",
                    "unexpected retrieval of value")
            }
            return *get_value_address();
        }

        error_type& get_error()
        {
            if (stores_value()) {
                HPX_THROW_EXCEPTION(invalid_status,
                    "value_or_error::get_error",
                    "unexpected retrieval of error value")
            }
            return *get_error_address();
        }

        error_type const& get_error() const
        {
            if (stores_value()) {
                HPX_THROW_EXCEPTION(invalid_status,
                    "value_or_error::get_error",
                    "unexpected retrieval of error value")
            }
            return *get_error_address();
        }

    private:
        //
        void construct_value(value_type const& v)
        {
            ::new (get_value_address()) value_type(v);
        }

        void construct_value(BOOST_RV_REF(value_type) v)
        {
            ::new (get_error_address()) value_type(boost::move(v));
        }

        void construct_error(error_type const& e)
        {
            ::new (get_error_address()) error_type(e);
        }

        //
        void assign_value(value_type const& v)
        {
            *get_value_address() = v;
        }

        void assign_value(BOOST_RV_REF(value_type) v)
        {
            *get_value_address() = boost::move(v);
        }

        void assign_error(error_type const& e)
        {
            *get_error_address() = e;
        }

        //
        void destruct_value()
        {
            get_value_address()->~value_type();
        }

        void destruct_error()
        {
            get_error_address()->~error_type();
        }

        // determine the required alignment, define aligned storage of proper
        // size
        typedef typename boost::mpl::fold<
            types, boost::mpl::size_t<1>, detail::max_alignment
        >::type max_alignment;

        typedef typename detail::max_value<
            types, boost::mpl::sizeof_<boost::mpl::_1>
        >::type max_size;

        typedef boost::aligned_storage<
            max_size::value, max_alignment::value
        > storage_type;

        // type safe accessors to the stored data
        typedef typename boost::add_pointer<value_type>::type pointer;
        typedef typename boost::add_pointer<value_type const>::type const_pointer;

        pointer get_value_address()
        {
            return static_cast<pointer>(data_.address());
        }
        const_pointer get_value_address() const
        {
            return static_cast<const_pointer>(data_.address());
        }

        // type safe accessors to the stored error data
        typedef typename boost::add_pointer<error_type>::type error_pointer;
        typedef typename boost::add_pointer<error_type const>::type const_error_pointer;

        error_pointer get_error_address()
        {
            return static_cast<error_pointer>(data_.address());
        }
        const_error_pointer get_error_address() const
        {
            return static_cast<const_error_pointer>(data_.address());
        }

        // member data
        storage_type data_;         // protected data
        bool has_value_;            // true if T, false if error

        BOOST_COPYABLE_AND_MOVABLE(value_or_error)
    };
}}

#endif
