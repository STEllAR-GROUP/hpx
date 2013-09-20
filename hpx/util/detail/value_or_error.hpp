//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_DETAIL_VALUE_OR_ERROR_FEB_28_2012_1220PM)
#define HPX_UTIL_DETAIL_VALUE_OR_ERROR_FEB_28_2012_1220PM

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
namespace hpx { namespace util { namespace detail
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

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    class value_or_error
    {
    protected:
        typedef T value_type;
        typedef boost::exception_ptr error_type;
        typedef boost::mpl::vector2<value_type, error_type> types;

        enum state { has_none, has_value, has_error };

    public:
        // constructors
        value_or_error()
          : state_(has_none)
        {}

        value_or_error(value_or_error const& rhs)
          : state_(rhs.state_)
        {
            switch (rhs.state_)
            {
            case has_none:
                {
                    break;
                }
            case has_value:
                {
                    construct_value(rhs.get_value());
                    break;
                }
            case has_error:
                {
                    construct_error(rhs.get_error());
                    break;
                }
            }
        }

        value_or_error(BOOST_RV_REF(value_or_error) rhs)
          : state_(rhs.state_)
        {
            switch (rhs.state_)
            {
            case has_none:
                {
                    break;
                }
            case has_value:
                {
                    construct_value(rhs.move_value());
                    break;
                }
            case has_error:
                {
                    construct_error(rhs.get_error());
                    break;
                }
            }
        }

        explicit value_or_error(BOOST_RV_REF(value_type) t)
          : state_(has_value)
        {
            construct_value(boost::move(t));
        }

        explicit value_or_error(error_type const& e)
          : state_(has_error)
        {
            construct_error(e);
        }

        ~value_or_error()
        {
            destruct();
        }

        // assignment from another value_or_error instance
        value_or_error& operator=(BOOST_COPY_ASSIGN_REF(value_or_error) rhs)
        {
            if (this != &rhs) {
                destruct();

                state_ = rhs.state_;
                switch (rhs.state_)
                {
                case has_none:
                    {
                        break;
                    }
                case has_value:
                    {
                        construct_value(rhs.get_value());
                        break;
                    }
                case has_error:
                    {
                        construct_error(rhs.get_error());
                        break;
                    }
                }
            }
            return *this;
        }
        value_or_error& operator=(BOOST_RV_REF(value_or_error) rhs)
        {
            if (this != &rhs) {
                destruct();
                
                state_ = rhs.state_;
                switch (rhs.state_)
                {
                case has_none:
                    {
                        break;
                    }
                case has_value:
                    {
                        construct_value(rhs.move_value());
                        break;
                    }
                case has_error:
                    {
                        construct_error(rhs.get_error());
                        break;
                    }
                }
            }
            return *this;
        }

        // assign from value or error type
        value_or_error& operator=(BOOST_COPY_ASSIGN_REF(value_type) t)
        {
            destruct();
            
            state_ = has_value;
            construct_value(t);

            return *this;
        }
        value_or_error& operator=(BOOST_RV_REF(value_type) t)
        {
            destruct();
            
            state_ = has_value;
            construct_value(boost::move(t));

            return *this;
        }

        value_or_error& operator=(error_type const& e)
        {
            destruct();
            
            state_ = has_error;
            construct_error(e);

            return *this;
        }

        // what is currently stored
        bool is_empty() const
        {
            return states_ == has_none;
        }
        bool stores_value() const
        {
            return state_ == has_value;
        }
        bool stores_error() const
        {
            return state_ == has_error;
        }

        // access stored data
#if __GNUC__ == 4 && __GNUC_MINOR__ == 4
        value_type move_value()
        {
            if (!stores_value()) {
                HPX_THROW_EXCEPTION(invalid_status,
                    "value_or_error::move_value",
                    "unexpected retrieval of value")
            }
            return boost::move(*get_value_address());
        }
#else
        BOOST_RV_REF(value_type) move_value()
        {
            if (!stores_value()) {
                HPX_THROW_EXCEPTION(invalid_status,
                    "value_or_error::move_value",
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
            if (!stores_error()) {
                HPX_THROW_EXCEPTION(invalid_status,
                    "value_or_error::get_error",
                    "unexpected retrieval of error value")
            }
            return *get_error_address();
        }

        error_type const& get_error() const
        {
            if (!stores_error()) {
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

        void construct_value(BOOST_RV_REF(value_type) v) //-V659
        {
            ::new (get_error_address()) value_type(boost::move(v));
        }

        void construct_error(error_type const& e)
        {
            ::new (get_error_address()) error_type(e);
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

        void destruct()
        {
            switch (state_)
            {
            case has_none:
                {
                    break;
                }
            case has_value:
                {
                    destruct_value();
                    break;
                }
            case has_error:
                {
                    destruct_error();
                    break;
                }
            }
            state_ = has_none;
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
        state state_;            // none, value, or error

        BOOST_COPYABLE_AND_MOVABLE(value_or_error)
    };
}}}

#endif
