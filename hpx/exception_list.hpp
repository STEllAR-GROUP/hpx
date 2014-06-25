//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_EXCEPTION_LIST_OCT_06_2008_0942AM)
#define HPX_EXCEPTION_LIST_OCT_06_2008_0942AM

#include <list>
#include <string>
#include <hpx/exception.hpp>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    /// 1. The class exception_list is a container of exception_ptr objects
    ///    parallel algorithms may use to communicate uncaught exceptions
    ///    encountered during parallel execution to the caller of the algorithm.
    /// 2. The type exception_list::const_iterator shall fulfill the
    ///    requirements of ForwardIterator.
    class HPX_EXCEPTION_EXPORT exception_list : public hpx::exception
    {
    private:
        typedef std::list<boost::exception_ptr> exception_list_type;
        exception_list_type exceptions_;

    public:
        typedef boost::exception_ptr value_type;
        typedef value_type const& reference;
        typedef value_type const& const_reference;
        typedef exception_list_type::size_type size_type;
        typedef exception_list_type::const_iterator iterator;
        typedef exception_list_type::const_iterator const_iterator;
        typedef std::iterator_traits<const_iterator>::difference_type
            difference_type;

        /// \throws nothing
        ~exception_list() throw() {}
        
        exception_list();
        explicit exception_list(boost::exception_ptr const& e);
        explicit exception_list(exception_list_type && l);

        ///
        void add(boost::exception_ptr const& e)
        {
            if (exceptions_.empty())
            {
                // set the error code for our base class
                static_cast<hpx::exception&>(*this) =
                    hpx::exception(hpx::get_error(e));
            }
            exceptions_.push_back(e);
        }

        std::size_t size() const BOOST_NOEXCEPT
        {
            return exceptions_.size();
        }

        exception_list_type::const_iterator begin() const BOOST_NOEXCEPT
        {
            return exceptions_.begin();
        }

        exception_list_type::const_iterator end() const BOOST_NOEXCEPT
        {
            return exceptions_.end();
        }

        ///
        boost::system::error_code get_error() const;

        ///
        std::string get_message() const;

        ///
        std::size_t get_error_count() const;
    };

}

#include <hpx/config/warnings_suffix.hpp>

#endif


