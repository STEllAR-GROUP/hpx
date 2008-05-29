//Copyright (c) 2006-2008 Emil Dotchevski and Reverge Studios, Inc.

//Distributed under the Boost Software License, Version 1.0. (See accompanying
//file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef UUID_FA5836A2CADA11DC8CD47C8555D89593
#define UUID_FA5836A2CADA11DC8CD47C8555D89593

#include <boost/exception/enable_current_exception.hpp>
#include <boost/exception/exception.hpp>
#include <boost/exception/detail/cloning_base.hpp>
#include <stdexcept>
#include <new>

namespace
boost
    {
    class
    unknown_exception:
        public exception,
        public std::exception
        {
        public:

        unknown_exception()
            {
            }

        explicit
        unknown_exception( boost::exception const & e ):
            boost::exception(e)
            {
            }

		~unknown_exception() throw()
			{
			}
        };

    typedef intrusive_ptr<exception_detail::clone_base const> exception_ptr;

    namespace
    exception_detail
        {
        template <class T>
        class
        current_exception_std_exception_wrapper:
            public T,
            public boost::exception
            {
            public:

            explicit
            current_exception_std_exception_wrapper( T const & e1 ):
                T(e1)
                {
                }

            current_exception_std_exception_wrapper( T const & e1, boost::exception const & e2 ):
                T(e1),
                boost::exception(e2)
                {
                }

			~current_exception_std_exception_wrapper() throw()
				{
				}
            };

        template <class T>
        exception_ptr
        current_exception_std_exception( T const & e1 )
            {
            if( boost::exception const * e2 = dynamic_cast<boost::exception const *>(&e1) )
                return exception_ptr(exception_detail::make_clone(current_exception_std_exception_wrapper<T>(e1,*e2)));
            else
                return exception_ptr(exception_detail::make_clone(current_exception_std_exception_wrapper<T>(e1)));
            }

        inline
        exception_ptr
        current_exception_unknown_exception()
            {
            return exception_ptr(exception_detail::make_clone(unknown_exception()));
            }

        inline
        exception_ptr
        current_exception_unknown_std_exception( std::exception const & e )
            {
            if( boost::exception const * be = dynamic_cast<boost::exception const *>(&e) )
                return exception_ptr(exception_detail::make_clone(unknown_exception(*be)));
            else
                return current_exception_unknown_exception();
            }

        inline
        exception_ptr
        current_exception_unknown_boost_exception( boost::exception const & e )
            {
            return exception_ptr(exception_detail::make_clone(unknown_exception(e)));
            }
        }

    inline
    exception_ptr
    current_exception()
        {
        try
            {
            throw;
            }
        catch(
        exception_detail::cloning_base & e )
            {
            exception_detail::clone_base const * c = e.clone();
            BOOST_ASSERT(c!=0);
            return exception_ptr(c);
            }
        catch(
        std::invalid_argument & e )
            {
            return exception_detail::current_exception_std_exception(e);
            }
        catch(
        std::out_of_range & e )
            {
            return exception_detail::current_exception_std_exception(e);
            }
        catch(
        std::logic_error & e )
            {
            return exception_detail::current_exception_std_exception(e);
            }
        catch(
        std::bad_alloc & e )
            {
            return exception_detail::current_exception_std_exception(e);
            }
        catch(
        std::bad_cast & e )
            {
            return exception_detail::current_exception_std_exception(e);
            }
        catch(
        std::bad_typeid & e )
            {
            return exception_detail::current_exception_std_exception(e);
            }
        catch(
        std::bad_exception & e )
            {
            return exception_detail::current_exception_std_exception(e);
            }
        catch(
        std::exception & e )
            {
            return exception_detail::current_exception_unknown_std_exception(e);
            }
        catch(
        boost::exception & e )
            {
            return exception_detail::current_exception_unknown_boost_exception(e);
            }
        catch(
        ... )
            {
            return exception_detail::current_exception_unknown_exception();
            }
        }

    template <class T>
    exception_ptr
    copy_exception( T const & e )
        {
        try
            {
            throw enable_current_exception(e);
            }
        catch( ... )
            {
            return current_exception();
            }
        }

    inline
    void
    rethrow_exception( exception_ptr const & p )
        {
        p->rethrow();
        }
    }

#endif
