//  boost process_times.hpp  ------------------------------------------------------//

//  Copyright Beman Dawes 1994, 2007, 2008

//  Distributed under the Boost Software License, Version 1.0.
//  See http://www.boost.org/LICENSE_1_0.txt

//  See http://www.boost.org/libs/system for documentation.

#ifndef BOOST_PROCESS_TIMES_HPP                  
#define BOOST_PROCESS_TIMES_HPP

#include <boost/chrono/chrono.hpp>
#include <boost/system/error_code.hpp>
#include <boost/cstdint.hpp>
#include <string>
#include <ostream>

#include <boost/config/abi_prefix.hpp> // must be the last #include

namespace boost
{
  namespace chrono
  {

    struct process_times;

//---------------------------------------------------------------------------------//
//                                process_clock                                    //
//---------------------------------------------------------------------------------//

    class BOOST_CHRONO_DECL process_clock
    {
    public:
        typedef nanoseconds                          duration;
        typedef duration::rep                        rep;
        typedef duration::period                     period;
        typedef chrono::time_point<process_clock>    time_point;
        static const bool is_monotonic =             true;

        static void now( process_times & times,
                         system::error_code & ec = system::throws );
    };

//---------------------------------------------------------------------------------//
//                                process_times                                    //
//---------------------------------------------------------------------------------//

    struct process_times
    {
      process_clock::duration                        real;       // real time
      process_clock::duration                        user;       // user cpu time
      process_clock::duration                        system;     // system cpu time
    };

//---------------------------------------------------------------------------------//
//                                 process_timer                                   //
//---------------------------------------------------------------------------------//

    class BOOST_CHRONO_DECL process_timer
    // BOOST_CHRONO_DECL is required to quiet compiler warnings even though
    // process_timer has no dynamically linked members, because process_timer is
    // used as a base class for run_timer, which does have dynamically linked members.
    {
    public:

      typedef process_clock                          clock;
      typedef process_clock::duration                duration;
      typedef process_clock::time_point              time_point;

      explicit process_timer( system::error_code & ec = system::throws )
      {
        start(ec);
      }

     ~process_timer() {}  // never throws

      void  start( system::error_code & ec = system::throws )
      { 
        process_clock::now( m_start, ec );
      }

      void  elapsed( process_times & times, system::error_code & ec = system::throws )
      {
        process_times end;
        process_clock::now( end, ec );
        times.real  = end.real - m_start.real;
        times.user       = end.user - m_start.user;
        times.system     = end.system - m_start.system;
      }

    protected:
      process_times   m_start;
    };

//---------------------------------------------------------------------------------//
//                                   run_timer                                     //
//---------------------------------------------------------------------------------//

    class BOOST_CHRONO_DECL run_timer : public process_timer
    {
    public:

      // each constructor form has two overloads to avoid a visible default to
      // std::cout, which in turn would require including <iostream>, with its
      // high associated cost, even when the standard streams are not used.

      explicit run_timer( system::error_code & ec = system::throws )
        : m_places(m_default_places), m_os(m_cout()) { start(ec); }
      explicit run_timer( std::ostream & os,
        system::error_code & ec = system::throws )
        : m_places(m_default_places), m_os(os) { start(ec); }

      explicit run_timer( const std::string & format,
        system::error_code & ec = system::throws )
        : m_places(m_default_places), m_os(m_cout()), m_format(format) { start(ec); }
      explicit run_timer( std::ostream & os, const std::string & format,
        system::error_code & ec = system::throws )
        : m_places(m_default_places), m_os(os), m_format(format) { start(ec); }

      explicit run_timer( const std::string & format, int places,
        system::error_code & ec = system::throws )
        : m_places(places), m_os(m_cout()), m_format(format) { start(ec); }
      explicit run_timer( std::ostream & os, const std::string & format,
        int places, system::error_code & ec = system::throws )
        : m_places(places), m_os(os), m_format(format) { start(ec); }

      explicit run_timer( int places,
        system::error_code & ec = system::throws )
        : m_places(places), m_os(m_cout()) { start(ec); }
      explicit run_timer( std::ostream & os, int places,
        system::error_code & ec = system::throws )
        : m_places(places), m_os(os) { start(ec); }

      explicit run_timer( int places, const std::string & format,
        system::error_code & ec = system::throws )
        : m_places(places), m_os(m_cout()), m_format(format) { start(ec); }
      explicit run_timer( std::ostream & os, int places, const std::string & format,
        system::error_code & ec = system::throws )
        : m_places(places), m_os(os), m_format(format) { start(ec); }

     ~run_timer()  // never throws
      { 
        system::error_code ec;
        if ( !reported() ) report( ec );
      }

      void  start( system::error_code & ec = system::throws )
      {
        m_reported = false;
        process_timer::start( ec );
      }

      void  report( system::error_code & ec = system::throws );

      void  test_report( duration real_, duration user_, duration system_ );

      bool  reported() const { return m_reported; }

      static int default_places() { return m_default_places; }

    private:
      int             m_places;
      std::ostream &  m_os;
      std::string     m_format;
      bool            m_reported;

      static std::ostream &  m_cout();
      static const int m_default_places = 3;
    };

  } // namespace chrono
} // namespace boost

#include <boost/config/abi_suffix.hpp> // pops abi_prefix.hpp pragmas

#endif  // BOOST_PROCESS_TIMES_HPP
