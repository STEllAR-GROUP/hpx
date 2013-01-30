// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


    template <typename Arg0 , typename Arg1>
    transfer_action(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1)
        : arguments_(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 )),
          priority_(
                detail::thread_priority<
                    static_cast<threads::thread_priority>(priority_value)
                >::call(priority_value)),
          stacksize_(
              detail::thread_stacksize<
                  static_cast<threads::thread_stacksize>(stacksize_value)
              >::call(threads::thread_stacksize_default))
    {}
    template <typename Arg0 , typename Arg1>
    transfer_action(threads::thread_priority priority,
              BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1)
        : arguments_(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 )),
          priority_(
                detail::thread_priority<
                    static_cast<threads::thread_priority>(priority_value)
                >::call(priority)),
          stacksize_(
              detail::thread_stacksize<
                  static_cast<threads::thread_stacksize>(stacksize_value)
              >::call(threads::thread_stacksize_default))
    {}
    template <typename Arg0 , typename Arg1 , typename Arg2>
    transfer_action(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2)
        : arguments_(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 )),
          priority_(
                detail::thread_priority<
                    static_cast<threads::thread_priority>(priority_value)
                >::call(priority_value)),
          stacksize_(
              detail::thread_stacksize<
                  static_cast<threads::thread_stacksize>(stacksize_value)
              >::call(threads::thread_stacksize_default))
    {}
    template <typename Arg0 , typename Arg1 , typename Arg2>
    transfer_action(threads::thread_priority priority,
              BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2)
        : arguments_(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 )),
          priority_(
                detail::thread_priority<
                    static_cast<threads::thread_priority>(priority_value)
                >::call(priority)),
          stacksize_(
              detail::thread_stacksize<
                  static_cast<threads::thread_stacksize>(stacksize_value)
              >::call(threads::thread_stacksize_default))
    {}
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    transfer_action(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3)
        : arguments_(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 )),
          priority_(
                detail::thread_priority<
                    static_cast<threads::thread_priority>(priority_value)
                >::call(priority_value)),
          stacksize_(
              detail::thread_stacksize<
                  static_cast<threads::thread_stacksize>(stacksize_value)
              >::call(threads::thread_stacksize_default))
    {}
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    transfer_action(threads::thread_priority priority,
              BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3)
        : arguments_(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 )),
          priority_(
                detail::thread_priority<
                    static_cast<threads::thread_priority>(priority_value)
                >::call(priority)),
          stacksize_(
              detail::thread_stacksize<
                  static_cast<threads::thread_stacksize>(stacksize_value)
              >::call(threads::thread_stacksize_default))
    {}
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    transfer_action(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4)
        : arguments_(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 )),
          priority_(
                detail::thread_priority<
                    static_cast<threads::thread_priority>(priority_value)
                >::call(priority_value)),
          stacksize_(
              detail::thread_stacksize<
                  static_cast<threads::thread_stacksize>(stacksize_value)
              >::call(threads::thread_stacksize_default))
    {}
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    transfer_action(threads::thread_priority priority,
              BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4)
        : arguments_(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 )),
          priority_(
                detail::thread_priority<
                    static_cast<threads::thread_priority>(priority_value)
                >::call(priority)),
          stacksize_(
              detail::thread_stacksize<
                  static_cast<threads::thread_stacksize>(stacksize_value)
              >::call(threads::thread_stacksize_default))
    {}
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    transfer_action(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5)
        : arguments_(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 )),
          priority_(
                detail::thread_priority<
                    static_cast<threads::thread_priority>(priority_value)
                >::call(priority_value)),
          stacksize_(
              detail::thread_stacksize<
                  static_cast<threads::thread_stacksize>(stacksize_value)
              >::call(threads::thread_stacksize_default))
    {}
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    transfer_action(threads::thread_priority priority,
              BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5)
        : arguments_(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 )),
          priority_(
                detail::thread_priority<
                    static_cast<threads::thread_priority>(priority_value)
                >::call(priority)),
          stacksize_(
              detail::thread_stacksize<
                  static_cast<threads::thread_stacksize>(stacksize_value)
              >::call(threads::thread_stacksize_default))
    {}
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    transfer_action(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6)
        : arguments_(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 )),
          priority_(
                detail::thread_priority<
                    static_cast<threads::thread_priority>(priority_value)
                >::call(priority_value)),
          stacksize_(
              detail::thread_stacksize<
                  static_cast<threads::thread_stacksize>(stacksize_value)
              >::call(threads::thread_stacksize_default))
    {}
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    transfer_action(threads::thread_priority priority,
              BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6)
        : arguments_(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 )),
          priority_(
                detail::thread_priority<
                    static_cast<threads::thread_priority>(priority_value)
                >::call(priority)),
          stacksize_(
              detail::thread_stacksize<
                  static_cast<threads::thread_stacksize>(stacksize_value)
              >::call(threads::thread_stacksize_default))
    {}
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    transfer_action(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7)
        : arguments_(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 )),
          priority_(
                detail::thread_priority<
                    static_cast<threads::thread_priority>(priority_value)
                >::call(priority_value)),
          stacksize_(
              detail::thread_stacksize<
                  static_cast<threads::thread_stacksize>(stacksize_value)
              >::call(threads::thread_stacksize_default))
    {}
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    transfer_action(threads::thread_priority priority,
              BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7)
        : arguments_(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 )),
          priority_(
                detail::thread_priority<
                    static_cast<threads::thread_priority>(priority_value)
                >::call(priority)),
          stacksize_(
              detail::thread_stacksize<
                  static_cast<threads::thread_stacksize>(stacksize_value)
              >::call(threads::thread_stacksize_default))
    {}
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    transfer_action(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8)
        : arguments_(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 )),
          priority_(
                detail::thread_priority<
                    static_cast<threads::thread_priority>(priority_value)
                >::call(priority_value)),
          stacksize_(
              detail::thread_stacksize<
                  static_cast<threads::thread_stacksize>(stacksize_value)
              >::call(threads::thread_stacksize_default))
    {}
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    transfer_action(threads::thread_priority priority,
              BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8)
        : arguments_(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 )),
          priority_(
                detail::thread_priority<
                    static_cast<threads::thread_priority>(priority_value)
                >::call(priority)),
          stacksize_(
              detail::thread_stacksize<
                  static_cast<threads::thread_stacksize>(stacksize_value)
              >::call(threads::thread_stacksize_default))
    {}
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    transfer_action(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9)
        : arguments_(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 )),
          priority_(
                detail::thread_priority<
                    static_cast<threads::thread_priority>(priority_value)
                >::call(priority_value)),
          stacksize_(
              detail::thread_stacksize<
                  static_cast<threads::thread_stacksize>(stacksize_value)
              >::call(threads::thread_stacksize_default))
    {}
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    transfer_action(threads::thread_priority priority,
              BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9)
        : arguments_(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 )),
          priority_(
                detail::thread_priority<
                    static_cast<threads::thread_priority>(priority_value)
                >::call(priority)),
          stacksize_(
              detail::thread_stacksize<
                  static_cast<threads::thread_stacksize>(stacksize_value)
              >::call(threads::thread_stacksize_default))
    {}
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10>
    transfer_action(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10)
        : arguments_(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 )),
          priority_(
                detail::thread_priority<
                    static_cast<threads::thread_priority>(priority_value)
                >::call(priority_value)),
          stacksize_(
              detail::thread_stacksize<
                  static_cast<threads::thread_stacksize>(stacksize_value)
              >::call(threads::thread_stacksize_default))
    {}
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10>
    transfer_action(threads::thread_priority priority,
              BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10)
        : arguments_(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 )),
          priority_(
                detail::thread_priority<
                    static_cast<threads::thread_priority>(priority_value)
                >::call(priority)),
          stacksize_(
              detail::thread_stacksize<
                  static_cast<threads::thread_stacksize>(stacksize_value)
              >::call(threads::thread_stacksize_default))
    {}
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11>
    transfer_action(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11)
        : arguments_(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 )),
          priority_(
                detail::thread_priority<
                    static_cast<threads::thread_priority>(priority_value)
                >::call(priority_value)),
          stacksize_(
              detail::thread_stacksize<
                  static_cast<threads::thread_stacksize>(stacksize_value)
              >::call(threads::thread_stacksize_default))
    {}
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11>
    transfer_action(threads::thread_priority priority,
              BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11)
        : arguments_(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 )),
          priority_(
                detail::thread_priority<
                    static_cast<threads::thread_priority>(priority_value)
                >::call(priority)),
          stacksize_(
              detail::thread_stacksize<
                  static_cast<threads::thread_stacksize>(stacksize_value)
              >::call(threads::thread_stacksize_default))
    {}
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12>
    transfer_action(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12)
        : arguments_(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 )),
          priority_(
                detail::thread_priority<
                    static_cast<threads::thread_priority>(priority_value)
                >::call(priority_value)),
          stacksize_(
              detail::thread_stacksize<
                  static_cast<threads::thread_stacksize>(stacksize_value)
              >::call(threads::thread_stacksize_default))
    {}
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12>
    transfer_action(threads::thread_priority priority,
              BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12)
        : arguments_(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 )),
          priority_(
                detail::thread_priority<
                    static_cast<threads::thread_priority>(priority_value)
                >::call(priority)),
          stacksize_(
              detail::thread_stacksize<
                  static_cast<threads::thread_stacksize>(stacksize_value)
              >::call(threads::thread_stacksize_default))
    {}
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13>
    transfer_action(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13)
        : arguments_(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 )),
          priority_(
                detail::thread_priority<
                    static_cast<threads::thread_priority>(priority_value)
                >::call(priority_value)),
          stacksize_(
              detail::thread_stacksize<
                  static_cast<threads::thread_stacksize>(stacksize_value)
              >::call(threads::thread_stacksize_default))
    {}
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13>
    transfer_action(threads::thread_priority priority,
              BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13)
        : arguments_(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 )),
          priority_(
                detail::thread_priority<
                    static_cast<threads::thread_priority>(priority_value)
                >::call(priority)),
          stacksize_(
              detail::thread_stacksize<
                  static_cast<threads::thread_stacksize>(stacksize_value)
              >::call(threads::thread_stacksize_default))
    {}
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14>
    transfer_action(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14)
        : arguments_(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ) , boost::forward<Arg14>( arg14 )),
          priority_(
                detail::thread_priority<
                    static_cast<threads::thread_priority>(priority_value)
                >::call(priority_value)),
          stacksize_(
              detail::thread_stacksize<
                  static_cast<threads::thread_stacksize>(stacksize_value)
              >::call(threads::thread_stacksize_default))
    {}
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14>
    transfer_action(threads::thread_priority priority,
              BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14)
        : arguments_(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ) , boost::forward<Arg14>( arg14 )),
          priority_(
                detail::thread_priority<
                    static_cast<threads::thread_priority>(priority_value)
                >::call(priority)),
          stacksize_(
              detail::thread_stacksize<
                  static_cast<threads::thread_stacksize>(stacksize_value)
              >::call(threads::thread_stacksize_default))
    {}
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15>
    transfer_action(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15)
        : arguments_(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ) , boost::forward<Arg14>( arg14 ) , boost::forward<Arg15>( arg15 )),
          priority_(
                detail::thread_priority<
                    static_cast<threads::thread_priority>(priority_value)
                >::call(priority_value)),
          stacksize_(
              detail::thread_stacksize<
                  static_cast<threads::thread_stacksize>(stacksize_value)
              >::call(threads::thread_stacksize_default))
    {}
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15>
    transfer_action(threads::thread_priority priority,
              BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15)
        : arguments_(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ) , boost::forward<Arg14>( arg14 ) , boost::forward<Arg15>( arg15 )),
          priority_(
                detail::thread_priority<
                    static_cast<threads::thread_priority>(priority_value)
                >::call(priority)),
          stacksize_(
              detail::thread_stacksize<
                  static_cast<threads::thread_stacksize>(stacksize_value)
              >::call(threads::thread_stacksize_default))
    {}
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16>
    transfer_action(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15 , BOOST_FWD_REF(Arg16) arg16)
        : arguments_(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ) , boost::forward<Arg14>( arg14 ) , boost::forward<Arg15>( arg15 ) , boost::forward<Arg16>( arg16 )),
          priority_(
                detail::thread_priority<
                    static_cast<threads::thread_priority>(priority_value)
                >::call(priority_value)),
          stacksize_(
              detail::thread_stacksize<
                  static_cast<threads::thread_stacksize>(stacksize_value)
              >::call(threads::thread_stacksize_default))
    {}
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16>
    transfer_action(threads::thread_priority priority,
              BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15 , BOOST_FWD_REF(Arg16) arg16)
        : arguments_(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ) , boost::forward<Arg14>( arg14 ) , boost::forward<Arg15>( arg15 ) , boost::forward<Arg16>( arg16 )),
          priority_(
                detail::thread_priority<
                    static_cast<threads::thread_priority>(priority_value)
                >::call(priority)),
          stacksize_(
              detail::thread_stacksize<
                  static_cast<threads::thread_stacksize>(stacksize_value)
              >::call(threads::thread_stacksize_default))
    {}
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17>
    transfer_action(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15 , BOOST_FWD_REF(Arg16) arg16 , BOOST_FWD_REF(Arg17) arg17)
        : arguments_(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ) , boost::forward<Arg14>( arg14 ) , boost::forward<Arg15>( arg15 ) , boost::forward<Arg16>( arg16 ) , boost::forward<Arg17>( arg17 )),
          priority_(
                detail::thread_priority<
                    static_cast<threads::thread_priority>(priority_value)
                >::call(priority_value)),
          stacksize_(
              detail::thread_stacksize<
                  static_cast<threads::thread_stacksize>(stacksize_value)
              >::call(threads::thread_stacksize_default))
    {}
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17>
    transfer_action(threads::thread_priority priority,
              BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15 , BOOST_FWD_REF(Arg16) arg16 , BOOST_FWD_REF(Arg17) arg17)
        : arguments_(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ) , boost::forward<Arg14>( arg14 ) , boost::forward<Arg15>( arg15 ) , boost::forward<Arg16>( arg16 ) , boost::forward<Arg17>( arg17 )),
          priority_(
                detail::thread_priority<
                    static_cast<threads::thread_priority>(priority_value)
                >::call(priority)),
          stacksize_(
              detail::thread_stacksize<
                  static_cast<threads::thread_stacksize>(stacksize_value)
              >::call(threads::thread_stacksize_default))
    {}
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18>
    transfer_action(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15 , BOOST_FWD_REF(Arg16) arg16 , BOOST_FWD_REF(Arg17) arg17 , BOOST_FWD_REF(Arg18) arg18)
        : arguments_(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ) , boost::forward<Arg14>( arg14 ) , boost::forward<Arg15>( arg15 ) , boost::forward<Arg16>( arg16 ) , boost::forward<Arg17>( arg17 ) , boost::forward<Arg18>( arg18 )),
          priority_(
                detail::thread_priority<
                    static_cast<threads::thread_priority>(priority_value)
                >::call(priority_value)),
          stacksize_(
              detail::thread_stacksize<
                  static_cast<threads::thread_stacksize>(stacksize_value)
              >::call(threads::thread_stacksize_default))
    {}
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18>
    transfer_action(threads::thread_priority priority,
              BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15 , BOOST_FWD_REF(Arg16) arg16 , BOOST_FWD_REF(Arg17) arg17 , BOOST_FWD_REF(Arg18) arg18)
        : arguments_(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ) , boost::forward<Arg14>( arg14 ) , boost::forward<Arg15>( arg15 ) , boost::forward<Arg16>( arg16 ) , boost::forward<Arg17>( arg17 ) , boost::forward<Arg18>( arg18 )),
          priority_(
                detail::thread_priority<
                    static_cast<threads::thread_priority>(priority_value)
                >::call(priority)),
          stacksize_(
              detail::thread_stacksize<
                  static_cast<threads::thread_stacksize>(stacksize_value)
              >::call(threads::thread_stacksize_default))
    {}
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18 , typename Arg19>
    transfer_action(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15 , BOOST_FWD_REF(Arg16) arg16 , BOOST_FWD_REF(Arg17) arg17 , BOOST_FWD_REF(Arg18) arg18 , BOOST_FWD_REF(Arg19) arg19)
        : arguments_(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ) , boost::forward<Arg14>( arg14 ) , boost::forward<Arg15>( arg15 ) , boost::forward<Arg16>( arg16 ) , boost::forward<Arg17>( arg17 ) , boost::forward<Arg18>( arg18 ) , boost::forward<Arg19>( arg19 )),
          priority_(
                detail::thread_priority<
                    static_cast<threads::thread_priority>(priority_value)
                >::call(priority_value)),
          stacksize_(
              detail::thread_stacksize<
                  static_cast<threads::thread_stacksize>(stacksize_value)
              >::call(threads::thread_stacksize_default))
    {}
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18 , typename Arg19>
    transfer_action(threads::thread_priority priority,
              BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15 , BOOST_FWD_REF(Arg16) arg16 , BOOST_FWD_REF(Arg17) arg17 , BOOST_FWD_REF(Arg18) arg18 , BOOST_FWD_REF(Arg19) arg19)
        : arguments_(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ) , boost::forward<Arg14>( arg14 ) , boost::forward<Arg15>( arg15 ) , boost::forward<Arg16>( arg16 ) , boost::forward<Arg17>( arg17 ) , boost::forward<Arg18>( arg18 ) , boost::forward<Arg19>( arg19 )),
          priority_(
                detail::thread_priority<
                    static_cast<threads::thread_priority>(priority_value)
                >::call(priority)),
          stacksize_(
              detail::thread_stacksize<
                  static_cast<threads::thread_stacksize>(stacksize_value)
              >::call(threads::thread_stacksize_default))
    {}
