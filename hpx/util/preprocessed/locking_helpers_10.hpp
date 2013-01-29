// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


namespace boost { namespace detail
{
    template <typename MutexType0 , typename MutexType1 , typename MutexType2 , typename MutexType3 , typename MutexType4 , typename MutexType5>
    inline unsigned lock_helper(MutexType0 & m0 , MutexType1 & m1 , MutexType2 & m2 , MutexType3 & m3 , MutexType4 & m4 , MutexType5 & m5)
    {
        boost::unique_lock<MutexType0> l0(m0);
        if (unsigned const failed_lock = try_lock_internal(m1 , m2 , m3 , m4 , m5))
        {
            return failed_lock;
        }
        l0.release();
        return 0;
    }
    template <typename MutexType0 , typename MutexType1 , typename MutexType2 , typename MutexType3 , typename MutexType4 , typename MutexType5>
    inline unsigned try_lock_internal(MutexType0 & m0 , MutexType1 & m1 , MutexType2 & m2 , MutexType3 & m3 , MutexType4 & m4 , MutexType5 & m5)
    {
        boost::unique_lock<MutexType0> l0(m0, boost::try_to_lock);
        if (!l0)
        {
            return 1;
        }
        if (unsigned const failed_lock = try_lock_internal(m1 , m2 , m3 , m4 , m5))
        {
            return failed_lock + 1;
        }
        l0.release();
        return 0;
    }
}}
namespace boost
{
    template <typename MutexType0 , typename MutexType1 , typename MutexType2 , typename MutexType3 , typename MutexType4 , typename MutexType5>
    inline void lock(MutexType0 & m0 , MutexType1 & m1 , MutexType2 & m2 , MutexType3 & m3 , MutexType4 & m4 , MutexType5 & m5)
    {
        unsigned lock_first = 0;
        for (;;)
        {
            switch (lock_first)
            {
                case 0: lock_first = detail::lock_helper( m0 , m1 , m2 , m3 , m4 , m5); if (!lock_first) return; break; case 1: lock_first = detail::lock_helper( m1 , m2 , m3 , m4 , m5 , m0); if (!lock_first) return; lock_first = (lock_first+ 1) % 6; break; case 2: lock_first = detail::lock_helper( m2 , m3 , m4 , m5 , m0 , m1); if (!lock_first) return; lock_first = (lock_first+ 2) % 6; break; case 3: lock_first = detail::lock_helper( m3 , m4 , m5 , m0 , m1 , m2); if (!lock_first) return; lock_first = (lock_first+ 3) % 6; break; case 4: lock_first = detail::lock_helper( m4 , m5 , m0 , m1 , m2 , m3); if (!lock_first) return; lock_first = (lock_first+ 4) % 6; break; case 5: lock_first = detail::lock_helper( m5 , m0 , m1 , m2 , m3 , m4); if (!lock_first) return; lock_first = (lock_first+ 5) % 6; break;
            }
        }
    }
}
namespace boost { namespace detail
{
    template <typename MutexType0 , typename MutexType1 , typename MutexType2 , typename MutexType3 , typename MutexType4 , typename MutexType5 , typename MutexType6>
    inline unsigned lock_helper(MutexType0 & m0 , MutexType1 & m1 , MutexType2 & m2 , MutexType3 & m3 , MutexType4 & m4 , MutexType5 & m5 , MutexType6 & m6)
    {
        boost::unique_lock<MutexType0> l0(m0);
        if (unsigned const failed_lock = try_lock_internal(m1 , m2 , m3 , m4 , m5 , m6))
        {
            return failed_lock;
        }
        l0.release();
        return 0;
    }
    template <typename MutexType0 , typename MutexType1 , typename MutexType2 , typename MutexType3 , typename MutexType4 , typename MutexType5 , typename MutexType6>
    inline unsigned try_lock_internal(MutexType0 & m0 , MutexType1 & m1 , MutexType2 & m2 , MutexType3 & m3 , MutexType4 & m4 , MutexType5 & m5 , MutexType6 & m6)
    {
        boost::unique_lock<MutexType0> l0(m0, boost::try_to_lock);
        if (!l0)
        {
            return 1;
        }
        if (unsigned const failed_lock = try_lock_internal(m1 , m2 , m3 , m4 , m5 , m6))
        {
            return failed_lock + 1;
        }
        l0.release();
        return 0;
    }
}}
namespace boost
{
    template <typename MutexType0 , typename MutexType1 , typename MutexType2 , typename MutexType3 , typename MutexType4 , typename MutexType5 , typename MutexType6>
    inline void lock(MutexType0 & m0 , MutexType1 & m1 , MutexType2 & m2 , MutexType3 & m3 , MutexType4 & m4 , MutexType5 & m5 , MutexType6 & m6)
    {
        unsigned lock_first = 0;
        for (;;)
        {
            switch (lock_first)
            {
                case 0: lock_first = detail::lock_helper( m0 , m1 , m2 , m3 , m4 , m5 , m6); if (!lock_first) return; break; case 1: lock_first = detail::lock_helper( m1 , m2 , m3 , m4 , m5 , m6 , m0); if (!lock_first) return; lock_first = (lock_first+ 1) % 7; break; case 2: lock_first = detail::lock_helper( m2 , m3 , m4 , m5 , m6 , m0 , m1); if (!lock_first) return; lock_first = (lock_first+ 2) % 7; break; case 3: lock_first = detail::lock_helper( m3 , m4 , m5 , m6 , m0 , m1 , m2); if (!lock_first) return; lock_first = (lock_first+ 3) % 7; break; case 4: lock_first = detail::lock_helper( m4 , m5 , m6 , m0 , m1 , m2 , m3); if (!lock_first) return; lock_first = (lock_first+ 4) % 7; break; case 5: lock_first = detail::lock_helper( m5 , m6 , m0 , m1 , m2 , m3 , m4); if (!lock_first) return; lock_first = (lock_first+ 5) % 7; break; case 6: lock_first = detail::lock_helper( m6 , m0 , m1 , m2 , m3 , m4 , m5); if (!lock_first) return; lock_first = (lock_first+ 6) % 7; break;
            }
        }
    }
}
namespace boost { namespace detail
{
    template <typename MutexType0 , typename MutexType1 , typename MutexType2 , typename MutexType3 , typename MutexType4 , typename MutexType5 , typename MutexType6 , typename MutexType7>
    inline unsigned lock_helper(MutexType0 & m0 , MutexType1 & m1 , MutexType2 & m2 , MutexType3 & m3 , MutexType4 & m4 , MutexType5 & m5 , MutexType6 & m6 , MutexType7 & m7)
    {
        boost::unique_lock<MutexType0> l0(m0);
        if (unsigned const failed_lock = try_lock_internal(m1 , m2 , m3 , m4 , m5 , m6 , m7))
        {
            return failed_lock;
        }
        l0.release();
        return 0;
    }
    template <typename MutexType0 , typename MutexType1 , typename MutexType2 , typename MutexType3 , typename MutexType4 , typename MutexType5 , typename MutexType6 , typename MutexType7>
    inline unsigned try_lock_internal(MutexType0 & m0 , MutexType1 & m1 , MutexType2 & m2 , MutexType3 & m3 , MutexType4 & m4 , MutexType5 & m5 , MutexType6 & m6 , MutexType7 & m7)
    {
        boost::unique_lock<MutexType0> l0(m0, boost::try_to_lock);
        if (!l0)
        {
            return 1;
        }
        if (unsigned const failed_lock = try_lock_internal(m1 , m2 , m3 , m4 , m5 , m6 , m7))
        {
            return failed_lock + 1;
        }
        l0.release();
        return 0;
    }
}}
namespace boost
{
    template <typename MutexType0 , typename MutexType1 , typename MutexType2 , typename MutexType3 , typename MutexType4 , typename MutexType5 , typename MutexType6 , typename MutexType7>
    inline void lock(MutexType0 & m0 , MutexType1 & m1 , MutexType2 & m2 , MutexType3 & m3 , MutexType4 & m4 , MutexType5 & m5 , MutexType6 & m6 , MutexType7 & m7)
    {
        unsigned lock_first = 0;
        for (;;)
        {
            switch (lock_first)
            {
                case 0: lock_first = detail::lock_helper( m0 , m1 , m2 , m3 , m4 , m5 , m6 , m7); if (!lock_first) return; break; case 1: lock_first = detail::lock_helper( m1 , m2 , m3 , m4 , m5 , m6 , m7 , m0); if (!lock_first) return; lock_first = (lock_first+ 1) % 8; break; case 2: lock_first = detail::lock_helper( m2 , m3 , m4 , m5 , m6 , m7 , m0 , m1); if (!lock_first) return; lock_first = (lock_first+ 2) % 8; break; case 3: lock_first = detail::lock_helper( m3 , m4 , m5 , m6 , m7 , m0 , m1 , m2); if (!lock_first) return; lock_first = (lock_first+ 3) % 8; break; case 4: lock_first = detail::lock_helper( m4 , m5 , m6 , m7 , m0 , m1 , m2 , m3); if (!lock_first) return; lock_first = (lock_first+ 4) % 8; break; case 5: lock_first = detail::lock_helper( m5 , m6 , m7 , m0 , m1 , m2 , m3 , m4); if (!lock_first) return; lock_first = (lock_first+ 5) % 8; break; case 6: lock_first = detail::lock_helper( m6 , m7 , m0 , m1 , m2 , m3 , m4 , m5); if (!lock_first) return; lock_first = (lock_first+ 6) % 8; break; case 7: lock_first = detail::lock_helper( m7 , m0 , m1 , m2 , m3 , m4 , m5 , m6); if (!lock_first) return; lock_first = (lock_first+ 7) % 8; break;
            }
        }
    }
}
namespace boost { namespace detail
{
    template <typename MutexType0 , typename MutexType1 , typename MutexType2 , typename MutexType3 , typename MutexType4 , typename MutexType5 , typename MutexType6 , typename MutexType7 , typename MutexType8>
    inline unsigned lock_helper(MutexType0 & m0 , MutexType1 & m1 , MutexType2 & m2 , MutexType3 & m3 , MutexType4 & m4 , MutexType5 & m5 , MutexType6 & m6 , MutexType7 & m7 , MutexType8 & m8)
    {
        boost::unique_lock<MutexType0> l0(m0);
        if (unsigned const failed_lock = try_lock_internal(m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8))
        {
            return failed_lock;
        }
        l0.release();
        return 0;
    }
    template <typename MutexType0 , typename MutexType1 , typename MutexType2 , typename MutexType3 , typename MutexType4 , typename MutexType5 , typename MutexType6 , typename MutexType7 , typename MutexType8>
    inline unsigned try_lock_internal(MutexType0 & m0 , MutexType1 & m1 , MutexType2 & m2 , MutexType3 & m3 , MutexType4 & m4 , MutexType5 & m5 , MutexType6 & m6 , MutexType7 & m7 , MutexType8 & m8)
    {
        boost::unique_lock<MutexType0> l0(m0, boost::try_to_lock);
        if (!l0)
        {
            return 1;
        }
        if (unsigned const failed_lock = try_lock_internal(m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8))
        {
            return failed_lock + 1;
        }
        l0.release();
        return 0;
    }
}}
namespace boost
{
    template <typename MutexType0 , typename MutexType1 , typename MutexType2 , typename MutexType3 , typename MutexType4 , typename MutexType5 , typename MutexType6 , typename MutexType7 , typename MutexType8>
    inline void lock(MutexType0 & m0 , MutexType1 & m1 , MutexType2 & m2 , MutexType3 & m3 , MutexType4 & m4 , MutexType5 & m5 , MutexType6 & m6 , MutexType7 & m7 , MutexType8 & m8)
    {
        unsigned lock_first = 0;
        for (;;)
        {
            switch (lock_first)
            {
                case 0: lock_first = detail::lock_helper( m0 , m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8); if (!lock_first) return; break; case 1: lock_first = detail::lock_helper( m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8 , m0); if (!lock_first) return; lock_first = (lock_first+ 1) % 9; break; case 2: lock_first = detail::lock_helper( m2 , m3 , m4 , m5 , m6 , m7 , m8 , m0 , m1); if (!lock_first) return; lock_first = (lock_first+ 2) % 9; break; case 3: lock_first = detail::lock_helper( m3 , m4 , m5 , m6 , m7 , m8 , m0 , m1 , m2); if (!lock_first) return; lock_first = (lock_first+ 3) % 9; break; case 4: lock_first = detail::lock_helper( m4 , m5 , m6 , m7 , m8 , m0 , m1 , m2 , m3); if (!lock_first) return; lock_first = (lock_first+ 4) % 9; break; case 5: lock_first = detail::lock_helper( m5 , m6 , m7 , m8 , m0 , m1 , m2 , m3 , m4); if (!lock_first) return; lock_first = (lock_first+ 5) % 9; break; case 6: lock_first = detail::lock_helper( m6 , m7 , m8 , m0 , m1 , m2 , m3 , m4 , m5); if (!lock_first) return; lock_first = (lock_first+ 6) % 9; break; case 7: lock_first = detail::lock_helper( m7 , m8 , m0 , m1 , m2 , m3 , m4 , m5 , m6); if (!lock_first) return; lock_first = (lock_first+ 7) % 9; break; case 8: lock_first = detail::lock_helper( m8 , m0 , m1 , m2 , m3 , m4 , m5 , m6 , m7); if (!lock_first) return; lock_first = (lock_first+ 8) % 9; break;
            }
        }
    }
}
namespace boost { namespace detail
{
    template <typename MutexType0 , typename MutexType1 , typename MutexType2 , typename MutexType3 , typename MutexType4 , typename MutexType5 , typename MutexType6 , typename MutexType7 , typename MutexType8 , typename MutexType9>
    inline unsigned lock_helper(MutexType0 & m0 , MutexType1 & m1 , MutexType2 & m2 , MutexType3 & m3 , MutexType4 & m4 , MutexType5 & m5 , MutexType6 & m6 , MutexType7 & m7 , MutexType8 & m8 , MutexType9 & m9)
    {
        boost::unique_lock<MutexType0> l0(m0);
        if (unsigned const failed_lock = try_lock_internal(m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8 , m9))
        {
            return failed_lock;
        }
        l0.release();
        return 0;
    }
    template <typename MutexType0 , typename MutexType1 , typename MutexType2 , typename MutexType3 , typename MutexType4 , typename MutexType5 , typename MutexType6 , typename MutexType7 , typename MutexType8 , typename MutexType9>
    inline unsigned try_lock_internal(MutexType0 & m0 , MutexType1 & m1 , MutexType2 & m2 , MutexType3 & m3 , MutexType4 & m4 , MutexType5 & m5 , MutexType6 & m6 , MutexType7 & m7 , MutexType8 & m8 , MutexType9 & m9)
    {
        boost::unique_lock<MutexType0> l0(m0, boost::try_to_lock);
        if (!l0)
        {
            return 1;
        }
        if (unsigned const failed_lock = try_lock_internal(m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8 , m9))
        {
            return failed_lock + 1;
        }
        l0.release();
        return 0;
    }
}}
namespace boost
{
    template <typename MutexType0 , typename MutexType1 , typename MutexType2 , typename MutexType3 , typename MutexType4 , typename MutexType5 , typename MutexType6 , typename MutexType7 , typename MutexType8 , typename MutexType9>
    inline void lock(MutexType0 & m0 , MutexType1 & m1 , MutexType2 & m2 , MutexType3 & m3 , MutexType4 & m4 , MutexType5 & m5 , MutexType6 & m6 , MutexType7 & m7 , MutexType8 & m8 , MutexType9 & m9)
    {
        unsigned lock_first = 0;
        for (;;)
        {
            switch (lock_first)
            {
                case 0: lock_first = detail::lock_helper( m0 , m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8 , m9); if (!lock_first) return; break; case 1: lock_first = detail::lock_helper( m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8 , m9 , m0); if (!lock_first) return; lock_first = (lock_first+ 1) % 10; break; case 2: lock_first = detail::lock_helper( m2 , m3 , m4 , m5 , m6 , m7 , m8 , m9 , m0 , m1); if (!lock_first) return; lock_first = (lock_first+ 2) % 10; break; case 3: lock_first = detail::lock_helper( m3 , m4 , m5 , m6 , m7 , m8 , m9 , m0 , m1 , m2); if (!lock_first) return; lock_first = (lock_first+ 3) % 10; break; case 4: lock_first = detail::lock_helper( m4 , m5 , m6 , m7 , m8 , m9 , m0 , m1 , m2 , m3); if (!lock_first) return; lock_first = (lock_first+ 4) % 10; break; case 5: lock_first = detail::lock_helper( m5 , m6 , m7 , m8 , m9 , m0 , m1 , m2 , m3 , m4); if (!lock_first) return; lock_first = (lock_first+ 5) % 10; break; case 6: lock_first = detail::lock_helper( m6 , m7 , m8 , m9 , m0 , m1 , m2 , m3 , m4 , m5); if (!lock_first) return; lock_first = (lock_first+ 6) % 10; break; case 7: lock_first = detail::lock_helper( m7 , m8 , m9 , m0 , m1 , m2 , m3 , m4 , m5 , m6); if (!lock_first) return; lock_first = (lock_first+ 7) % 10; break; case 8: lock_first = detail::lock_helper( m8 , m9 , m0 , m1 , m2 , m3 , m4 , m5 , m6 , m7); if (!lock_first) return; lock_first = (lock_first+ 8) % 10; break; case 9: lock_first = detail::lock_helper( m9 , m0 , m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8); if (!lock_first) return; lock_first = (lock_first+ 9) % 10; break;
            }
        }
    }
}
