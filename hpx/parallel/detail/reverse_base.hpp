        template <typename ReverseIter>
        typename ReverseIter::iterator_type
        reverse_base(ReverseIter && it)
        {
            return it.base();
        }

        template <typename ReverseIter>
        hpx::future<typename ReverseIter::iterator_type>
        reverse_base(hpx::future<ReverseIter> && it)
        {
            return
                it.then(
                    [](hpx::future<ReverseIter> && it)
                    {
                        return it.get().base();
                    });
        }

