int main()
{
#ifndef __AVX512F__
    static_assert(false);
#endif
    return 0;
}
