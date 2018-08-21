// Base-2 logarithm bithack.

#ifndef _AFJDFJSDFSD_PYOPENCL_HEADER_SEEN_BITLOG_HPP
#define _AFJDFJSDFSD_PYOPENCL_HEADER_SEEN_BITLOG_HPP


#include <climits>
#include <cstdint>


namespace pyopencl
{
  extern const char log_table_8[];

  inline unsigned bitlog2_16(uint16_t v)
  {
    if (unsigned long t = v >> 8)
      return 8+log_table_8[t];
    else 
      return log_table_8[v];
  }

  inline unsigned bitlog2_32(uint32_t v)
  {
    if (uint16_t t = v >> 16)
      return 16+bitlog2_16(t);
    else 
      return bitlog2_16(v);
  }

  inline unsigned bitlog2(unsigned long v)
  {
#if (ULONG_MAX != 4294967295)
    if (uint32_t t = v >> 32)
      return 32+bitlog2_32(t);
    else 
#endif
      return bitlog2_32(v);
  }
}





#endif
