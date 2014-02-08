// This file was automatically created from queue_defs.v. Do not edit.
//// Thread Manager definitions

// per-thread memory space (64-bit data):
// higher-order bits of the address determine the thread slot number;
// TM_REQ_BITS is the number of LSBits in the address indicating the
// register number to service a given request
#define TM_REQ_BITS 8
// suported requests
#define TM_REQ_NOOP    0x0
// bit index of or-able requests
#define TM_REQ_SETHEAD_BIT 0
#define TM_REQ_SETLAST_BIT 1
#define TM_REQ_GETHEAD_BIT 2
#define TM_REQ_GETLAST_BIT 3
// the following are or-able, but only one of GET/SET may be present
#define TM_REQ_SETHEAD (0x1 << TM_REQ_SETHEAD_BIT)
#define TM_REQ_SETLAST (0x1 << TM_REQ_SETLAST_BIT)
#define TM_REQ_GETHEAD (0x1 << TM_REQ_GETHEAD_BIT)
#define TM_REQ_GETLAST (0x1 << TM_REQ_GETLAST_BIT)
// subrequests
#define TM_REQ_RESET   0xf0
#define TM_REQ_GETCNT  0x10
#define TM_REQ_GETSTAT 0x20
#define TM_REQ_GETSIZE 0x40
// subrequest mask
#define TM_REQ_SUBMSK_BITLO 4
#define TM_REQ_SUBMSK_BITHI 7
#define TM_REQ_SUBREQ_RSH   4

//// status codes:
#define TM_STAT_BITS 4
#define TM_STAT_SUCCESS 0x0
#define TM_STAT_INVALID 0x1
#define TM_STAT_NOSPACE 0x2
#define TM_STAT_EMPTY   0x3

//// storage capacity
#define TM_ADDR_BITS 14
