#ifndef MEMORY_FREE_H
#define MEMORY_FREE_H

#ifdef __arm__
// For ARM Cortex-M0 and M4 based boards like the Nano 33 BLE
extern "C" char* sbrk(int incr);

int freeMemory() {
  char top;
  return &top - reinterpret_cast<char*>(sbrk(0));
}

#else
// Alternative implementation for other boards
extern char _end;
extern "C" char *sbrk(int i);

int freeMemory() {
  char *heapend = (char*)sbrk(0);
  char *stack_ptr = (char*)__builtin_frame_address(0);
  return stack_ptr - heapend;
}
#endif

#endif // MEMORY_FREE_H