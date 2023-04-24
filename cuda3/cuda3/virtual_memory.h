#ifndef VIRTUAL_MEMORY_H
#define VIRTUAL_MEMORY_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <inttypes.h>
#include <unordered_map>
#include <list>

typedef unsigned char uchar;
typedef uint32_t u32;




class Node {
public:
	__device__ Node(int val = 0);
	__device__ ~Node();

	int value;
	Node* next;
	Node* prev;
	__device__ void connect(Node* n = nullptr);

};


class LinkedList {
public:
	__device__ LinkedList(int cap = 4);
	__device__ ~LinkedList();
	Node* head;
	Node* tail;

	__device__ int cap();
	__device__ int len();
	__device__ bool isEmpty();
	__device__ bool isFull();
	__device__ bool find(int value);
	__device__ Node* remove(int value);
	__device__ Node* pop();
	__device__ bool push(int value);
	__device__ void toString();
private:
	__device__ void setCap(int cap);
	int capacity;
	int length;
};



//class LRUCache {
//public:
//	__device__ LRUCache(int capacity);
//	__device__ ~LRUCache();
//	__device__ int get(int frame_num);
//	__device__ int put(int frame_num);
//	__device__ int pop();
//private:
//	int _capacity;
//	std::list<int> _cache;
//	std::unordered_map<int, std::list<int>::iterator> _map;
//};

struct VirtualMemory {
	uchar *buffer;
	uchar *storage;
	u32 *invert_page_table;
	int *pagefault_num_ptr;

	int PAGESIZE;
	int INVERT_PAGE_TABLE_SIZE;
	int PHYSICAL_MEM_SIZE;
	int STORAGE_SIZE;
	int PAGE_ENTRIES;

	//LinkedList* LRU_List;
	//int* free_frame_list;
	//uchar* storage_rw_ptr;
	//int* swap_list;
	//int last_frame;
};

// TODO
__device__ void vm_init(VirtualMemory *vm, uchar *buffer, uchar *storage,
                        u32 *invert_page_table, int *pagefault_num_ptr,
                        int PAGESIZE, int INVERT_PAGE_TABLE_SIZE,
                        int PHYSICAL_MEM_SIZE, int STORAGE_SIZE,
                        int PAGE_ENTRIES/*,LinkedList* L,int* free_frame_list,int* swap_list*/);
__device__ uchar vm_read(VirtualMemory *vm, u32 addr);
__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value);
__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset,
                            int input_size);
__device__ int pageFault(VirtualMemory* vm, int page_num);
__device__ int lru(LinkedList* L);
__device__ int lru(LinkedList* L, int value);
__device__ void swap_in(VirtualMemory* vm, int frame_num,int page_num);
__device__ void swap_out(VirtualMemory* vm, int frame_num);
__device__ int find_page_in_storage(VirtualMemory* vm, int page_num);

__device__ u32 addr_trans(VirtualMemory* vm, u32 logic_addr);
__device__ void find_empty_in_storage(VirtualMemory* vm, int page_num);


#endif
