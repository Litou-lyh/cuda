#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <ctime>
#include <unordered_map>

__device__ void init_invert_page_table(VirtualMemory *vm) {

	for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
		vm->invert_page_table[i] = 0x80000000; // invalid := MSB is 1
		vm->invert_page_table[i + vm->PAGE_ENTRIES] = i;
	}
}

__device__ void vm_init(VirtualMemory *vm, uchar *buffer, uchar *storage,
                        u32 *invert_page_table, int *pagefault_num_ptr,
                        int PAGESIZE, int INVERT_PAGE_TABLE_SIZE,
                        int PHYSICAL_MEM_SIZE, int STORAGE_SIZE,
                        int PAGE_ENTRIES/*,LinkedList* L,int* free_frame_list,int* swap_list*/) {
	// init variables
	vm->buffer = buffer;
	vm->storage = storage;
	vm->invert_page_table = invert_page_table;
	vm->pagefault_num_ptr = pagefault_num_ptr;

	// init constants
	vm->PAGESIZE = PAGESIZE;
	vm->INVERT_PAGE_TABLE_SIZE = INVERT_PAGE_TABLE_SIZE;
	vm->PHYSICAL_MEM_SIZE = PHYSICAL_MEM_SIZE;
	vm->STORAGE_SIZE = STORAGE_SIZE;
	vm->PAGE_ENTRIES = PAGE_ENTRIES;
	//vm->LRU_List = L;
	//vm->free_frame_list = free_frame_list;
	//vm->storage_rw_ptr = storage;
	//vm->swap_list = swap_list;
	// before first vm_write or vm_read
	init_invert_page_table(vm);

}

__device__ uchar vm_read(VirtualMemory *vm, u32 addr) {
	/* Complate vm_read function to read single element from data buffer */
	//clock_t time_start = clock();
	//u32 phy_addr = addr_trans(vm, addr);
	////clock_t time_end = clock();
	////printf("%f\n", (time_end - time_start) / (double)CLOCKS_PER_SEC);
	//if (addr % 1024 == 0)
	//	printf("logic address: %d  physical addr: %d\n", addr, phy_addr);

	return vm->buffer[addr]; //TODO
}

__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value) {
	/* Complete vm_write function to write value into data buffer */
	//clock_t time_start = clock();
	//u32 phy_addr = addr_trans(vm, addr);
	////clock_t time_end = clock();
	////printf("total trans: %f\n", (time_end - time_start) / (double)CLOCKS_PER_SEC);
	//if (addr % 1024 ==0)
	//	printf("logic address: %d  physical addr: %d\n", addr, phy_addr);
	//vm->buffer[phy_addr] = value;
}

__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset,
                            int input_size) {
	/* Complete snapshot function togther with vm_read to load elements from data
	* to result buffer */
	//for (int i = offset; i < vm->PHYSICAL_MEM_SIZE; i++) {
	//	
	//	*(results+i-offset) = vm_read(vm, i);

	//}
}

//__device__ u32 addr_trans(VirtualMemory* vm, u32 logic_addr) {
//	int offset = logic_addr % vm->PAGESIZE;
//	//clock_t time_start = clock();
//	int page_num = logic_addr / vm->PAGESIZE;
//	if (vm->invert_page_table[vm->last_frame] != 0x80000000 && vm->invert_page_table[vm->last_frame + vm->PAGE_ENTRIES] == page_num)
//	{
//		return vm->last_frame * vm->PAGESIZE + offset;
//	}
//
//	for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
//		if (i == 0)
//
//		if (vm->invert_page_table[i] != 0x80000000) {
//			if (vm->invert_page_table[i + vm->PAGE_ENTRIES] == page_num) {
//				lru(vm->LRU_List, i);			
//				// <--
//				return i * vm->PAGESIZE + offset;
//			}
//		}
//
//	}
//	//clock_t time_table = clock();
//
//	int frame_num = pageFault(vm, page_num);
//
//	//clock_t time_fault = clock();
//	//printf("table : fault %f  :  %f\n", (time_table - time_start) / (double)CLOCKS_PER_SEC, (time_fault - time_table) / (double)CLOCKS_PER_SEC);
//	vm->last_frame = frame_num;
//	return frame_num * vm->PAGESIZE + offset;
//	
//
//}
//
//__device__ int pageFault(VirtualMemory* vm, int page_num) {
//	*vm->pagefault_num_ptr += 1;
//	printf("pagefault_num: %d\n",*vm->pagefault_num_ptr);
//	if (vm->free_frame_list[0] != 0) {
//		int i;
//		for (i = 0; i < vm->PAGE_ENTRIES; i++) {
//			if (vm->free_frame_list[i+1] == 0) {
//				break;
//			}
//		}
//		swap_in(vm,i, page_num);
//		lru(vm->LRU_List, i);													// <---
//
//		vm->free_frame_list[i + 1] = 1;
//		vm->free_frame_list[0] -= 1;
//		vm->invert_page_table[i] = 0x00000000;
//		vm->invert_page_table[i + vm->PAGE_ENTRIES] = page_num;
//		return i;
//	}
//	else {
//		int out = lru(vm->LRU_List);
//
//		if (page_num % 1024 == 0)
//			printf("swap out: %d\n", out);
//		swap_out(vm, out);
//		vm->free_frame_list[out + 1] = 0;
//		vm->free_frame_list[0] += 1;
//		vm->invert_page_table[out] = 0x80000000;
//		swap_in(vm, out,page_num);
//		lru(vm->LRU_List, out);
//		vm->invert_page_table[out] = 0x00000000;
//		vm->free_frame_list[out + 1] = 1;
//		vm->free_frame_list[0] -= 1;
//		vm->invert_page_table[out + vm->PAGE_ENTRIES] = page_num;
//		return out;
//	}
//}
//
//__device__ void swap_in(VirtualMemory* vm, int frame_num,int page_num) {
//	if (page_num % 1024 == 0)
//		printf("swap in: %d\n", page_num);
//	int a = find_page_in_storage(vm, page_num);
//	if (a == -1) return;
//	for (int i = 0; i < vm->PAGESIZE / 4; i++) {
//		vm->buffer[frame_num + i] = *(vm->storage_rw_ptr + i);
//	}
//}
//
//__device__ void swap_out(VirtualMemory* vm, int frame_num) {
//	find_empty_in_storage(vm,vm->invert_page_table[frame_num+vm->PAGE_ENTRIES]);
//	for (int i = 0; i < vm->PAGESIZE / 4; i++) {
//		*(vm->storage_rw_ptr + i) = vm->buffer[frame_num + i];
//	}
//}
//
//
//__device__ int lru(LinkedList* L) {
//	if (!L->isFull()) {
//		return -1;
//	}
//	Node* n = L->pop();
//	int val = n->value;
//	delete n;
//	return val;
//}
//
//__device__ int lru(LinkedList* L, int value) {
//	Node* n;
//	if (L->find(value)) {
//
//		n = L->remove(value);
//		L->push(n->value);
//		return -1;
//	}
//	else {
//		n = new Node(value);
//
//		if (!L->isFull()) {
//			L->push(n->value);
//			return -1;
//		}
//		else {
//			Node* n1 = L->pop();
//			L->push(n->value);
//			int val = n1->value;
//			delete n1;
//			return val;
//		}
//	}
//}
//
//__device__ int find_page_in_storage(VirtualMemory* vm, int page_num) {
//	for (int i = 0; i < 1024; i++) {
//		if (vm->swap_list[i] == page_num) {
//			vm->storage_rw_ptr = &vm->storage[vm->PAGESIZE * i];
//			return 0;
//		}
//	}
//	return -1;
//}
//__device__ void find_empty_in_storage(VirtualMemory* vm, int page_num) {
//	for (int i = 0; i < 1024; i++) {
//		if (vm->swap_list[i] == -1) {
//			vm->storage_rw_ptr = &vm->storage[vm->PAGESIZE  * i];
//		}
//	}
//}


//LRUCache::LRUCache(int capacity) {
//	_capacity = capacity;
//}
//LRUCache::~LRUCache() {
//
//}
//int LRUCache::get(int frame_num) {
//	const auto it = _map.find(frame_num);
//	if (it == _map.cend()) return -1;
//	_cache.splice(_cache.begin(), _cache, it->second);
//	return *it->second;
//}
//int LRUCache::put(int frame_num) {
//	const auto it = _map.find(frame_num);
//	if (it != _map.cend()) {
//		_cache.splice(_cache.begin(), _cache, it->second);
//		return -1;
//	}
//	else if (_cache.size() == _capacity) {
//		const auto& node = _cache.back();
//		_map.erase(node);
//		_cache.pop_back();
//		return node;
//	}
//}


__device__ Node::Node(int val) {
	value = val;
	next = NULL;
	prev = NULL;
}

__device__ Node::~Node() {
}

__device__ void Node::connect(Node* n) {
	if (n != NULL) {
		this->next = n;
		this->next->prev = this;
	}
}


__device__ LinkedList::LinkedList(int cap) {
	capacity = cap;
	head = NULL;
	tail = NULL;
	length = 0;
}

__device__ LinkedList::~LinkedList() {

}

__device__ int LinkedList::cap() {
	return this->capacity;
}

__device__ int LinkedList::len() {
	return this->length;
}

__device__ bool LinkedList::isEmpty() {
	return this->length == 0;
}

__device__ bool LinkedList::isFull() {
	return this->length == this->capacity;
}

__device__ bool LinkedList::find(int value) {

	Node* iter = this->head;
	while (iter != NULL) {
		if (iter->value == value) {
			return true;
		}
		iter = iter->next;
	}
	return false;
}

__device__ Node* LinkedList::remove(int value) {
	if (this->isEmpty()) {
		return NULL;
	}
	else {
		Node* iter = this->head;
		while (iter != NULL) {
			if (iter->value == value) {
				if (iter == this->head) {
					iter->next->prev = NULL;
					this->head = iter->next;
					iter->next = NULL;
					this->length -= 1;
					return iter;
				}
				else if (iter == this->tail) {
					iter->prev->next = NULL;
					this->tail = iter->prev;
					iter->prev = NULL;
					this->length -= 1;
					return iter;
				}
				else {
					iter->prev->connect(iter->next);
					iter->next = NULL;
					iter->prev = NULL;
					this->length -= 1;
					return iter;
				}
			}
			else {
				iter = iter->next;
			}
		}
		return NULL;
	}
}

__device__ Node* LinkedList::pop() {
	if (this->isEmpty()) {
		return NULL;
	}
	Node* n = this->head;
	this->head = n->next;
	this->head->prev = NULL;
	n->next = NULL;
	this->length -= 1;
	return n;
}

__device__ bool LinkedList::push(int value) {
	Node* n = new Node(value);
	if (this->isEmpty()) {
		this->head = n;
		this->tail = n;
	}
	else if (this->length == this->capacity) {
		return false;
	}
	else {
		this->tail->connect(n);
		this->tail = n;
	}
	this->length += 1;
	return true;
}

__device__ void LinkedList::setCap(int cap) {
	this->capacity = cap;
}

__device__ void LinkedList::toString() {
	if (this->isEmpty()) {
		printf("Empty linked list!\n");
		return;
	}
	Node* iter = this->head;
	while (iter != NULL) {
		printf("%d <-> ",iter->value);
		iter = iter->next;
	}
	printf("null\n");
	return;
}