//#include "virtual_memory.h"
//#include "LinkedList.h"
//#include <iostream>
//
//int pageFault(VirtualMemory* vm, int page_num);
//int lru(LinkedList* L);
//int lru(LinkedList* L, int value);
//void swap_in(VirtualMemory* vm,int frame_num);
//void swap_out(VirtualMemory* vm,int frame_num);
//
//u32 addr_trans(VirtualMemory* vm, u32 logic_addr) {
//	int offset = logic_addr % vm->PAGESIZE;
//	int page_num = logic_addr / vm->PAGESIZE;
//	if (page_num >= 0 && page_num <= 1023) {
//		for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
//			if (vm->invert_page_table[i] != 0x80000000) {
//				if (vm->invert_page_table[i + vm->PAGE_ENTRIES] == page_num) {
//					return i * vm->PAGESIZE + offset;
//				}
//			}
//
//		}
//		int frame_num = pageFault(vm,page_num);
//		return frame_num * vm->PAGESIZE + offset;
//	}
//	else {
//		std::cout << "[ERROR] Memory out of bound!" << std::endl;
//	}
//}
//
//int pageFault(VirtualMemory* vm,int page_num) {
//	if (vm->free_frame_list[0] != 0) {
//		int i;
//		for (i = 0; i < vm->PAGE_ENTRIES; i++) {
//			if (vm->free_frame_list[i] == 0) {
//				break;
//			}
//		}
//		swap_in(vm,i);
//		vm->free_frame_list[i + 1] = 1;
//		vm->free_frame_list[0] -= 1;
//		lru(vm->LRU_List, i);
//		vm->invert_page_table[i] = 0x00000000;
//		vm->invert_page_table[i + vm->PAGE_ENTRIES] = page_num;
//		return i;
//	}
//	else {
//		int out = lru(vm->LRU_List);
//		swap_out(vm,out);
//		vm->invert_page_table[out] = 0x80000000;
//		swap_in(vm,out);
//		vm->invert_page_table[out] = 0x00000000;
//		vm->invert_page_table[out+vm->PAGE_ENTRIES] = page_num;
//		return out;
//	}
//}
//
//void swap_in(VirtualMemory* vm,int frame_num) {
//	for (int i = 0; i < vm->PAGESIZE / 4; i++) {
//		vm->storage[frame_num + i] = *(vm->storage_rw_ptr + i);
//	}
//}
//
//void swap_out(VirtualMemory* vm, int frame_num) {
//	for (int i = 0; i < vm->PAGESIZE / 4; i++) {
//		*(vm->storage_rw_ptr + i) = vm->storage[frame_num + i];
//	}
//}
//
//
//int lru(LinkedList* L) {
//	if (!L->isFull()) {
//		return -1;
//	}
//	Node* n = L->pop();
//	return n->value;
//}
//
//int lru(LinkedList* L, int value) {
//	Node* n;
//	if (L->find(value)) {
//		n = L->remove(value);
//		L->push(n);
//		return -1;
//	}
//	else {
//		n = new Node(value);
//
//		if (!L->isFull()) {
//			L->push(n);
//			return -1;
//		}
//		else {
//			Node* n1 = L->pop();
//			L->push(n);
//			return n->value;
//		}
//	}
//}