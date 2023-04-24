//#include "LinkedList.h"
//#include "cuda.h"
//#include "cuda_runtime.h"
////#include <iostream>
//
//__device__ Node::Node(int val) {
//	value = val;
//	next = nullptr;
//	prev = nullptr;
//}
//
//__device__ Node::~Node() {
//}
//
//__device__ void Node::connect(Node* n) {
//	if (n != nullptr) {
//		this->next = n;
//		this->next->prev = this;
//	}
//}
//
//
//__device__ LinkedList::LinkedList(int cap) {
//	capacity = cap;
//	head = nullptr;
//	tail = nullptr;
//	length = 0;
//}
//
//__device__ LinkedList::~LinkedList() {
//
//}
//
//__device__ int LinkedList::cap() {
//	return this->capacity;
//}
//
//__device__ int LinkedList::len() {
//	return this->length;
//}
//
//__device__ bool LinkedList::isEmpty() {
//	return this->length == 0;
//}
//
//__device__ bool LinkedList::isFull() {
//	return this->length == this->capacity;
//}
//
//__device__ bool LinkedList::find(int value) {
//	Node* iter = this->head;
//	while (iter != nullptr) {
//		if (iter->value == value) {
//			return true;
//		}
//		iter = iter->next;
//	}
//	return false;
//}
//
//__device__ Node* LinkedList::remove(int value) {
//	if (this->isEmpty()) {
//		return nullptr;
//	}
//	else {
//		Node* iter = this->head;
//		while (iter != nullptr) {
//			if (iter->value == value) {
//				if (iter == this->head) {
//					iter->next->prev = nullptr;
//					this->head = iter->next;
//					iter->next = nullptr;
//					this->length -= 1;
//					return iter;
//				}
//				else if(iter == this->tail){
//					iter->prev->next = nullptr;
//					this->tail = iter->prev;
//					iter->prev = nullptr;
//					this->length -= 1;
//					return iter;
//				}
//				else {
//					iter->prev->connect(iter->next);
//					iter->next = nullptr;
//					iter->prev = nullptr;
//					this->length -= 1;
//					return iter;
//				}
//			}
//			else {
//				iter = iter->next;
//			}
//		}
//		return nullptr;
//	}
//}
//
//__device__ Node* LinkedList::pop() {
//	if (this->isEmpty()) {
//		return nullptr;
//	}
//	Node* n = this->head;
//	this->head = n->next;
//	this->head->prev = nullptr;
//	n->next = nullptr;
//	this->length -= 1;
//	return n;
//}
//
//__device__ bool LinkedList::push(Node* n) {
//	if (this->isEmpty()) {
//		this->head = n;
//		this->tail = n;
//	}
//	else if(this->length == this->capacity){
//		return false;
//	}
//	else {
//		this->tail->connect(n);
//		this->tail = n;
//	}
//	this->length += 1;
//	return true;
//}
//
//__device__ void LinkedList::setCap(int cap) {
//	this->capacity = cap;
//}
//
//__device__ void LinkedList::toString() {
//	if (this->isEmpty()) {
//		//std::cout << "Empty linked list!" << std::endl;
//		return;
//	}
//	Node* iter = this->head;
//	while (iter != nullptr) {
//		//std::cout << iter->value << " <-> ";
//		iter = iter->next;
//	}
//	//std::cout << "null" << std::endl;
//	return;
//}

