#include "LRU.h"
#include "LinkedList.h"


int lru(LinkedList* L) {
	if (! L->isFull()) {
		return -1;
	}
	Node* n = L->pop();
	return n->value;
}

int lruUpdate(LinkedList* L, int value) {
	Node* n;
	if (L->find(value)) {
		n = L->remove(value);
		L->push(n);
		return -1;
	}
	else {
		n = new Node(value);

		if (!L->isFull()) {
			L->push(n);
			return -1;
		}
		else {
			Node* n1 = L->pop();
			L->push(n);
			return n->value;
		}
	}
}
