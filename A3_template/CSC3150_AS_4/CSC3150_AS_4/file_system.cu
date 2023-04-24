#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

 //__device__ __managed__ u32 gtime = 0;

__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
							int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
							int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE, 
							int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS)
{
  // init variables
  fs->volume = volume;

  // init constants
  fs->SUPERBLOCK_SIZE = SUPERBLOCK_SIZE;
  fs->FCB_SIZE = FCB_SIZE;
  fs->FCB_ENTRIES = FCB_ENTRIES;
  fs->STORAGE_SIZE = VOLUME_SIZE;
  fs->STORAGE_BLOCK_SIZE = STORAGE_BLOCK_SIZE;
  fs->MAX_FILENAME_SIZE = MAX_FILENAME_SIZE;
  fs->MAX_FILE_NUM = MAX_FILE_NUM;
  fs->MAX_FILE_SIZE = MAX_FILE_SIZE;
  fs->FILE_BASE_ADDRESS = FILE_BASE_ADDRESS;
	  
  for (int i = 0; i < fs->SUPERBLOCK_SIZE / 8; i++) {
	  fs->volume[i] = 0b11111111;
  }
}

__device__ int pow(int base, int power) {			// Pow function
	if (power == 0) return 1;
	int result = base;
	for (int i = 0; i < power-1; i++) {
		result *= base;
	}
	return result;
}

__device__ int strlen(char* str) {					// Return the length of string
	int i = 0;
	while (*(str + i) != '\0') {
		i += 1;
		if (i > 21)
			return 21;
	}
	return i;
}

__device__ void compact(FileSystem* fs, u32 file_pointer) {

	printf("[OP] Compact \n");

	uchar* file = &fs->volume[fs->SUPERBLOCK_SIZE + file_pointer * fs->FCB_SIZE];
	int fragment = *(int*)(file + 28) * 32;
	int fragment_size = *(int*)(file + 24);
	if (fragment_size % 32 != 0) fragment_size = (fragment_size / fs->STORAGE_BLOCK_SIZE + 1) * fs->STORAGE_BLOCK_SIZE;
	int bit = (fragment + fragment_size)/32;									// Position in bit map

	if (bit < fs->SUPERBLOCK_SIZE) {
		if ((fs->volume[bit / 8] >> (7 - bit % 8)) % 2 == 0) {					// Modify storage block address in FCB
			for (int i = 0; i < fs->FCB_ENTRIES; i++) {
				if (*(int*)&fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 28] > *(int*)(file+28)) {
					*(int*)&fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 28] -= fragment_size/32;
				}
			}
		}
		while (true) {												
			if ((fs->volume[bit / 8] >> (7 - bit % 8)) % 2 == 0) {				// Check if the block is free, if so, it's end
				for (int i = 0; i < 32; i++) {									// else, move contents forwards
					fs->volume[fragment + fs->FILE_BASE_ADDRESS + i] = fs->volume[fragment + fs->FILE_BASE_ADDRESS + i + fragment_size];
				}
				fragment += 32;
				bit += 1;
			}
			else break;
		}
	}
	/* free ending bit map */
	for (int j = 0; j < fragment_size / fs->STORAGE_BLOCK_SIZE; j++) {
		fs->volume[(fragment / fs->STORAGE_BLOCK_SIZE + j) / 8] += pow(2,(7 - (fragment / fs->STORAGE_BLOCK_SIZE + j) % 8));
	}
	for (int i = 0; i < fragment_size; i++) {				// Reset the storage
		fs->volume[fs->FILE_BASE_ADDRESS + fragment + i] = '\0';
	}
	*(int*)(file + 28) = fragment / 32;											// Re-allocate block
	printf("[OP] Compact Finish! \n");
}

__device__ u32 fs_open(FileSystem *fs, char *filename, int operation) {

	printf("\n[OP] File open: %s \n",filename);

	/* Check the length of filename */
	if (strlen(filename) > 20) {
		printf("[ERROR] File name too long");
	}

	u32 block_num = 0;
	int file_pointer = 0;
	bool file_exist = false;

	/* Search the file in FCB */
	for (int i = 0; i < fs->FCB_ENTRIES; i++) {
		int j = 0;
		while (fs->volume[fs->SUPERBLOCK_SIZE + 32 * i + j] != '\0') {
			file_exist = true;
			if (fs->volume[fs->SUPERBLOCK_SIZE + 32 * i + j] != *(filename + j)) {
				file_exist = false;
				break;
			}
			j += 1;
		}
		if (file_exist == true) {
			file_pointer = i;
			break;
		}
	}
	/* If the file does not exist, create a new one */
	if (!file_exist) {																		
		if (operation == G_READ) {																			// Cannot read the file
			printf("[ERROR] No such file! Cannot read it!");
			return -1;
		}
		else if (operation == G_WRITE) {
			if (fs->current_file_num >= fs->MAX_FILE_NUM) {													
				printf("[ERROR] FCB is full!\n");															// Check if the FCB is full
				return -1;
			}
			//printf("[INFO] Current file num %d\n",fs->current_file_num);
			int i, j;
			bool free_flag = false;
			for (i = 0; i < fs->SUPERBLOCK_SIZE / 8; i++) {													// Find free block
				for (j=7; j >= 0; j--) {
					int k = (fs->volume[i] >> j);
					if (k % 2 == 1) {
						free_flag = true;
						break;
					}
				}
				if (free_flag) break;
			}
			fs->volume[i] -= pow(2, j);																		// Mark the bit as occupied
			block_num = (8 * i + 7 - j);
			printf("[INFO] File not exist\n");
			printf("[INFO] ------------------------ New block number %d\n", block_num);

			for (int i = 0; i < fs->FCB_ENTRIES; i++) {										
				if (fs->volume[fs->SUPERBLOCK_SIZE + 32 * i] == '\0') {
					file_pointer = i;
					break; 
				}
			}

			for (int count = 0; count < strlen(filename); count++) {										// Copy file name
				fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * file_pointer + count] = (uchar)*(filename + count);
			}
			*(u32*)&fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * file_pointer + 20] = fs->time;			// Record mod time
			*(int*)&fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * file_pointer + 24] = 0;				// Default size 32
			*(int*)&fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * file_pointer + 28] = block_num;			// Block address
			fs->current_file_num += 1;
		}
	}

	printf("[OP] File successfully open! \n");
	return file_pointer;
}

__device__ void fs_read(FileSystem *fs, uchar *output, u32 read_size, u32 file_pointer) {

	printf("\n[OP] Read file: %s\n",&fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * file_pointer]);

	fs->time += 1;

	if (/*!read permission*/0) {																			// User root always has permission
		return;
	}
	if (read_size > *(int*)(&fs->volume[fs->SUPERBLOCK_SIZE + file_pointer * fs->FCB_SIZE]+24)) {
		printf("[ERROR] Read size greater than file size!\n");
		return;
	}
	for (int i = 0; i < read_size; i++) {
		int file_addr = fs->FILE_BASE_ADDRESS + *(int*)&fs->volume[fs->FCB_SIZE * file_pointer + fs->SUPERBLOCK_SIZE + 28] * fs->STORAGE_BLOCK_SIZE;
		*(output + i) = fs->volume[file_addr + i];
	}
}

__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 input_size, u32 file_pointer) {

	printf("\n[OP] Write %s  Size %d \n", &fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * file_pointer],input_size);
	if (input_size > 1024) printf("[ERROR] Exceed max file size!\n");
	fs->time += 1;

	uchar* file = &fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * file_pointer];
	*(int*)(file +20) = fs->time;
	if (/*!write permission*/0) {
		return;
	}
	int original_block_num = 1;
	if (*(int*)(file + 24) > 0) original_block_num = (*(int*)(file + 24) - 1) / 32;
	if ((input_size-1)/32 != original_block_num) {			// If the input size occupies different #blocks with original file compact
		compact(fs, file_pointer);									// reallocate and write
		*(int*)(file + 24) = input_size;
		return fs_write(fs, input, input_size, file_pointer);		// recursion print write twice...
	}
	*(int*)(file + 24) = input_size;
	int file_addr = fs->FILE_BASE_ADDRESS + *(int*)&fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * file_pointer + 28] * fs->STORAGE_BLOCK_SIZE;

	for (int i = 0; i < input_size; i++) {		
		fs->volume[file_addr + i] =  *(input + i);

		if (i % fs->STORAGE_BLOCK_SIZE == 0) {						// Occupy bit map
			int bit = *(int*)(file + 28) + i/32;
			if ((fs->volume[bit / 8] >> (7 - bit % 8)) % 2 == 1) {
				fs->volume[bit / 8] -= pow(2, (7 - bit % 8));

			}
		}
	}

	printf("[OP] Write successfully! \n");
	return 0;
}

__device__ void fs_gsys(FileSystem *fs, int operation) {

	printf("\n[OP] List ");
	if (operation == LS_D) printf("LS_D ");
	else printf("LS_S ");
	printf("\n");

	/* gather all files */
	int* files = new int[1024];							// stack overflow on my computer， move to heap x_x
	int sort_item = 20;
	int count = 0;
	for (int i = 0; i < fs->FCB_ENTRIES; i++) {
		int fcb_pos = 32 * i;
		if (fs->volume[fs->SUPERBLOCK_SIZE + fcb_pos] != '\0') {
			files[count] = fcb_pos;
			count += 1;
		}
	}

	/* Sort */
	printf("Sort..");
	if (operation == LS_S) sort_item = 24;			// Sort_item(addr offset in fcb), indicates the type of sort
	for (int i = 0; i < count; i++) {
		for (int j = i + 1; j < count; j++) {
			if (*(int*)(&(fs->volume[fs->SUPERBLOCK_SIZE + files[i]]) + sort_item)\
				< *(int*)(&(fs->volume[fs->SUPERBLOCK_SIZE + files[j]]) + sort_item)) {
				int temp = files[j];
				files[j] = files[i];
				files[i] = temp;
			}
			else if (sort_item == 24 && *(int*)(&(fs->volume[fs->SUPERBLOCK_SIZE + files[i]])\
				+ sort_item) == *(int*)(&(fs->volume[fs->SUPERBLOCK_SIZE + files[j]]) + sort_item)) {

				if (*(int*)(&(fs->volume[fs->SUPERBLOCK_SIZE + files[i]]) + 20) < *(int*)\
					(&(fs->volume[fs->SUPERBLOCK_SIZE + files[j]]) + 20)) {
					int temp = files[j];
					files[j] = files[i];
					files[i] = temp;
				}
			}
		}
	}

	printf("Display..\n");
	printf(" Filename | Mod time | Size | Owner | Permission |\n");
	for (int i = 0; i < count; i++) {
		printf("%s          %d       %d      root     rwe\n",
			(char*)&(fs->volume[fs->SUPERBLOCK_SIZE + files[i]]),
			*(int*)(&(fs->volume[fs->SUPERBLOCK_SIZE + files[i]]) + 20),
			*(int*)(&(fs->volume[fs->SUPERBLOCK_SIZE + files[i]]) + 24));
	}
}

__device__ void fs_gsys(FileSystem *fs, int operation, char *filename) {

	printf("\n[OP] Remove %s \n",filename);

	u32 file_pointer;
	uchar* file;
	bool file_exist = false;
	int i;
	int j = 0;
	for (i = 0; i < fs->FCB_ENTRIES; i++) {
		if (fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i] == '\0') continue;			// Find the file
		printf("[INFO] Filename match! ");
		while (fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + j] != '\0') {
			file_exist = true;
			printf("%c", fs->volume[fs->SUPERBLOCK_SIZE + 32 * i + j]);
			if (fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + j] != *(filename + j)) {
				file_exist = false;
				break;
			}
			j += 1;
		}
		printf("\n");
		if (file_exist == true) {
			file_pointer = i;
			file = &fs->volume[fs->SUPERBLOCK_SIZE + 32 * file_pointer];
			break;
		}
	}

	if (!file_exist) {
		printf("[ERROR] No such file! Cannot remove!");
		return;
	}
	else if (operation < RM) {
		// List this file and print you give a list operation
		printf(" Filename | Mod time | Size | Owner | Permission |\n");
		printf("%s          %d       %d      root     rwe\n",
			file, *(int*)(file + 24), *(int*)(file + 20));
	}
	else {
		compact(fs, *(int*)(file + 28));
		int file_block_size = *(int*)(file + 24);
		if (file_block_size % 32 != 0) file_block_size = (file_block_size / fs->STORAGE_BLOCK_SIZE + 1) * fs->STORAGE_BLOCK_SIZE;

		*file = '\0';											// Free the fcb block
		fs->current_file_num -= 1;
	}
	printf("\n[OP] Successfully Removed \n");
}















