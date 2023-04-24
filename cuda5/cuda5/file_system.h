#ifndef VIRTUAL_MEMORY_H
#define VIRTUAL_MEMORY_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <inttypes.h>

typedef unsigned char uchar;
typedef uint32_t u32;

#define G_WRITE 1
#define G_READ 0
#define LS_D 0
#define LS_S 1
#define RM 2
#define CD 3
#define CD_P 4
#define CD_ROOT 333
#define RM_RF 5
#define PWD 6






struct FileSystem {
	uchar *volume;
	int SUPERBLOCK_SIZE;
	int FCB_SIZE;
	int FCB_ENTRIES;
	int STORAGE_SIZE;
	int STORAGE_BLOCK_SIZE;
	int MAX_FILENAME_SIZE;
	int MAX_FILE_NUM;
	int MAX_FILE_SIZE;
	int FILE_BASE_ADDRESS;

	int current_file_num = 0;
	u32 time = 0;

	uchar* dir;
	int current_dir = -1;
	int DIR_SIZE;
	int DIR_ENTRIES;
};


__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
	int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
	int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE,
	int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS, uchar* dir,int DIR_SIZE,int DIR_ENTRIES);

__device__ u32 fs_open(FileSystem *fs, char *s, int op);
__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp);
__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp);
__device__ void fs_gsys(FileSystem *fs, int op);
__device__ void fs_gsys(FileSystem *fs, int op, char *s);


__device__ int pow(int base, int power);
__device__ int strlen(char* str);
__device__ void compact(FileSystem* fs, u32 file_pointer);



__device__ void mkdir(FileSystem* fs, char* dir_name);
__device__ void cd(FileSystem* fs, int operation, char* dir_name);
__device__ void cd_p(FileSystem* fs, int operation);
__device__ void cd_root(FileSystem* fs, int operation, char* dir_name);
__device__ void fs_gsys1(FileSystem *fs, int operation);
__device__ void pwd(FileSystem* fs, int operation);
__device__ void rmrf(FileSystem* fs, int operation, char* dir_name);



#endif
