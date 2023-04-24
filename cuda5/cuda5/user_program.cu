#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>

__device__ void user_program(FileSystem *fs, uchar *input, uchar *output) {
	
	/////////////////////// Bonus Test Case ///////////////
	u32 fp = fs_open(fs, "t.txt\0", G_WRITE);
	fs_write(fs, input, 64, fp);
	fp = fs_open(fs, "b.txt\0", G_WRITE);
	fs_write(fs, input + 32, 32, fp);
	fp = fs_open(fs, "t.txt\0", G_WRITE);
	fs_write(fs, input + 32, 32, fp);
	fp = fs_open(fs, "t.txt\0", G_READ);
	fs_read(fs, output, 32, fp);
	fs_gsys1(fs, LS_D);
	fs_gsys1(fs, LS_S);
	mkdir(fs, "app\0");
	fs_gsys1(fs, LS_D);
	fs_gsys1(fs, LS_S);
	cd(fs, CD,"app\0");
	fs_gsys1(fs, LS_S);
	fp = fs_open(fs, "a.txt\0", G_WRITE);
	fs_write(fs, input + 128, 64, fp);
	fp = fs_open(fs, "b.txt\0", G_WRITE);
	fs_write(fs, input + 256, 32, fp);
	mkdir(fs, "soft\0");
	fs_gsys1(fs, LS_S);
	fs_gsys1(fs, LS_D);
	cd(fs, CD, "soft\0");
	fs_gsys(fs, PWD);
	fp = fs_open(fs, "A.txt\0", G_WRITE);
	fs_write(fs, input + 256, 64, fp);
	fp = fs_open(fs, "B.txt\0", G_WRITE);
	fs_write(fs, input + 256, 1024, fp);
	fp = fs_open(fs, "C.txt\0", G_WRITE);
	fs_write(fs, input + 256, 1024, fp);
	fp = fs_open(fs, "D.txt\0", G_WRITE);
	fs_write(fs, input + 256, 1024, fp);
	fs_gsys1(fs, LS_S);
	cd_p(fs, CD_P);
	fs_gsys1(fs, LS_S);
	pwd(fs, PWD);
	cd_p(fs, CD_P);
	fs_gsys1(fs, LS_S);
	cd(fs, CD, "app\0");
	rmrf(fs, RM_RF, "soft\0");
	fs_gsys1(fs, LS_S);
	cd_p(fs, CD_P);
	fs_gsys1(fs, LS_S);

}
