# **CSC3150** **Notes**





## III. Memory management

### **Chapter 8 Virtual Memory**

#### 8.3 Copy-on-Write

​		Copy-on-Write technique allows parent and child to initially share the same pages and only copy a page if someone wants to modify it.

​		Many OS provide a **free page pool** with **zero-fill-on-demand** (erased) for such requests.



------



#### 8.4 Page Replacement

##### 8.4.1 Basic Page Replacement

1. Find the location of the desired page on the disk

2. Find a free frame:

    a. If there is a free frame, use it

    b. If no free frame, use pr al to select a **victim** frame

    c. Write victim frame to disk

3.  Read the desired page into freed frame, update the page and frame table

​		If no free frame page fault operations are doubled, we use a **modified bit** to promote it. If the victim page is modified, then it is required to write back to disk, while if it’s not modified, then there is no need for write (already in disk).

​		We must develop **frame-allocation algorithm** and **page replacement algorithm**. 



##### 8.4.2 FIFO Page Replacement (queue)

​		FIFO: not perform good, may cause Belady’s anomaly: page fault increase as #frames increase



##### 8.4.4 LRU Page Replacement

1.Counter-based 2. Stack-based

​		Both updated every memory reference and need hardware support, slow down memory operations.



##### 8.4.5 LRU-Approximation Page Replacement

​		Many OS provide a **reference bit**. 

**Additional-Reference-Bits Algorithm**:

8-bit reference:  used: 1 -> 01101001 => 10110100 > 01111111	

**Second-Chance-Algorithm** (circular queue):

Based on FIFO, if a is to be selected, if its reference bit is set (1), give it a second chance and clear it reference bit, reset the arrival time, and put it back to the end of the queue.

**Enhanced-Second-Chance-Algorithm** (two reference bits):

reference bits (0, 0): not recently used or modified; best to replace.

​						  (0, 1): could be replaced, but need to write back

​						   (1, 0) or (1, 1): are likely to be used again



##### 8.4.6 Counting-Based Page Replacement

**Least-frequently-used (LRU)**: 

**Most-frequently-used (LMU)**: pages with small count are just arrive and not used 



##### 8.4.7 Page-Buffering Algorithm

a. OS keep a pool of free frames, and read into those free frames before write the victim frame back, and later add the victim frame into free list.

b. Maintain a list of modified pages, choose modified page to write back and reset modified bit.

c. Keep a pool of free frames and reuse when it was mistakenly swap to disk.



------



#### 8.5 Allocation of Frames

##### 8.5.1 Minimum #frames

A min #frames will be allocated to each process in order to improve performance.



##### 8.5.2 Allocation Algorithms

**Equal allocation**: Split all the frames into m/n-size for n processes, use the rest m%n frames as free frame pool.

**Proportional allocation**: Allocation according to the size of the process



##### 8.5.3 Global vs Local Allocation

​		**Global**: allow to replace any frames including other process

​		**Local**: only replace in process field

​		“Global replacement generally results in greater system throughput and is therefore the more common method.”



##### 8.4.5 Non-Uniform Memory Access (NUMA)

​		Systems in which memory access time vary significantly, caused by multiple system boards. Solaris create **lgroup** to group near CPUs, memory and try to allocate resource with the group.



------



#### 8.6 Thrashing

​		**Thrashing** is high paging activity. A process is thrashing if it’s spending more time paging than executing.

##### 8.6.1 Cause of thrashing 

​		Early OS wants to increase the utilization of CPU, and introduces more processes. If the original processes suddenly need more frames and take up the frames of later processes, causing “snow-slide” chain reaction to cause frequent paging. What’s more, this decreases the CPU utilization, then OS want to introduce new processes again...

​		To prevent this, we need to provide a process with as many frames as it needs. one method is **working-set model**. 



##### 8.6.2 Working-set model

​		This method is based on the assumption of locality.

​		It defines a parameter $\Delta$  to denote the length of the **working-set window**. The most recent $\Delta$ page references is the **working set**. If $\Delta$ is too small, the working set cannot cover the entire locality; if $\Delta$ is too large, it may overlap localities. Process *i* needs $\Delta_i$  or $WSS_i$ frames, D = $\sum WSS_i$ , if D> m (total #frames), thrashing will occur. OS will choose some processes to suspend.

​		“Working-set strategy prevents thrashing while keeping the degree of multiprogramming as high as possible.”



##### 8.6.3 Page-Fault Frequency

​		Set upper bound and lower bound of **PFF**. If touches upper bound, give new frame; if touch lower bound, remove a frame.



------



#### 8.7 Memory-Mapped Files

#### 8.8 Allocating Kernel Memory

​		Different from user-mode process. For: 1. many data structures less than a page  in size 2. some hardware require contiguous pages.

##### 8.8.1 Buddy System

​		The buddy system allocate memory from a fixed-size segment consisting of physically contiguous pages in **power-of-2 allocator**. Divide until got the suitable size, if need larger: combine with **coalescing**.



##### 8.8.2 Slab Allocation

​		A **slab** is made up of one or more physically contiguous pages. A **cache** consists of one or more slabs. There is a single cache for each unique kernel data structure. Each cache is populated with **objects** that are instantiations of the kernel data structure the cache represents.

​		The slab-allocation algorithm uses cache to store kernel objects. When created, the objects are marked as **free**, assign them and marked as **used**.

​		Benefits of slab allocation:

1. No memory wasted due to external fragmentation. Because the kernel just define the fixed-sized blocks and return address.
2. Memory request quickly. Objects are created in advance and just change the mark when allocating or deallocating.



------



#### 8.9 Other Consideration

##### 8.9.1 Prepaging

​		To reduce page faults when initial the processes, some OS with bring all needed pages in advance. 

​		Suppose *s* pages are prepaged and a fraction of $\alpha$ of these *s* pages is actually used.  We compare $s*\alpha$ and $s*(1-\alpha)$. If $\alpha$ is close to 1, prepaging wins.



##### 8.9.2 Page Size

​		**Large page size** is desirable because we want to reduce the memory occupied by page table for each process. Also for seek time.

​		On the other hand, we want **small page size** to reduce the waste of memory in the final pages.  Also for I/O time.



##### 8.9.3 Inverted Page Table

​		Cause incomplete page information for swap-out pages, need an **external page table**. Only access when page fault, however, may cause **double page fault**.

## IV. Storage Management

### **Chapter 10 File System Implementation**

#### 10.2 File System Implementation

##### 10.2.1 Overview

- **Boot-control block**: contain information needed to boot, called the **boot block** in UFS, **partition boot sector** in NTFS
- **Volume-control block**: contains volume details
- **Directory structure**: 
- **Per-file FCB**:

In -memory structures:

- **Mount table**
- **Directory-structure Cache**
- **system-wide open-file table**
- **Per-process open-file table**





##### 10.4.1 Contiguous Allocation

​		Allocation defined by disk address + length. Sequential and direct access are both provided. However, there are problems: 

1. **Finding free space** for new file is difficult, need to manage free space and suffer from **external fragmentation**. One of the solution is **compaction**. It’s time-consuming, and normally not permitted off-line (unmounted).
2. **Determine space size** for files is also hard. Files sometimes cannot enlarge since the neighboring space is occupied. One solution is to copy the file to another larger space, still time-consuming. 
3. Some OS minimize the drawbacks by adding **extent** - some extra space.

#### 10.5 Free-Space Management

##### 10.5.1 Bit Vector/Map

Free - 1, Allocated - 0





























