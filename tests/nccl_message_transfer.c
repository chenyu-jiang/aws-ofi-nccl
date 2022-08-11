/*
 * Copyright (c) 2018 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

/*
 * This test validates functionality of NCCL's connection establishment and
 * data transfer APIs
 */
#include <unistd.h>
#include <time.h>
#include <limits.h>
#include "test-common.h"

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define GETBW(s, t, n) ((s) * (n) * 8 / (t) / 1e3)

double diff_microseconds(struct timespec start, struct timespec end)
{
	double microseconds;
	if ((end.tv_nsec-start.tv_nsec)<0) {
		microseconds = (end.tv_sec-start.tv_sec-1) * 1e6;
		microseconds += (1e9 + end.tv_nsec-start.tv_nsec) / 1e3;
	} else {
		microseconds = (end.tv_sec-start.tv_sec) * 1e6;
		microseconds += (end.tv_nsec-start.tv_nsec) / 1e3;
	}
	return microseconds;
}

#define MAX_LINE_SIZE 1024
#define MAX_N_LINES 1048576

void reset_buffers(nccl_ofi_req_t ** req, int* req_completed, int num_requests) {
	memset(req, 0, sizeof(nccl_ofi_req_t *) * num_requests);
	memset(req_completed, 0, sizeof(int) * num_requests);
}

ncclResult_t alloc_and_reg_buffers(ncclNet_t *extNet, int num_requests, size_t buf_size, int buffer_type, 
	ofiComm_t *comm, void** mhandle, char** buf) {
	for (int idx = 0; idx < num_requests; idx++) {
		OFINCCLCHECK(allocate_buff((void **)&buf[idx], buf_size, buffer_type));
		OFINCCLCHECK(extNet->regMr((void *)comm, (void *)buf[idx], buf_size,
					buffer_type, &mhandle[idx]));
	}
	return ncclSuccess;
}

ncclResult_t test_for_completion(ncclNet_t* extNet, int num_requests, int* req_completed, nccl_ofi_req_t ** req) {
	/* Test for completions */
	int done, received_size;
	int inflight_reqs = num_requests;
	while (true) {
		for (int idx = 0; idx < num_requests; idx++) {
			if (req_completed[idx])
				continue;

			OFINCCLCHECK(extNet->test((void *)req[idx], &done, &received_size));
			if (done) {
				inflight_reqs--;
				req_completed[idx] = 1;
			}
		}

		if (inflight_reqs == 0)
			break;
	}
	return ncclSuccess;
}

int main(int argc, char* argv[])
{
	int rank, proc_name_len, num_ranks, local_rank = 0;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

	char all_proc_name[num_ranks][MPI_MAX_PROCESSOR_NAME];

	MPI_Get_processor_name(all_proc_name[rank], &proc_name_len);
	MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, all_proc_name,
			MPI_MAX_PROCESSOR_NAME, MPI_BYTE, MPI_COMM_WORLD);

	/* Determine local rank */
	for (int i = 0; i < num_ranks; i++) {
		if (!strcmp(all_proc_name[rank], all_proc_name[i])) {
			if (i < rank) {
				++local_rank;
			}
		}
	}

	int num_requests = 0;
	int num_iters = 0;
	int num_warmup_iters = 0;
	size_t send_size = 0;
	int dev_id = -1;
	int remote_rank = -1;
	int is_client = 0;

	// read parameters from file instread of using args
	if(argc != 2) {
		printf("Usage: %s ARGS_FILE\n", argv[0]);
		printf("ARGS_FILE format: one arg per line. Empty line separates ranks.\n");
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
		return EXIT_FAILURE;
	}

	char* args_file = argv[1];
	FILE *fp;
	fp = fopen(args_file,"r");
	if (!fp) {
		fprintf(stderr, "Failed to open file %s.\n", args_file);
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
		return EXIT_FAILURE;
	}
	size_t line_size = MAX_LINE_SIZE;
	char* line_buff = malloc(MAX_LINE_SIZE);
	int curr_rank = 0;
	int nchar = 0;
	char* rank_argv[MAX_N_LINES];
	rank_argv[0] = argv[0];
	int rank_argc = 1;
	while((nchar = getline(&line_buff, &line_size, fp)) != -1) {
		if (nchar == 1) {
			// empty line, new rank
			curr_rank ++;
			continue;
		}
		if (curr_rank == rank) {
			// add arg to current rank
			rank_argv[rank_argc] = malloc(nchar);
			memcpy(rank_argv[rank_argc],line_buff,nchar-1);
			rank_argv[rank_argc][nchar-1] = '\0';
			rank_argc++;
		}
	}
	if (curr_rank != num_ranks) {
		fprintf(stderr, "[Rank %d] Number of ranks in ARGS_FILE is not equal to world size.", rank);
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
		return EXIT_FAILURE;
	}
	free(line_buff);
	fclose(fp);

	if (rank_argc != 8) {
		fprintf(stderr, "[Rank %d] Invalid number of arguments. Expected 7, got %d\n", rank, rank_argc - 1);
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	}
	printf("[Rank %d] Read args file complete. argc: %d\n", rank, rank_argc);

	num_requests = atoi(rank_argv[1]);
	num_iters = atoi(rank_argv[2]);
	num_warmup_iters = atoi(rank_argv[3]);
	send_size = strtoul(rank_argv[4], NULL, 0);
	dev_id = atoi(rank_argv[5]);
	remote_rank = atoi(rank_argv[6]);
	is_client = atoi(rank_argv[7]);

	if (num_requests <= 0) {
		fprintf(stderr, "[Rank %d] Invalid number of requests %d.\n", rank, num_requests);
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
		return EXIT_FAILURE;
	}
	if (num_iters <= 0) {
		fprintf(stderr, "[Rank %d] Invalid number of iterations %d.\n", rank, num_requests);
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
		return EXIT_FAILURE;
	}
	if (send_size == 0 || send_size == ULONG_MAX) {
		fprintf(stderr, "[Rank %d] Invalid send size %zu.\n", rank, send_size);
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
		return EXIT_FAILURE;
	}
	if (dev_id == -1) {
		fprintf(stderr, "[Rank %d] Device id uninitialized.\n", rank);
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
		return EXIT_FAILURE;
	}
	if (remote_rank == -1) {
		fprintf(stderr, "[Rank %d] Remote rank uninitialized.\n", rank);
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
		return EXIT_FAILURE;
	}

	size_t recv_size = send_size + 200;
	int buffer_type = NCCL_PTR_HOST;

	/* Plugin defines */
	int ndev, dev, cuda_dev;
	sendComm_t *sComm = NULL;
	listenComm_t *lComm = NULL;
	recvComm_t *rComm = NULL;
	ncclNet_t *extNet = NULL;
	char src_handle[NCCL_NET_HANDLE_MAXSIZE] = {0};

	ofi_log_function = logger;

	NCCL_OFI_INFO(NCCL_INIT, "[Rank %d] num_requests: %d, num_iters: %d, num_warmup_iters: %d, send_size: %d, dev_id: %d, remote_rank: %d, is_client: %d", 
						rank, num_requests, num_iters, num_warmup_iters, send_size, dev_id, remote_rank, is_client);

	/* Initialisation for data transfer */
	nccl_ofi_req_t *req[num_requests];
	void *mhandle[num_requests];
	int req_completed[num_requests];
	char *send_buf[num_requests];
	memset(send_buf, 0, sizeof(char *) * num_requests);
	char *recv_buf[num_requests];
	memset(recv_buf, 0, sizeof(char *) * num_requests);

#if (NCCL_VERSION_CODE >= NCCL_VERSION(2, 12, 0))
	/* For grouped recvs */
	int tag = 1;
	int nrecv = NCCL_OFI_MAX_RECVS;
	int *sizes = (int *)malloc(sizeof(int)*nrecv);
	int *tags = (int *)malloc(sizeof(int)*nrecv);
	for (int recv_n = 0; recv_n < nrecv; recv_n++) {
		sizes[recv_n] = recv_size;
		tags[recv_n] = tag;
	}
#endif

	/* Set CUDA device for subsequent device memory allocation, in case GDR is used */
	int n_local_devs;
	CUDACHECK(cudaGetDeviceCount(&n_local_devs));
	if (dev_id >= n_local_devs) {
		fprintf(stderr, "Invalid device id %d\n", dev_id);
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
		return EXIT_FAILURE;
	}

	cuda_dev = dev_id;
	NCCL_OFI_TRACE(NCCL_NET, "[Rank %d] Using CUDA device %d for memory allocation", rank, cuda_dev);
	CUDACHECK(cudaSetDevice(cuda_dev));

	/* Get external Network from NCCL-OFI library */
	extNet = get_extNet();
	if (extNet == NULL) {
		fprintf(stderr, "[Rank %d] Failed to get extNet from NCCL-OFI library.\n", rank);
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
		return EXIT_FAILURE;
	}

	/* Init API */
	OFINCCLCHECK(extNet->init(&logger));
	NCCL_OFI_INFO(NCCL_NET, "[Rank %d] Process rank started. NCCLNet device used on %s is %s.",
			rank, all_proc_name[rank], extNet->name);

	/* Devices API */
	OFINCCLCHECK(extNet->devices(&ndev));
	NCCL_OFI_INFO(NCCL_NET, "[Rank %d] Received %d network devices", rank, ndev);

	/* Indicates if NICs support GPUDirect */
	int support_gdr[ndev];

#if (NCCL_VERSION_CODE >= NCCL_VERSION(2, 6, 4))
	/* Get Properties for the device */
	for (dev = 0; dev < ndev; dev++) {
		ncclNetProperties_t props = {0};
		OFINCCLCHECK(extNet->getProperties(dev, &props));
		print_dev_props(dev, &props);

		/* Set CUDA support */
		support_gdr[dev] = is_gdr_supported_nic(props.ptrSupport);
	}
#else
	/* Get PCIe path and plugin memory pointer support */
	for (dev = 0; dev < ndev; dev++) {
		char *path = NULL;
		int supported_types = 0;
		extNet->pciPath(dev, &path);
		OFINCCLCHECK(extNet->ptrSupport(dev, &supported_types));
		NCCL_OFI_TRACE(NCCL_INIT, "[Rank %d] Dev %d has path %s and supports pointers of type %d", rank, dev, path, supported_types);

		/* Set CUDA support */
		support_gdr[dev] = is_gdr_supported_nic(supported_types);
	}
#endif

	/* Choose specific device per rank for communication */
	// dev = rand() % ndev;
	dev = cuda_dev / ((n_local_devs + (ndev - 1)) / ndev);
	NCCL_OFI_TRACE(NCCL_INIT, "[Rank %d] Uses device %d for communication", rank, dev);

	if (support_gdr[dev] == 1) {
		NCCL_OFI_INFO(NCCL_INIT | NCCL_NET,
				"[Rank %d] Network supports communication using CUDA buffers. Dev: %d", rank, dev);
		buffer_type = NCCL_PTR_CUDA;
	}

	/* Listen API */
	char handle[NCCL_NET_HANDLE_MAXSIZE];
	NCCL_OFI_INFO(NCCL_NET, "[Rank %d] Server: Listening on dev %d", rank, dev);
	OFINCCLCHECK(extNet->listen(dev, (void *)&handle, (void **)&lComm));

	struct timespec start, end;
	struct timespec group_end;

	// measure the performance of MPI barrier
	MPI_Barrier(MPI_COMM_WORLD); // sync first

	clock_gettime(CLOCK_MONOTONIC, &start);
	for(int i=0; i < 20; i++) {
		MPI_Barrier(MPI_COMM_WORLD);
	}
	clock_gettime(CLOCK_MONOTONIC, &end);
	double barrier_time = diff_microseconds(start, end) / 20;
	double global_barrier_time = 0;
	MPI_Allreduce(&barrier_time, &global_barrier_time, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	global_barrier_time /= num_ranks;
	if(rank == 0) {
		NCCL_OFI_INFO(NCCL_NET, "[Rank %d] Global barrier time: %f microseconds.", rank, global_barrier_time);
	}

	/* Allocate expected buffer */
	char *expected_buf = NULL;
	OFINCCLCHECK(allocate_buff((void **)&expected_buf, send_size, NCCL_PTR_HOST));

	double time_per_iter[num_iters];
	double time_per_iter_group[num_iters];

	if (is_client) {

		/* MPI send */
		MPI_Send(&handle, NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, remote_rank, 0, MPI_COMM_WORLD);

		/* MPI recv */
		MPI_Recv((void *)src_handle, NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR,
				remote_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		/* Connect API */
		NCCL_OFI_INFO(NCCL_NET, "[Rank %d] Send connection request to rank %d", rank, remote_rank);
		while (sComm == NULL) {
			OFINCCLCHECK(extNet->connect(dev, (void *)src_handle, (void **)&sComm));
		}

		/* Accept API */
		NCCL_OFI_INFO(NCCL_NET, "[Rank %d] Server: Start accepting requests", rank);
		while (rComm == NULL) {
			OFINCCLCHECK(extNet->accept((void *)lComm, (void **)&rComm));
		}
		NCCL_OFI_INFO(NCCL_NET, "[Rank %d] Successfully accepted connection from rank %d",
				rank, remote_rank);

		NCCL_OFI_INFO(NCCL_NET, "[Rank %d] Send %d requests to rank %d", rank, num_requests,
					remote_rank);

		// allocate send buffer and reg send memory regions
		OFINCCLCHECK(alloc_and_reg_buffers(extNet, num_requests, send_size, buffer_type, sComm, mhandle, send_buf));
		NCCL_OFI_INFO(NCCL_NET, "[Rank %d] Successfully registered send memory for %d requests", rank, num_requests);

		for(int iter = 0; iter < num_iters + num_warmup_iters; iter ++) {
			// reset req and req_completed buffer
			reset_buffers(req, req_completed, num_requests);
			// populate expected buffer
			OFINCCLCHECK(initialize_buff((void *)expected_buf, send_size, NCCL_PTR_HOST));
			// populate send buffer
			for(int idx = 0; idx < num_requests; idx++) {
				OFINCCLCHECK(initialize_buff((void *)send_buf[idx], send_size, buffer_type));
			}
			// barrier
			MPI_Barrier(MPI_COMM_WORLD);
			// start timer
			clock_gettime(CLOCK_MONOTONIC, &start);
			for (int idx = 0; idx < num_requests; idx++) {
#if (NCCL_VERSION_CODE >= NCCL_VERSION(2, 12, 0)) /* Support NCCL v2.12 */
				while (req[idx] == NULL) {
					OFINCCLCHECK(extNet->isend((void *)sComm, (void *)send_buf[idx], send_size, tag,
								mhandle[idx], (void **)&req[idx]));
				}
#else
				while (req[idx] == NULL) {
					OFINCCLCHECK(extNet->isend((void *)sComm, (void *)send_buf[idx], send_size,
								mhandle[idx], (void **)&req[idx]));
				}
#endif
			}
#if OFI_NCCL_TRACE
			NCCL_OFI_TRACE(NCCL_NET, "[Rank %d] Successfully posted %d send requests to rank %d", rank, num_requests, remote_rank);
#endif
			OFINCCLCHECK(test_for_completion(extNet, num_requests, req_completed, req));
			// stop timer
			clock_gettime(CLOCK_MONOTONIC, &end);
			MPI_Barrier(MPI_COMM_WORLD);
			clock_gettime(CLOCK_MONOTONIC, &group_end);
			NCCL_OFI_TRACE(NCCL_NET, "[Rank %d] Got completions for %d requests.", rank, num_requests);
			if(iter >= num_warmup_iters) {
				// calculate time
				double time_elapsed = diff_microseconds(start, end);
				double group_time_elapsed = diff_microseconds(start, group_end);
				time_per_iter[iter - num_warmup_iters] = time_elapsed;
				time_per_iter_group[iter - num_warmup_iters] = group_time_elapsed - global_barrier_time;
			}
		}
	} else {

		/* MPI recv */
		MPI_Recv((void *)src_handle, NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, remote_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		/* MPI send */
		MPI_Send((void *)handle, NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, remote_rank, 0, MPI_COMM_WORLD);

		/* Connect API */
		NCCL_OFI_INFO(NCCL_NET, "[Rank %d] Send connection request to rank %d", rank, remote_rank);
		while (sComm == NULL) {
			OFINCCLCHECK(extNet->connect(dev, (void *)src_handle, (void **)&sComm));
		}

		/* Accept API */
		NCCL_OFI_INFO(NCCL_NET, "[Rank %d] Server: Start accepting requests", rank);
		while (rComm == NULL) {
			OFINCCLCHECK(extNet->accept((void *)lComm, (void **)&rComm));
		}
		NCCL_OFI_INFO(NCCL_NET, "[Rank %d] Successfully accepted connection from rank %d",
				rank, remote_rank);

		// allocate and register recv memory regions
		OFINCCLCHECK(alloc_and_reg_buffers(extNet, num_requests, recv_size, buffer_type, rComm, mhandle, recv_buf));
		NCCL_OFI_INFO(NCCL_NET, "[Rank %d] Successfully registered receive memory for %d requests", rank, num_requests);

		for(int iter = 0; iter < num_iters + num_warmup_iters; iter ++) {
			// reset req and req_completed buffer
			reset_buffers(req, req_completed, num_requests);
			// populate expected buffer
			OFINCCLCHECK(initialize_buff((void *)expected_buf, send_size, NCCL_PTR_HOST));
			// populate recv buffer
			for(int idx = 0; idx < num_requests; idx++) {
				OFINCCLCHECK(initialize_buff((void *)recv_buf[idx], recv_size, buffer_type));
			}
			// barrier
			MPI_Barrier(MPI_COMM_WORLD);
			// start timer
			clock_gettime(CLOCK_MONOTONIC, &start);
			for (int idx = 0; idx < num_requests; idx++) {
#if (NCCL_VERSION_CODE >= NCCL_VERSION(2, 12, 0)) /* Support NCCL v2.12 */
				while (req[idx] == NULL) {
					OFINCCLCHECK(extNet->irecv((void *)rComm, nrecv, (void *)&recv_buf[idx],
								sizes, tags, &mhandle[idx], (void **)&req[idx]));
				}
#else
				while (req[idx] == NULL) {
					OFINCCLCHECK(extNet->irecv((void *)rComm, (void *)recv_buf[idx],
								recv_size, mhandle[idx], (void **)&req[idx]));
				}
#endif
			}
#if OFI_NCCL_TRACE
			NCCL_OFI_TRACE(NCCL_NET, "[Rank %d] Successfully posted %d recv requests from rank %d", rank, num_requests, remote_rank);
#endif
			OFINCCLCHECK(test_for_completion(extNet, num_requests, req_completed, req));
			// stop timer. don't time data flush for now
			clock_gettime(CLOCK_MONOTONIC, &end);
			MPI_Barrier(MPI_COMM_WORLD);
			clock_gettime(CLOCK_MONOTONIC, &group_end);
			NCCL_OFI_TRACE(NCCL_NET, "[Rank %d] Got completions for %d requests.", rank, num_requests);
			for(int idx = 0; idx < num_requests; idx++) {
				if (buffer_type == NCCL_PTR_CUDA) {
					NCCL_OFI_TRACE(NCCL_NET,
							"[Rank %d] Issue flush for data consistency. Request idx: %d", rank, idx);
#if (NCCL_VERSION_CODE >= NCCL_VERSION(2, 8, 0)) /* Support NCCL v2.8 */
					nccl_ofi_req_t *iflush_req = NULL;
#if (NCCL_VERSION_CODE >= NCCL_VERSION(2, 12, 0)) /* Support NCCL v2.12 */
					OFINCCLCHECK(extNet->iflush((void *)rComm, nrecv,
								(void **)&recv_buf[idx],
								sizes, &mhandle[idx], (void **)&iflush_req));
#else
					OFINCCLCHECK(extNet->iflush((void *)rComm,
								(void *)recv_buf[idx],
								recv_size, mhandle[idx], (void **)&iflush_req));
#endif
					int done = 0;
					if (iflush_req) {
						while (!done) {
							OFINCCLCHECK(extNet->test((void *)iflush_req, &done, NULL));
						}
					}
#else
					OFINCCLCHECK(extNet->flush((void *)rComm,
								(void *)recv_buf[idx],
								recv_size, mhandle[idx]));
#endif
				}
				// check recv buffer validity
				if ((buffer_type == NCCL_PTR_CUDA) && !ofi_nccl_gdr_flush_disable()) {
					/* Data validation may fail if flush operations are disabled */
				} else {
					OFINCCLCHECK(validate_data(recv_buf[idx], expected_buf, send_size, buffer_type))
				};
			}
			if(iter >= num_warmup_iters) {
				// calculate time
				double time_elapsed = diff_microseconds(start, end);
				double group_time_elapsed = diff_microseconds(start, group_end);
				time_per_iter[iter - num_warmup_iters] = time_elapsed;
				time_per_iter_group[iter - num_warmup_iters] = group_time_elapsed - global_barrier_time;
			}
		}
	}

	/* Deregister memory handle */
	for(int idx = 0; idx < num_requests; idx++) {
		if (is_client) {
			OFINCCLCHECK(extNet->deregMr((void *)sComm, mhandle[idx]));
		} else {
			OFINCCLCHECK(extNet->deregMr((void *)rComm, mhandle[idx]));
		}
	}

	/* Deallocate buffers */
	OFINCCLCHECK(deallocate_buffer(expected_buf, NCCL_PTR_HOST));
	for (int idx = 0; idx < num_requests; idx++) {
		if (send_buf[idx])
			OFINCCLCHECK(deallocate_buffer(send_buf[idx], buffer_type));
		if (recv_buf[idx])
			OFINCCLCHECK(deallocate_buffer(recv_buf[idx], buffer_type));
	}

	// synchronize time between all ranks, and between send ranks and recv ranks
	MPI_Comm subcomm;
	MPI_Comm_split(MPI_COMM_WORLD, is_client, rank, &subcomm);
	int subcomm_rank = -1;
	int subcomm_nranks = -1;
	MPI_Comm_rank(subcomm, &subcomm_rank);
	MPI_Comm_size(subcomm, &subcomm_nranks);

	for (int comm_idx = 0; comm_idx < 2; comm_idx ++) {
		MPI_Comm comm = (comm_idx == 0) ? subcomm : MPI_COMM_WORLD;
		int comm_rank = (comm_idx == 0) ? subcomm_rank : rank;
		int comm_num_ranks = (comm_idx == 0) ? subcomm_nranks : num_ranks;

		double local_min = time_per_iter[0];
		double local_min_group = time_per_iter_group[0];
		double local_max = time_per_iter[0];
		double local_max_group = time_per_iter_group[0];
		double local_sum = 0;
		double local_sum_group = 0;
		for(int i=0; i< num_iters; i++) {
			local_min = MIN(local_min, time_per_iter[i]);
			local_min_group = MIN(local_min_group, time_per_iter_group[i]);
			local_max = MAX(local_max, time_per_iter[i]);
			local_max_group = MAX(local_max_group, time_per_iter_group[i]);
			local_sum += time_per_iter[i];
			local_sum_group += time_per_iter_group[i];
		}
		double global_min, global_min_group, global_max, global_max_group, global_sum, global_sum_group;
		MPI_Reduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
		MPI_Reduce(&local_min_group, &global_min_group, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
		MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
		MPI_Reduce(&local_max_group, &global_max_group, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
		MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
		MPI_Reduce(&local_sum_group, &global_sum_group, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

		double global_max_per_iter[num_iters];
		double global_max_per_iter_group[num_iters];
		double global_min_per_iter[num_iters];
		double global_min_per_iter_group[num_iters];
		double global_sum_per_iter[num_iters];
		double global_sum_per_iter_group[num_iters];
		MPI_Reduce(time_per_iter, global_max_per_iter, num_iters, MPI_DOUBLE, MPI_MAX, 0, comm);
		MPI_Reduce(time_per_iter_group, global_max_per_iter_group, num_iters, MPI_DOUBLE, MPI_MAX, 0, comm);
		MPI_Reduce(time_per_iter, global_min_per_iter, num_iters, MPI_DOUBLE, MPI_MIN, 0, comm);
		MPI_Reduce(time_per_iter_group, global_min_per_iter_group, num_iters, MPI_DOUBLE, MPI_MIN, 0, comm);
		MPI_Reduce(time_per_iter, global_sum_per_iter, num_iters, MPI_DOUBLE, MPI_SUM, 0, comm);
		MPI_Reduce(time_per_iter_group, global_sum_per_iter_group, num_iters, MPI_DOUBLE, MPI_SUM, 0, comm);


		for(int print_idx = 0; print_idx < 3; print_idx ++) {
			MPI_Barrier(MPI_COMM_WORLD);
			if(((comm_idx == 0 && ((print_idx == 0 && is_client) || (print_idx == 1 && !is_client))) || (comm_idx == 1 && print_idx == 2)) && comm_rank == 0) {
				double global_avg = global_sum / (double)(comm_num_ranks * num_iters);
				double global_avg_group = global_sum_group / (double)(comm_num_ranks * num_iters);
				double global_max_per_iter_avg = 0;
				double global_max_per_iter_avg_group = 0;
				double global_min_per_iter_avg = 0;
				double global_min_per_iter_avg_group = 0;
				double global_avg_per_iter_min = global_sum_per_iter[0] / comm_num_ranks;
				double global_avg_per_iter_min_group = global_sum_per_iter_group[0] / comm_num_ranks;
				double global_avg_per_iter_max = global_sum_per_iter[0] / comm_num_ranks;
				double global_avg_per_iter_max_group = global_sum_per_iter_group[0] / comm_num_ranks;
				for(int i=0; i< num_iters; i++) {
					global_max_per_iter_avg += global_max_per_iter[i];
					global_max_per_iter_avg_group += global_max_per_iter_group[i];
					global_min_per_iter_avg += global_min_per_iter[i];
					global_min_per_iter_avg_group += global_min_per_iter_group[i];
					global_avg_per_iter_min = MIN(global_avg_per_iter_min, global_sum_per_iter[i] / comm_num_ranks);
					global_avg_per_iter_min_group = MIN(global_avg_per_iter_min_group, global_sum_per_iter_group[i] / comm_num_ranks);
					global_avg_per_iter_max = MAX(global_avg_per_iter_max, global_sum_per_iter[i] / comm_num_ranks);
					global_avg_per_iter_max_group = MAX(global_avg_per_iter_max_group, global_sum_per_iter_group[i] / comm_num_ranks);
				}
				global_max_per_iter_avg /= (double)num_iters;
				global_max_per_iter_avg_group /= (double)num_iters;
				global_min_per_iter_avg /= (double)num_iters;
				global_min_per_iter_avg_group /= (double)num_iters;

				if (comm_idx == 0) {
					if (is_client) {
						printf("[Rank %d] Time elapsed for send:\n", rank);
					} else {
						printf("[Rank %d] Time elapsed for recv:\n", rank);
					}
				} else {
					printf("[Rank %d] Time elapsed for all:\n", rank);
				}
				printf("[Rank %d]     MPI_Barrier cost: %f microseconds.\n", rank, global_barrier_time);
				printf("[Rank %d]     Min: %f microseconds, equivalent bandwidth: %f Gbps per session.\n", rank, global_min, GETBW(send_size, global_min, num_requests));
				printf("[Rank %d]     Group Min: %f microseconds, equivalent bandwidth: %f Gbps per session.\n", rank, global_min_group, GETBW(send_size, global_min_group, num_requests));
				printf("[Rank %d]     Max: %f microseconds, equivalent bandwidth: %f Gbps per session.\n", rank, global_max, GETBW(send_size, global_max, num_requests));
				printf("[Rank %d]     Group Max: %f microseconds, equivalent bandwidth: %f Gbps per session.\n", rank, global_max_group, GETBW(send_size, global_max_group, num_requests));
				printf("[Rank %d]     Avg MaxPerIter: %f microseconds, equivalent bandwidth: %f Gbps per session.\n", rank, global_max_per_iter_avg, GETBW(send_size, global_max_per_iter_avg, num_requests));
				printf("[Rank %d]     Group Avg MaxPerIter: %f microseconds, equivalent bandwidth: %f Gbps per session.\n", rank, global_max_per_iter_avg_group, GETBW(send_size, global_max_per_iter_avg_group, num_requests));
				printf("[Rank %d]     Avg MinPerIter: %f microseconds, equivalent bandwidth: %f Gbps per session.\n", rank, global_min_per_iter_avg, GETBW(send_size, global_min_per_iter_avg, num_requests));
				printf("[Rank %d]     Group Avg MinPerIter: %f microseconds, equivalent bandwidth: %f Gbps per session.\n", rank, global_min_per_iter_avg_group, GETBW(send_size, global_min_per_iter_avg_group, num_requests));
				printf("[Rank %d]     Max AveragePerIter: %f microseconds, equivalent bandwidth: %f Gbps per session.\n", rank, global_avg_per_iter_max, GETBW(send_size, global_avg_per_iter_max, num_requests));
				printf("[Rank %d]     Group Max AveragePerIter: %f microseconds, equivalent bandwidth: %f Gbps per session.\n", rank, global_avg_per_iter_max_group, GETBW(send_size, global_avg_per_iter_max_group, num_requests));
				printf("[Rank %d]     Min AveragePerIter: %f microseconds, equivalent bandwidth: %f Gbps per session.\n", rank, global_avg_per_iter_min, GETBW(send_size, global_avg_per_iter_min, num_requests));
				printf("[Rank %d]     Group Min AveragePerIter: %f microseconds, equivalent bandwidth: %f Gbps per session.\n", rank, global_avg_per_iter_min_group, GETBW(send_size, global_avg_per_iter_min_group, num_requests));
				printf("[Rank %d]     Avg: %f microseconds, equivalent bandwidth: %f Gbps per session.\n", rank, global_avg, GETBW(send_size, global_avg, num_requests));
				printf("[Rank %d]     Group Avg: %f microseconds, equivalent bandwidth: %f Gbps per session.\n", rank, global_avg_group, GETBW(send_size, global_avg_group, num_requests));
				printf("\n");
			}
			MPI_Barrier(MPI_COMM_WORLD);
		}

		MPI_Barrier(MPI_COMM_WORLD);
	}


	OFINCCLCHECK(extNet->closeListen((void *)lComm));
	OFINCCLCHECK(extNet->closeSend((void *)sComm));
	OFINCCLCHECK(extNet->closeRecv((void *)rComm));

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();

	return 0;
}
