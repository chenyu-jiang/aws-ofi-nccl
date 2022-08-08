/*
 * Copyright (c) 2018 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

/*
 * This test validates functionality of NCCL's connection establishment and
 * data transfer APIs
 */
#include <unistd.h>
#include <time.h>
#include "test-common.h"

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


int main(int argc, char* argv[])
{
	int opt;
	int num_requests = 0;
	size_t send_size = 0;
	int dev_id = -1;
	// int is_client = 0;

	while ((opt = getopt(argc, argv, "n:s:d:")) != -1) {
		switch (opt) {
		case 'n':
			num_requests = atoi(optarg);
			break;
		case 's':
			send_size = strtoul(optarg, NULL, 0);
			break;
		case 'd':
			dev_id = atoi(optarg);
			break;
		// case 'c':
		// 	is_client = 1;
		// 	break;
		default:
			fprintf(stderr, "Usage: %s -n <num requests> -s <send size of a single request> -d <device>\n", argv[0]);
			exit(EXIT_FAILURE);
		}
	}
	if (num_requests <= 0) {
		fprintf(stderr, "Invalid number of requests %d.\n", num_requests);
		exit(EXIT_FAILURE);
	}
	if (send_size <= 0) {
		fprintf(stderr, "Invalid send size %zu.\n", send_size);
		exit(EXIT_FAILURE);
	}
	if (dev_id == -1) {
		fprintf(stderr, "Device id uninitialized.\n");
		exit(EXIT_FAILURE);
	}

	size_t recv_size = send_size + 200;

	int rank, proc_name_len, num_ranks, local_rank = 0;
	int buffer_type = NCCL_PTR_HOST;

	/* Plugin defines */
	int ndev, dev, cuda_dev, i;
	sendComm_t *sComm = NULL;
	listenComm_t *lComm = NULL;
	recvComm_t *rComm = NULL;
	ncclNet_t *extNet = NULL;
	char src_handle[NCCL_NET_HANDLE_MAXSIZE] = {0};

	ofi_log_function = logger;

	/* Initialisation for data transfer */
	nccl_ofi_req_t *req[num_requests];
	memset(req, 0, sizeof(nccl_ofi_req_t *) * num_requests);
	void *mhandle[num_requests];
	int req_completed[num_requests];
	memset(req_completed, 0, sizeof(int) * num_requests);
	int inflight_reqs = num_requests;
	char *send_buf[num_requests];
	memset(send_buf, 0, sizeof(char *) * num_requests);
	char *recv_buf[num_requests];
	memset(recv_buf, 0, sizeof(char *) * num_requests);
	int done, received_size, idx;

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

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

	char all_proc_name[num_ranks][MPI_MAX_PROCESSOR_NAME];

	MPI_Get_processor_name(all_proc_name[rank], &proc_name_len);
	MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, all_proc_name,
			MPI_MAX_PROCESSOR_NAME, MPI_BYTE, MPI_COMM_WORLD);

	/* Determine local rank */
	for (i = 0; i < num_ranks; i++) {
		if (!strcmp(all_proc_name[rank], all_proc_name[i])) {
			if (i < rank) {
				++local_rank;
			}
		}
	}

	/* Set CUDA device for subsequent device memory allocation, in case GDR is used */
	int n_local_devs;
	CUDACHECK(cudaGetDeviceCount(&n_local_devs));
	if (dev_id >= n_local_devs) {
		fprintf(stderr, "Invalid device id %d\n", dev_id);
		exit(EXIT_FAILURE);
	}

	cuda_dev = dev_id;
	NCCL_OFI_TRACE(NCCL_NET, "Using CUDA device %d for memory allocation", cuda_dev);
	CUDACHECK(cudaSetDevice(cuda_dev));

	/* Get external Network from NCCL-OFI library */
	extNet = get_extNet();
	if (extNet == NULL)
		return -1;

	/* Init API */
	OFINCCLCHECK(extNet->init(&logger));
	NCCL_OFI_INFO(NCCL_NET, "Process rank %d started. NCCLNet device used on %s is %s.",
			rank, all_proc_name[rank], extNet->name);

	/* Devices API */
	OFINCCLCHECK(extNet->devices(&ndev));
	NCCL_OFI_INFO(NCCL_NET, "Received %d network devices", ndev);

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
		NCCL_OFI_TRACE(NCCL_INIT, "Dev %d has path %s and supports pointers of type %d", dev, path, supported_types);

		/* Set CUDA support */
		support_gdr[dev] = is_gdr_supported_nic(supported_types);
	}
#endif

	/* Choose specific device per rank for communication */
	// dev = rand() % ndev;
	dev = cuda_dev / ((n_local_devs + (ndev - 1)) / ndev);
	NCCL_OFI_TRACE(NCCL_INIT, "Rank %d uses device %d for communication", rank, dev);

	if (support_gdr[dev] == 1) {
		NCCL_OFI_INFO(NCCL_INIT | NCCL_NET,
				"Network supports communication using CUDA buffers. Dev: %d", dev);
		buffer_type = NCCL_PTR_CUDA;
	}

	/* Listen API */
	char handle[NCCL_NET_HANDLE_MAXSIZE];
	NCCL_OFI_INFO(NCCL_NET, "Server: Listening on dev %d", dev);
	OFINCCLCHECK(extNet->listen(dev, (void *)&handle, (void **)&lComm));

	struct timespec start, end;

	/* Allocate and populate expected buffer */
	char *expected_buf = NULL;
	OFINCCLCHECK(allocate_buff((void **)&expected_buf, send_size, NCCL_PTR_HOST));
	OFINCCLCHECK(initialize_buff((void *)expected_buf, send_size, NCCL_PTR_HOST));

	if (rank == 0) {

		/* MPI send */
		MPI_Send(&handle, NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, (rank + 1), 0, MPI_COMM_WORLD);

		/* MPI recv */
		MPI_Recv((void *)src_handle, NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR,
				(rank + 1), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		/* Connect API */
		NCCL_OFI_INFO(NCCL_NET, "Send connection request to rank %d", rank + 1);
		while (sComm == NULL) {
			OFINCCLCHECK(extNet->connect(dev, (void *)src_handle, (void **)&sComm));
		}

		/* Accept API */
		NCCL_OFI_INFO(NCCL_NET, "Server: Start accepting requests");
		while (rComm == NULL) {
			OFINCCLCHECK(extNet->accept((void *)lComm, (void **)&rComm));
		}
		NCCL_OFI_INFO(NCCL_NET, "Successfully accepted connection from rank %d",
				rank + 1);

		/* Send num_requests to Rank 1 */
		NCCL_OFI_INFO(NCCL_NET, "Send %d requests to rank %d", num_requests,
				rank + 1);
		for (idx = 0; idx < num_requests; idx++) {
			OFINCCLCHECK(allocate_buff((void **)&send_buf[idx], send_size, buffer_type));
			OFINCCLCHECK(initialize_buff((void *)send_buf[idx], send_size, buffer_type));

			OFINCCLCHECK(extNet->regMr((void *)sComm, (void *)send_buf[idx], send_size,
						buffer_type, &mhandle[idx]));
			NCCL_OFI_TRACE(NCCL_NET,
					"Successfully registered send memory for request %d of rank %d",
					idx, rank);
		}
		// start timer
		clock_gettime(CLOCK_MONOTONIC, &start);
		for (idx = 0; idx < num_requests; idx++) {
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
		NCCL_OFI_INFO(NCCL_NET, "Successfully posted %d send requests to rank %d", num_requests,
				rank + 1);
	}
	else if (rank == 1) {

		/* MPI recv */
		MPI_Recv((void *)src_handle, NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, (rank - 1), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		/* MPI send */
		MPI_Send((void *)handle, NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, (rank - 1), 0, MPI_COMM_WORLD);

		/* Connect API */
		NCCL_OFI_INFO(NCCL_NET, "Send connection request to rank %d", rank - 1);
		while (sComm == NULL) {
			OFINCCLCHECK(extNet->connect(dev, (void *)src_handle, (void **)&sComm));
		}

		/* Accept API */
		NCCL_OFI_INFO(NCCL_NET, "Server: Start accepting requests");
		while (rComm == NULL) {
			OFINCCLCHECK(extNet->accept((void *)lComm, (void **)&rComm));
		}
		NCCL_OFI_INFO(NCCL_NET, "Successfully accepted connection from rank %d",
				rank - 1);

		/* Receive num_requests from Rank 0 */
		NCCL_OFI_INFO(NCCL_NET, "Rank %d posting %d receive buffers", rank,
				num_requests);
		for (idx = 0; idx < num_requests; idx++) {
			OFINCCLCHECK(allocate_buff((void **)&recv_buf[idx], recv_size, buffer_type));
			OFINCCLCHECK(extNet->regMr((void *)rComm, (void *)recv_buf[idx], recv_size,
						buffer_type, &mhandle[idx]));
			NCCL_OFI_TRACE(NCCL_NET, "Successfully registered receive memory for request %d of rank %d", idx, rank);
		}
		// start timer
		clock_gettime(CLOCK_MONOTONIC, &start);
		for (idx = 0; idx < num_requests; idx++) {
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
		NCCL_OFI_INFO(NCCL_NET, "Successfully posted %d recv requests from rank %d", num_requests,
		rank - 1);
	}

	/* Test for completions */
	while (true) {
		for (idx = 0; idx < num_requests; idx++) {
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
	// stop timer. don't time data flush for now
	clock_gettime(CLOCK_MONOTONIC, &end);

	for(idx = 0; idx < num_requests; idx++) {
		if ((rank == 1) && (buffer_type == NCCL_PTR_CUDA)) {
			NCCL_OFI_TRACE(NCCL_NET,
					"Issue flush for data consistency. Request idx: %d",
					idx);
#if (NCCL_VERSION_CODE >= NCCL_VERSION(2, 8, 0)) /* Support NCCL v2.8 */
			nccl_ofi_req_t *iflush_req = NULL;
#if (NCCL_VERSION_CODE >= NCCL_VERSION(2, 12, 0)) /* Support NCCL v2.12 */
			OFINCCLCHECK(extNet->iflush((void *)rComm, nrecv,
						(void **)&recv_buf[idx],
						sizes, &mhandle[idx], (void **)&iflush_req));
#else
			OFINCCLCHECK(extNet->iflush((void *)rComm,
						(void **)recv_buf[idx],
						recv_size, mhandle[idx], (void **)&iflush_req));
#endif
			done = 0;
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

		/* Deregister memory handle */
		if (rank == 0) {
			OFINCCLCHECK(extNet->deregMr((void *)sComm, mhandle[idx]));
		}
		else if (rank == 1) {
			if ((buffer_type == NCCL_PTR_CUDA) && !ofi_nccl_gdr_flush_disable()) {
				/* Data validation may fail if flush operations are disabled */
			} else
				OFINCCLCHECK(validate_data(recv_buf[idx], expected_buf, send_size, buffer_type));

			OFINCCLCHECK(extNet->deregMr((void *)rComm, mhandle[idx]));
		}
	}
	NCCL_OFI_INFO(NCCL_NET, "Got completions for %d requests for rank %d",
			num_requests, rank);

	/* Deallocate buffers */
	OFINCCLCHECK(deallocate_buffer(expected_buf, NCCL_PTR_HOST));
	for (idx = 0; idx < num_requests; idx++) {
		if (send_buf[idx])
			OFINCCLCHECK(deallocate_buffer(send_buf[idx], buffer_type));
		if (recv_buf[idx])
			OFINCCLCHECK(deallocate_buffer(recv_buf[idx], buffer_type));
	}

	// calculate time
	double time_elapsed = diff_microseconds(start, end);
	NCCL_OFI_INFO(NCCL_INIT, "Rank %d: Time elapsed: %f microseconds, bandwidth: %f Gbps.", rank, time_elapsed, send_size * num_requests * 8 / time_elapsed / 1e3);

	OFINCCLCHECK(extNet->closeListen((void *)lComm));
	OFINCCLCHECK(extNet->closeSend((void *)sComm));
	OFINCCLCHECK(extNet->closeRecv((void *)rComm));

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();

	return 0;
}
