/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "mpiimpl.h"

/* Algorithm: Rabenseifner's Algorithm
 *
 * Restrictions: Built-in ops only
 *
 * This algorithm is from http://www.hlrs.de/mpi/myreduce.html.
.
 * This algorithm implements the allreduce in two steps: first a
 * reduce-scatter, followed by an allgather. A recursive-halving algorithm
 * (beginning with processes that are distance 1 apart) is used for the
 * reduce-scatter, and a recursive doubling algorithm is used for the
 * allgather. The non-power-of-two case is handled by dropping to the nearest
 * lower power-of-two: the first few even-numbered processes send their data to
 * their right neighbors (rank+1), and the reduce-scatter and allgather happen
 * among the remaining power-of-two processes. At the end, the first few
 * even-numbered processes get the result from their right neighbors.
 *
 * For the power-of-two case, the cost for the reduce-scatter is:
 *
 * lgp.alpha + n.((p-1)/p).beta + n.((p-1)/p).gamma.
 *
 * The cost for the allgather:
 *
 * lgp.alpha +.n.((p-1)/p).beta
 *
 * Therefore, the total cost is:
 *
 * Cost = 2.lgp.alpha + 2.n.((p-1)/p).beta + n.((p-1)/p).gamma
 *
 * For the non-power-of-two case:
 *
 * Cost = (2.floor(lgp)+2).alpha + (2.((p-1)/p) + 2).n.beta + n.(1+(p-1)/p).gamma
 */




int MPIR_Allreduce_intra_pro_reduce_scatter_allgather(const void *sendbuf,
                                                  void *recvbuf,
                                                  int count,
                                                  MPI_Datatype datatype,
                                                  MPI_Op op,
                                                  MPIR_Comm * comm_ptr, MPIR_Errflag_t * errflag)
{
    MPIR_CHKLMEM_DECL(3);
    int comm_size, rank;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    int mask, dst, pof2, newrank, rem, newdst, i,
        send_idx, recv_idx, last_idx, send_cnt, recv_cnt, *cnts, *disps;
    MPI_Aint true_extent, true_lb, extent;
    void *tmp_buf;

    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;

    if (print_fun_name==1 && rank==0) fprintf(stderr,"Pro_reduce_scatter_allgather\n");

    /* need to allocate temporary buffer to store incoming data */
    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);
    MPIR_Datatype_get_extent_macro(datatype, extent);

    MPIR_CHKLMEM_MALLOC(tmp_buf, void *, count * (MPL_MAX(extent, true_extent)), mpi_errno,
                        "temporary buffer", MPL_MEM_BUFFER);

    /* adjust for potential negative lower bound in datatype */
    tmp_buf = (void *) ((char *) tmp_buf - true_lb);

    /* copy local data into recvbuf */
    if (sendbuf != MPI_IN_PLACE) {
        mpi_errno = MPIR_Localcopy(sendbuf, count, datatype, recvbuf, count, datatype);
        MPIR_ERR_CHECK(mpi_errno);
    }

    /* get nearest power-of-two less than or equal to comm_size */
    pof2 = comm_ptr->coll.pof2;

    rem = comm_size - pof2;

    /* In the non-power-of-two case, all even-numbered
     * processes of rank < 2*rem send their data to
     * (rank+1). These even-numbered processes no longer
     * participate in the algorithm until the very end. The
     * remaining processes form a nice power-of-two. */

    if (rank < 2 * rem) {
        if (rank % 2 == 0) {    /* even */
            mpi_errno = MPIC_Send(recvbuf, count,
                                  datatype, rank + 1, MPIR_ALLREDUCE_TAG, comm_ptr, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag =
                    MPIX_ERR_PROC_FAILED ==
                    MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
                MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }

            /* temporarily set the rank to -1 so that this
             * process does not pariticipate in recursive
             * doubling */
            newrank = -1;
        } else {        /* odd */
            mpi_errno = MPIC_Recv(tmp_buf, count,
                                  datatype, rank - 1,
                                  MPIR_ALLREDUCE_TAG, comm_ptr, MPI_STATUS_IGNORE, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag =
                    MPIX_ERR_PROC_FAILED ==
                    MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
                MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }

            /* do the reduction on received data. since the
             * ordering is right, it doesn't matter whether
             * the operation is commutative or not. */
            mpi_errno = MPIR_Reduce_local(tmp_buf, recvbuf, count, datatype, op);
            MPIR_ERR_CHECK(mpi_errno);

            /* change the rank */
            newrank = rank / 2;
        }
    } else      /* rank >= 2*rem */
        newrank = rank - rem;

    /* If op is user-defined or count is less than pof2, use
     * recursive doubling algorithm. Otherwise do a reduce-scatter
     * followed by allgather. (If op is user-defined,
     * derived datatypes are allowed and the user could pass basic
     * datatypes on one process and derived on another as long as
     * the type maps are the same. Breaking up derived
     * datatypes to do the reduce-scatter is tricky, therefore
     * using recursive doubling in that case.) */

#ifdef HAVE_ERROR_CHECKING
    MPIR_Assert(HANDLE_IS_BUILTIN(op));
    MPIR_Assert(count >= pof2);
#endif /* HAVE_ERROR_CHECKING */

    if (newrank != -1) {

        int nprocs,context_id,datatype_sz,len,temp_rank,basic_seg_size;	   
        unsigned long seg_size,start_seg,last_seg,ciphertext_len,plaintext_len=0;
        int binary[128];
        // int crnt_lvl;
        nprocs=comm_size;
        seg_size = count / nprocs;
		// basic_seg_size = seg_size;
        int prank = rank_test;


        MPIR_CHKLMEM_MALLOC(cnts, int *, pof2 * sizeof(int), mpi_errno, "counts", MPL_MEM_BUFFER);
        MPIR_CHKLMEM_MALLOC(disps, int *, pof2 * sizeof(int), mpi_errno, "displacements", MPL_MEM_BUFFER);

        for (i = 0; i < pof2; i++)
            cnts[i] = count / pof2;
        if ((count % pof2) > 0) {
            for (i = 0; i < (count % pof2); i++)
                cnts[i] += 1;
        }

        if (pof2)
            disps[0] = 0;
        for (i = 1; i < pof2; i++)
            disps[i] = disps[i - 1] + cnts[i - 1];

        for (i=0; i<10; i++)	binary[i]=0;

                
        temp_rank = rank;        
        i = 0;
        // Decimal to Binary

        while (temp_rank > 0) { 		 
            binary[i] = temp_rank % 2; 
            temp_rank = temp_rank >> 1; 
            i++;             
        }                
        
        int offset,indx;
        int v = 1,k=0 , j=0;
        int Max_lvl=0;
        int Max_pow=comm_size;
        int lvl=1;
        int indx_pow;
        int offset_index=0;
        int lvl_pow;
        int base_jump, jump=0;

        // if (binary[0]) offset=0; else offset=8; // Mohsen: comm_size/2 ??
        
        while (v < comm_size) {					
            v = v << 1;
            Max_lvl++;
        }

        mask = 0x1;
        send_idx = recv_idx = 0;
        last_idx = pof2;
        // crnt_lvl = 0;
        int next_free=0;
        int last_oprd=0;

        int MyIndex[16]={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
        
        while (mask < pof2) {
            newdst = newrank ^ mask;
            /* find real rank of dest */
            dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

#if 0            
            if (rank==prank) {
            fprintf(stderr,"recvbuf = {%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f} [%d] nprocs=%d seg_size=%d \n", 
                *((float *) (recvbuf+MyIndex[0]*extent)) ,*((float *) (recvbuf+MyIndex[1]*extent)) ,
                *((float *) (recvbuf+MyIndex[2]*extent)) ,*((float *) (recvbuf+MyIndex[3]*extent)) ,
                *((float *) (recvbuf+MyIndex[4]*extent)) ,*((float *) (recvbuf+MyIndex[5]*extent)) ,
                *((float *) (recvbuf+MyIndex[6]*extent)) ,*((float *) (recvbuf+MyIndex[7]*extent)) , 
                *((float *) (recvbuf+MyIndex[8]*extent)) ,*((float *) (recvbuf+MyIndex[9]*extent)) ,
                *((float *) (recvbuf+MyIndex[10]*extent)),*((float *) (recvbuf+MyIndex[11]*extent)) ,
                *((float *) (recvbuf+MyIndex[12]*extent)),*((float *) (recvbuf+MyIndex[13]*extent)) ,
                *((float *) (recvbuf+MyIndex[14]*extent)),*((float *) (recvbuf+MyIndex[15]*extent)) , 
                rank, nprocs, seg_size );

            fprintf(stderr,"Tempbuf = {%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f} [%d] nprocs=%d seg_size=%d \n", 
                *((float *) (tmp_buf+MyIndex[0]*extent)) ,*((float *) (tmp_buf+MyIndex[1]*extent)) ,
                *((float *) (tmp_buf+MyIndex[2]*extent)) ,*((float *) (tmp_buf+MyIndex[3]*extent)) ,
                *((float *) (tmp_buf+MyIndex[4]*extent)) ,*((float *) (tmp_buf+MyIndex[5]*extent)) ,
                *((float *) (tmp_buf+MyIndex[6]*extent)) ,*((float *) (tmp_buf+MyIndex[7]*extent)) , 
                *((float *) (tmp_buf+MyIndex[8]*extent)) ,*((float *) (tmp_buf+MyIndex[9]*extent)) ,
                *((float *) (tmp_buf+MyIndex[10]*extent)),*((float *) (tmp_buf+MyIndex[11]*extent)) ,
                *((float *) (tmp_buf+MyIndex[12]*extent)),*((float *) (tmp_buf+MyIndex[13]*extent)) ,
                *((float *) (tmp_buf+MyIndex[14]*extent)),*((float *) (tmp_buf+MyIndex[15]*extent)) , 
                rank, nprocs, seg_size );    

            }
#endif
            send_cnt = recv_cnt = 0;
#if 1            
            if (newrank < newdst) {
                send_idx = recv_idx + pof2 / (mask * 2);
                for (i = send_idx; i < last_idx; i++)
                    send_cnt += cnts[i];
                for (i = recv_idx; i < send_idx; i++)
                    recv_cnt += cnts[i];
            } else {
                recv_idx = send_idx + pof2 / (mask * 2);
                for (i = send_idx; i < recv_idx; i++)
                    send_cnt += cnts[i];
                for (i = recv_idx; i < last_idx; i++)
                    recv_cnt += cnts[i];
            }
#endif
            /* Add By Mohsen */
					
            offset_index = ((binary[lvl-1]+1)%2);  /* Not binary[] */

            lvl_pow = 1;		            
            for (i=1 ; i<=lvl ; i++){
                lvl_pow *= 2;
                int index = lvl-i-1;
                if (index>=0) 
                    offset_index += binary[index] * lvl_pow;
            }
            
            offset = offset_index * (Max_pow/lvl_pow);
            
            // if (rank==prank) fprintf(stderr,COLOR_CYAN"Rank=%d send_cnt=%d mask=%d\t offset_index=%d offset=%d comm_size=%d Max_pow=%d lvl_pow=%d Max_lvl=%d lvl=%d binary=%d%d%d"COLOR_RESET"\n",rank,send_cnt,mask,offset_index, offset,comm_size,Max_pow,lvl_pow,Max_lvl,lvl,binary[0],binary[1],binary[2]);
        
            for (i=0 ; i<(Max_pow/lvl_pow) ; i++){
                
                indx_pow=1;
                indx=0;
                for (j=Max_lvl-1 ; j>=lvl ; j--){			
                    if (i%indx_pow==0) { 
                        binary[j]=(binary[j]+1)%2; /* Not binary[] */
                    }
                    indx += binary[j]*indx_pow;                    
                    indx_pow *= 2;
                }

                indx += offset;
                
                // if (rank==prank) fprintf(stderr,"[%d]recvbuf[%d]:(%0.f)  %d , %d   (%d/%d)\n",rank,send_cnt,(float *) recvbuf +indx*seg_size*extent ,indx*seg_size,indx,i,Max_pow/lvl_pow);
                
                if (lvl>1){
                    
                    jump=0;
                    for (k=2; k<=lvl; k++){		
                        // if (rank==prank) fprintf(stderr,COLOR_YELLOW"Rank=%d send_cnt=%d mask=%d\t tmp_buf_index=%d recvbuf_index=%d jump=%d k=%d"COLOR_RESET"\n",rank,send_cnt,mask,last_oprd*seg_size+jump,indx*seg_size,jump,k);
                                                
                        /* tmp_buf contains data received in this step.
                        * recvbuf contains data accumulated so far */

                        // if (rank==prank) fprintf(stderr,COLOR_CYAN"[%d] Bfr Reduce send_cnt=%d  mask=%d\t tmp= %.0f recv= %.0f indx=%d lvl=%d"COLOR_RESET"\n",rank,send_cnt,mask, (float *)(tmp_buf+(last_oprd*seg_size+jump+1)*extent), (float *)recvbuf+(indx*seg_size+1)*extent,indx,lvl);
                        mpi_errno = MPIR_Reduce_local((char *) (tmp_buf+(last_oprd*seg_size+jump)*extent),(char *) (recvbuf+indx*seg_size*extent),seg_size,datatype,op);	
                        // if (rank==prank) fprintf(stderr,COLOR_MAGENTA"[%d] Aft Reduce send_cnt=%d  mask=%d\t tmp= %.0f recv= %.0f indx=%d lvl=%d"COLOR_RESET"\n",rank,send_cnt,mask, (float *)(tmp_buf+(last_oprd*seg_size+jump+1)*extent), (float *)recvbuf+(indx*seg_size+1)*extent,indx,lvl);
                

                        if (k==2) base_jump=seg_size*(comm_size/4);	else base_jump = base_jump/2;
                        jump += base_jump;
                    }
                    
                    last_oprd++;
                }

                /* Send data from recvbuf. Recv into tmp_buf */
                mpi_errno = MPIC_Sendrecv((char *) recvbuf +
                                        indx*seg_size*extent,
                                        seg_size, datatype,
                                        dst, MPIR_ALLREDUCE_TAG,
                                        (char *) tmp_buf +
                                        next_free*seg_size*extent,
                                        seg_size, datatype, dst,
                                        MPIR_ALLREDUCE_TAG, comm_ptr, MPI_STATUS_IGNORE, errflag);
                if (mpi_errno) {
                    /* for communication errors, just record the error but continue */
                    *errflag =
                        MPIX_ERR_PROC_FAILED ==
                        MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
                    MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                }
            
                next_free++;
						
            }

            /* update send_idx for next iteration */
            lvl++;
            send_idx = recv_idx;
            mask <<= 1;

            /* update last_idx, but not in last iteration
             * because the value is needed in the allgather
             * step below. */
            if (mask < pof2)
                last_idx = recv_idx + pof2 / mask;
        }        
        
        indx_pow=1;
        indx=0;
        for (i=Max_lvl-1 ; i>=0 ; i--){
            indx += binary[i]*indx_pow;
            // if (rank==prank) fprintf(stderr,COLOR_GREEN"LAST:[%d] indx=%d Max_lvl=%d binary[%d]=%d"COLOR_RESET"\n",rank,Max_lvl,i,binary[i]);
            indx_pow *= 2;            
        }


        jump=0;
        for (k=2; k<=lvl; k++){		
            // if (rank==prank) fprintf(stderr,COLOR_YELLOW"LAST:[%d] indx=%d send_cnt=%d mask=%d\t tmp_buf_index=%d recvbuf_index=%d jump=%d k=%d"COLOR_RESET"\n",rank,indx,send_cnt,mask,last_oprd*seg_size+jump,indx*seg_size,jump,k);            
        
            
            // if (rank==prank) fprintf(stderr,COLOR_CYAN"LAST:[%d] Bfr Reduce send_cnt=%d  mask=%d\t tmp= %.0f recv= %.0f indx=%d lvl=%d"COLOR_RESET"\n",rank,send_cnt,mask, (float *)(tmp_buf+(last_oprd*seg_size+jump+1)*extent), (float *)recvbuf+(indx*seg_size+1)*extent,indx,lvl);
            mpi_errno = MPIR_Reduce_local((char *) (tmp_buf+(last_oprd*seg_size+jump)*extent),(char *) (recvbuf+indx*seg_size*extent),seg_size,datatype,op);	
            // if (rank==prank) fprintf(stderr,COLOR_MAGENTA"LAST:[%d] Aft Reduce send_cnt=%d  mask=%d\t tmp= %.0f recv= %.0f indx=%d lvl=%d"COLOR_RESET"\n",rank,send_cnt,mask, (float *)(tmp_buf+(last_oprd*seg_size+jump+1)*extent), (float *)recvbuf+(indx*seg_size+1)*extent,indx,lvl);
            if (k==2) base_jump=seg_size*(comm_size/4);	else base_jump = base_jump/2;
            jump += base_jump;
        }
        

#if 0
        if (rank==prank) {
            fprintf(stderr,"F-recvbuf = {%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f} [%d] nprocs=%d seg_size=%d \n", 
                *((float *) (recvbuf+MyIndex[0]*extent)) ,*((float *) (recvbuf+MyIndex[1]*extent)) ,
                *((float *) (recvbuf+MyIndex[2]*extent)) ,*((float *) (recvbuf+MyIndex[3]*extent)) ,
                *((float *) (recvbuf+MyIndex[4]*extent)) ,*((float *) (recvbuf+MyIndex[5]*extent)) ,
                *((float *) (recvbuf+MyIndex[6]*extent)) ,*((float *) (recvbuf+MyIndex[7]*extent)) , 
                *((float *) (recvbuf+MyIndex[8]*extent)) ,*((float *) (recvbuf+MyIndex[9]*extent)) ,
                *((float *) (recvbuf+MyIndex[10]*extent)),*((float *) (recvbuf+MyIndex[11]*extent)) ,
                *((float *) (recvbuf+MyIndex[12]*extent)),*((float *) (recvbuf+MyIndex[13]*extent)) ,
                *((float *) (recvbuf+MyIndex[14]*extent)),*((float *) (recvbuf+MyIndex[15]*extent)) , 
                rank, nprocs, seg_size );

            fprintf(stderr,"F-Tempbuf = {%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f} [%d] nprocs=%d seg_size=%d \n", 
                *((float *) (tmp_buf+MyIndex[0]*extent)) ,*((float *) (tmp_buf+MyIndex[1]*extent)) ,
                *((float *) (tmp_buf+MyIndex[2]*extent)) ,*((float *) (tmp_buf+MyIndex[3]*extent)) ,
                *((float *) (tmp_buf+MyIndex[4]*extent)) ,*((float *) (tmp_buf+MyIndex[5]*extent)) ,
                *((float *) (tmp_buf+MyIndex[6]*extent)) ,*((float *) (tmp_buf+MyIndex[7]*extent)) , 
                *((float *) (tmp_buf+MyIndex[8]*extent)) ,*((float *) (tmp_buf+MyIndex[9]*extent)) ,
                *((float *) (tmp_buf+MyIndex[10]*extent)),*((float *) (tmp_buf+MyIndex[11]*extent)) ,
                *((float *) (tmp_buf+MyIndex[12]*extent)),*((float *) (tmp_buf+MyIndex[13]*extent)) ,
                *((float *) (tmp_buf+MyIndex[14]*extent)),*((float *) (tmp_buf+MyIndex[15]*extent)) , 
                rank, nprocs, seg_size );    

            }
#endif
        /* now do the allgather */

        mask >>= 1;
        while (mask > 0) {
            newdst = newrank ^ mask;
            /* find real rank of dest */
            dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

            send_cnt = recv_cnt = 0;
            if (newrank < newdst) {
                /* update last_idx except on first iteration */
                if (mask != pof2 / 2)
                    last_idx = last_idx + pof2 / (mask * 2);

                recv_idx = send_idx + pof2 / (mask * 2);
                for (i = send_idx; i < recv_idx; i++)
                    send_cnt += cnts[i];
                for (i = recv_idx; i < last_idx; i++)
                    recv_cnt += cnts[i];
            } else {
                recv_idx = send_idx - pof2 / (mask * 2);
                for (i = send_idx; i < last_idx; i++)
                    send_cnt += cnts[i];
                for (i = recv_idx; i < send_idx; i++)
                    recv_cnt += cnts[i];
            }

            mpi_errno = MPIC_Sendrecv((char *) recvbuf +
                                      disps[send_idx] * extent,
                                      send_cnt, datatype,
                                      dst, MPIR_ALLREDUCE_TAG,
                                      (char *) recvbuf +
                                      disps[recv_idx] * extent,
                                      recv_cnt, datatype, dst,
                                      MPIR_ALLREDUCE_TAG, comm_ptr, MPI_STATUS_IGNORE, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag =
                    MPIX_ERR_PROC_FAILED ==
                    MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
                MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }

            if (newrank > newdst)
                send_idx = recv_idx;

            mask >>= 1;
        }
    }
    /* In the non-power-of-two case, all odd-numbered
     * processes of rank < 2*rem send the result to
     * (rank-1), the ranks who didn't participate above. */
    if (rank < 2 * rem) {
        if (rank % 2)   /* odd */
            mpi_errno = MPIC_Send(recvbuf, count,
                                  datatype, rank - 1, MPIR_ALLREDUCE_TAG, comm_ptr, errflag);
        else    /* even */
            mpi_errno = MPIC_Recv(recvbuf, count,
                                  datatype, rank + 1,
                                  MPIR_ALLREDUCE_TAG, comm_ptr, MPI_STATUS_IGNORE, errflag);
        if (mpi_errno) {
            /* for communication errors, just record the error but continue */
            *errflag =
                MPIX_ERR_PROC_FAILED ==
                MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
            MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
    }
  fn_exit:
    MPIR_CHKLMEM_FREEALL();
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}




int MPIR_Allreduce_intra_reduce_scatter_allgather(const void *sendbuf,
                                                  void *recvbuf,
                                                  int count,
                                                  MPI_Datatype datatype,
                                                  MPI_Op op,
                                                  MPIR_Comm * comm_ptr, MPIR_Errflag_t * errflag)
{
    MPIR_CHKLMEM_DECL(3);
    int comm_size, rank;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    int mask, dst, pof2, newrank, rem, newdst, i,
        send_idx, recv_idx, last_idx, send_cnt, recv_cnt, *cnts, *disps;
    MPI_Aint true_extent, true_lb, extent;
    void *tmp_buf;

    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;

    if (print_fun_name==1 && rank==0)  fprintf(stderr,"Normal_reduce_scatter_allgather\n");

    /* need to allocate temporary buffer to store incoming data */
    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);
    MPIR_Datatype_get_extent_macro(datatype, extent);

    MPIR_CHKLMEM_MALLOC(tmp_buf, void *, count * (MPL_MAX(extent, true_extent)), mpi_errno,
                        "temporary buffer", MPL_MEM_BUFFER);

    /* adjust for potential negative lower bound in datatype */
    tmp_buf = (void *) ((char *) tmp_buf - true_lb);

    /* copy local data into recvbuf */
    if (sendbuf != MPI_IN_PLACE) {
        mpi_errno = MPIR_Localcopy(sendbuf, count, datatype, recvbuf, count, datatype);
        MPIR_ERR_CHECK(mpi_errno);
    }

    /* get nearest power-of-two less than or equal to comm_size */
    pof2 = comm_ptr->coll.pof2;

    rem = comm_size - pof2;

    /* In the non-power-of-two case, all even-numbered
     * processes of rank < 2*rem send their data to
     * (rank+1). These even-numbered processes no longer
     * participate in the algorithm until the very end. The
     * remaining processes form a nice power-of-two. */

    if (rank < 2 * rem) {
        if (rank % 2 == 0) {    /* even */
            mpi_errno = MPIC_Send(recvbuf, count,
                                  datatype, rank + 1, MPIR_ALLREDUCE_TAG, comm_ptr, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag =
                    MPIX_ERR_PROC_FAILED ==
                    MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
                MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }

            /* temporarily set the rank to -1 so that this
             * process does not pariticipate in recursive
             * doubling */
            newrank = -1;
        } else {        /* odd */
            mpi_errno = MPIC_Recv(tmp_buf, count,
                                  datatype, rank - 1,
                                  MPIR_ALLREDUCE_TAG, comm_ptr, MPI_STATUS_IGNORE, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag =
                    MPIX_ERR_PROC_FAILED ==
                    MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
                MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }

            /* do the reduction on received data. since the
             * ordering is right, it doesn't matter whether
             * the operation is commutative or not. */
            mpi_errno = MPIR_Reduce_local(tmp_buf, recvbuf, count, datatype, op);
            MPIR_ERR_CHECK(mpi_errno);

            /* change the rank */
            newrank = rank / 2;
        }
    } else      /* rank >= 2*rem */
        newrank = rank - rem;

    /* If op is user-defined or count is less than pof2, use
     * recursive doubling algorithm. Otherwise do a reduce-scatter
     * followed by allgather. (If op is user-defined,
     * derived datatypes are allowed and the user could pass basic
     * datatypes on one process and derived on another as long as
     * the type maps are the same. Breaking up derived
     * datatypes to do the reduce-scatter is tricky, therefore
     * using recursive doubling in that case.) */

#ifdef HAVE_ERROR_CHECKING
    MPIR_Assert(HANDLE_IS_BUILTIN(op));
    MPIR_Assert(count >= pof2);
#endif /* HAVE_ERROR_CHECKING */

    if (newrank != -1) {
        MPIR_CHKLMEM_MALLOC(cnts, int *, pof2 * sizeof(int), mpi_errno, "counts", MPL_MEM_BUFFER);
        MPIR_CHKLMEM_MALLOC(disps, int *, pof2 * sizeof(int), mpi_errno, "displacements",
                            MPL_MEM_BUFFER);

        for (i = 0; i < pof2; i++)
            cnts[i] = count / pof2;
        if ((count % pof2) > 0) {
            for (i = 0; i < (count % pof2); i++)
                cnts[i] += 1;
        }

        if (pof2)
            disps[0] = 0;
        for (i = 1; i < pof2; i++)
            disps[i] = disps[i - 1] + cnts[i - 1];

        mask = 0x1;
        send_idx = recv_idx = 0;
        last_idx = pof2;
        while (mask < pof2) {
            newdst = newrank ^ mask;
            /* find real rank of dest */
            dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

            send_cnt = recv_cnt = 0;
            if (newrank < newdst) {
                send_idx = recv_idx + pof2 / (mask * 2);
                for (i = send_idx; i < last_idx; i++)
                    send_cnt += cnts[i];
                for (i = recv_idx; i < send_idx; i++)
                    recv_cnt += cnts[i];
            } else {
                recv_idx = send_idx + pof2 / (mask * 2);
                for (i = send_idx; i < recv_idx; i++)
                    send_cnt += cnts[i];
                for (i = recv_idx; i < last_idx; i++)
                    recv_cnt += cnts[i];
            }

            /* Send data from recvbuf. Recv into tmp_buf */
            mpi_errno = MPIC_Sendrecv((char *) recvbuf +
                                      disps[send_idx] * extent,
                                      send_cnt, datatype,
                                      dst, MPIR_ALLREDUCE_TAG,
                                      (char *) tmp_buf +
                                      disps[recv_idx] * extent,
                                      recv_cnt, datatype, dst,
                                      MPIR_ALLREDUCE_TAG, comm_ptr, MPI_STATUS_IGNORE, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag =
                    MPIX_ERR_PROC_FAILED ==
                    MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
                MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }

            /* tmp_buf contains data received in this step.
             * recvbuf contains data accumulated so far */

            /* This algorithm is used only for predefined ops
             * and predefined ops are always commutative. */
            mpi_errno = MPIR_Reduce_local(((char *) tmp_buf + disps[recv_idx] * extent),
                                          ((char *) recvbuf + disps[recv_idx] * extent),
                                          recv_cnt, datatype, op);
            MPIR_ERR_CHECK(mpi_errno);

            /* update send_idx for next iteration */
            send_idx = recv_idx;
            mask <<= 1;

            /* update last_idx, but not in last iteration
             * because the value is needed in the allgather
             * step below. */
            if (mask < pof2)
                last_idx = recv_idx + pof2 / mask;
        }

        /* now do the allgather */

        mask >>= 1;
        while (mask > 0) {
            newdst = newrank ^ mask;
            /* find real rank of dest */
            dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

            send_cnt = recv_cnt = 0;
            if (newrank < newdst) {
                /* update last_idx except on first iteration */
                if (mask != pof2 / 2)
                    last_idx = last_idx + pof2 / (mask * 2);

                recv_idx = send_idx + pof2 / (mask * 2);
                for (i = send_idx; i < recv_idx; i++)
                    send_cnt += cnts[i];
                for (i = recv_idx; i < last_idx; i++)
                    recv_cnt += cnts[i];
            } else {
                recv_idx = send_idx - pof2 / (mask * 2);
                for (i = send_idx; i < last_idx; i++)
                    send_cnt += cnts[i];
                for (i = recv_idx; i < send_idx; i++)
                    recv_cnt += cnts[i];
            }

            mpi_errno = MPIC_Sendrecv((char *) recvbuf +
                                      disps[send_idx] * extent,
                                      send_cnt, datatype,
                                      dst, MPIR_ALLREDUCE_TAG,
                                      (char *) recvbuf +
                                      disps[recv_idx] * extent,
                                      recv_cnt, datatype, dst,
                                      MPIR_ALLREDUCE_TAG, comm_ptr, MPI_STATUS_IGNORE, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag =
                    MPIX_ERR_PROC_FAILED ==
                    MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
                MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }

            if (newrank > newdst)
                send_idx = recv_idx;

            mask >>= 1;
        }
    }
    /* In the non-power-of-two case, all odd-numbered
     * processes of rank < 2*rem send the result to
     * (rank-1), the ranks who didn't participate above. */
    if (rank < 2 * rem) {
        if (rank % 2)   /* odd */
            mpi_errno = MPIC_Send(recvbuf, count,
                                  datatype, rank - 1, MPIR_ALLREDUCE_TAG, comm_ptr, errflag);
        else    /* even */
            mpi_errno = MPIC_Recv(recvbuf, count,
                                  datatype, rank + 1,
                                  MPIR_ALLREDUCE_TAG, comm_ptr, MPI_STATUS_IGNORE, errflag);
        if (mpi_errno) {
            /* for communication errors, just record the error but continue */
            *errflag =
                MPIX_ERR_PROC_FAILED ==
                MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
            MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
    }
  fn_exit:
    MPIR_CHKLMEM_FREEALL();
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}


#if 0
 ////   Under developing code

int MPIR_Allreduce_intra_pro_reduce_scatter_allgather(const void *sendbuf,
                                                  void *recvbuf,
                                                  int count,
                                                  MPI_Datatype datatype,
                                                  MPI_Op op,
                                                  MPIR_Comm * comm_ptr, MPIR_Errflag_t * errflag)
{
    MPIR_CHKLMEM_DECL(3);
    int comm_size, rank;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    int mask, dst, pof2, newrank, rem, newdst, i,
        send_idx, recv_idx, last_idx, send_cnt, recv_cnt, *cnts, *disps;
    MPI_Aint true_extent, true_lb, extent;
    void *tmp_buf;

    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;

    /* need to allocate temporary buffer to store incoming data */
    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);
    MPIR_Datatype_get_extent_macro(datatype, extent);

    MPIR_CHKLMEM_MALLOC(tmp_buf, void *, count * (MPL_MAX(extent, true_extent)), mpi_errno,
                        "temporary buffer", MPL_MEM_BUFFER);

    /* adjust for potential negative lower bound in datatype */
    tmp_buf = (void *) ((char *) tmp_buf - true_lb);

    /* copy local data into recvbuf */
    if (sendbuf != MPI_IN_PLACE) {
        mpi_errno = MPIR_Localcopy(sendbuf, count, datatype, recvbuf, count, datatype);
        MPIR_ERR_CHECK(mpi_errno);
    }

    /* get nearest power-of-two less than or equal to comm_size */
    pof2 = comm_ptr->coll.pof2;

    rem = comm_size - pof2;

    /* In the non-power-of-two case, all even-numbered
     * processes of rank < 2*rem send their data to
     * (rank+1). These even-numbered processes no longer
     * participate in the algorithm until the very end. The
     * remaining processes form a nice power-of-two. */

    if (rank < 2 * rem) {
        if (rank % 2 == 0) {    /* even */
            mpi_errno = MPIC_Send(recvbuf, count,
                                  datatype, rank + 1, MPIR_ALLREDUCE_TAG, comm_ptr, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag =
                    MPIX_ERR_PROC_FAILED ==
                    MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
                MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }

            /* temporarily set the rank to -1 so that this
             * process does not pariticipate in recursive
             * doubling */
            newrank = -1;
        } else {        /* odd */
            mpi_errno = MPIC_Recv(tmp_buf, count,
                                  datatype, rank - 1,
                                  MPIR_ALLREDUCE_TAG, comm_ptr, MPI_STATUS_IGNORE, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag =
                    MPIX_ERR_PROC_FAILED ==
                    MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
                MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }

            /* do the reduction on received data. since the
             * ordering is right, it doesn't matter whether
             * the operation is commutative or not. */
            mpi_errno = MPIR_Reduce_local(tmp_buf, recvbuf, count, datatype, op);
            MPIR_ERR_CHECK(mpi_errno);

            /* change the rank */
            newrank = rank / 2;
        }
    } else      /* rank >= 2*rem */
        newrank = rank - rem;

    /* If op is user-defined or count is less than pof2, use
     * recursive doubling algorithm. Otherwise do a reduce-scatter
     * followed by allgather. (If op is user-defined,
     * derived datatypes are allowed and the user could pass basic
     * datatypes on one process and derived on another as long as
     * the type maps are the same. Breaking up derived
     * datatypes to do the reduce-scatter is tricky, therefore
     * using recursive doubling in that case.) */

#ifdef HAVE_ERROR_CHECKING
    MPIR_Assert(HANDLE_IS_BUILTIN(op));
    MPIR_Assert(count >= pof2);
#endif /* HAVE_ERROR_CHECKING */

    if (newrank != -1) {

        int nprocs,context_id,datatype_sz,len,temp_rank,basic_seg_size;	   
        unsigned long seg_size,start_seg,last_seg,ciphertext_len,plaintext_len=0;
        int binary[128];
        int crnt_lvl;
        nprocs=comm_size;
        seg_size = count / nprocs;
		// basic_seg_size = seg_size;
        int prank = rank_test;


        MPIR_CHKLMEM_MALLOC(cnts, int *, pof2 * sizeof(int), mpi_errno, "counts", MPL_MEM_BUFFER);
        MPIR_CHKLMEM_MALLOC(disps, int *, pof2 * sizeof(int), mpi_errno, "displacements",
                            MPL_MEM_BUFFER);

        for (i = 0; i < pof2; i++)
            cnts[i] = count / pof2;
        if ((count % pof2) > 0) {
            for (i = 0; i < (count % pof2); i++)
                cnts[i] += 1;
        }

        if (pof2)
            disps[0] = 0;
        for (i = 1; i < pof2; i++)
            disps[i] = disps[i - 1] + cnts[i - 1];

        for (i=0; i<10; i++)	binary[i]=0;

                
        temp_rank = rank;        
        i = 0;
        // Decimal to Binary
        
        //if (rank==prank) fprintf(stderr,"binary0=[%d][%d][%d]\n",binary[0],binary[1],binary[2]);

        while (temp_rank > 0) { 		 
            binary[i] = temp_rank % 2; 
            temp_rank = temp_rank >> 1; 
            i++; 
            //if (rank==prank) fprintf(stderr,"binary+=[%d][%d][%d]\n",binary[0],binary[1],binary[2]);
        }
        
        //if (rank==prank) fprintf(stderr,"binary1=[%d][%d][%d]\n",binary[0],binary[1],binary[2]);
        
        int offset,indx;
        int v = 1,k=0 , j=0;
        int Max_lvl=0;
        int Max_pow=comm_size;
        int lvl=1;
        int indx_pow;
        int offset_index=0;
        int lvl_pow;
        int base_jump, jump=0;

        if (binary[0]) offset=0; else offset=8; // Mohsen: comm_size/2 ??
        
        while (v < comm_size) {					
            v = v << 1;
            Max_lvl++;
        }

        mask = 0x1;
        send_idx = recv_idx = 0;
        last_idx = pof2;
        // crnt_lvl = 0;
        int next_free=0;
        int last_oprd=0;

        int MyIndex[16]={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
        
        while (mask < pof2) {
            newdst = newrank ^ mask;
            /* find real rank of dest */
            dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

            
            if (rank==prank) {
            fprintf(stderr,"recvbuf = {%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f} [%d] nprocs=%d seg_size=%d \n", 
                *((float *) (recvbuf+MyIndex[0]*extent)) ,*((float *) (recvbuf+MyIndex[1]*extent)) ,
                *((float *) (recvbuf+MyIndex[2]*extent)) ,*((float *) (recvbuf+MyIndex[3]*extent)) ,
                *((float *) (recvbuf+MyIndex[4]*extent)) ,*((float *) (recvbuf+MyIndex[5]*extent)) ,
                *((float *) (recvbuf+MyIndex[6]*extent)) ,*((float *) (recvbuf+MyIndex[7]*extent)) , 
                *((float *) (recvbuf+MyIndex[8]*extent)) ,*((float *) (recvbuf+MyIndex[9]*extent)) ,
                *((float *) (recvbuf+MyIndex[10]*extent)),*((float *) (recvbuf+MyIndex[11]*extent)) ,
                *((float *) (recvbuf+MyIndex[12]*extent)),*((float *) (recvbuf+MyIndex[13]*extent)) ,
                *((float *) (recvbuf+MyIndex[14]*extent)),*((float *) (recvbuf+MyIndex[15]*extent)) , 
                rank, nprocs, seg_size );

            fprintf(stderr,"Tempbuf = {%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f} [%d] nprocs=%d seg_size=%d \n", 
                *((float *) (tmp_buf+MyIndex[0]*extent)) ,*((float *) (tmp_buf+MyIndex[1]*extent)) ,
                *((float *) (tmp_buf+MyIndex[2]*extent)) ,*((float *) (tmp_buf+MyIndex[3]*extent)) ,
                *((float *) (tmp_buf+MyIndex[4]*extent)) ,*((float *) (tmp_buf+MyIndex[5]*extent)) ,
                *((float *) (tmp_buf+MyIndex[6]*extent)) ,*((float *) (tmp_buf+MyIndex[7]*extent)) , 
                *((float *) (tmp_buf+MyIndex[8]*extent)) ,*((float *) (tmp_buf+MyIndex[9]*extent)) ,
                *((float *) (tmp_buf+MyIndex[10]*extent)),*((float *) (tmp_buf+MyIndex[11]*extent)) ,
                *((float *) (tmp_buf+MyIndex[12]*extent)),*((float *) (tmp_buf+MyIndex[13]*extent)) ,
                *((float *) (tmp_buf+MyIndex[14]*extent)),*((float *) (tmp_buf+MyIndex[15]*extent)) , 
                rank, nprocs, seg_size );    

            }

            send_cnt = recv_cnt = 0;
            if (newrank < newdst) {
                send_idx = recv_idx + pof2 / (mask * 2);
                for (i = send_idx; i < last_idx; i++)
                    send_cnt += cnts[i];
                for (i = recv_idx; i < send_idx; i++)
                    recv_cnt += cnts[i];
            } else {
                recv_idx = send_idx + pof2 / (mask * 2);
                for (i = send_idx; i < recv_idx; i++)
                    send_cnt += cnts[i];
                for (i = recv_idx; i < last_idx; i++)
                    recv_cnt += cnts[i];
            }

            /* Add By Mohsen */
					
            offset_index = ((binary[lvl-1]+1)%2);  /* Not binary[] */

            lvl_pow = 1;		            
            for (i=1 ; i<=lvl ; i++){
                lvl_pow *= 2;
                int index = lvl-i-1;
                if (index>=0) 
                    offset_index += binary[index] * lvl_pow;
            }
            //if (rank==prank) fprintf(stderr,"binary3=[%d][%d][%d]\n",binary[0],binary[1],binary[2]);
            offset = offset_index * (Max_pow/lvl_pow);
            
            if (rank==prank) fprintf(stderr,COLOR_CYAN"Rank=%d send_cnt=%d mask=%d\t offset_index=%d offset=%d comm_size=%d Max_pow=%d lvl_pow=%d Max_lvl=%d lvl=%d binary=%d%d%d"COLOR_RESET"\n",rank,send_cnt,mask,offset_index, offset,comm_size,Max_pow,lvl_pow,Max_lvl,lvl,binary[0],binary[1],binary[2]);
        
            for (i=0 ; i<(Max_pow/lvl_pow) ; i++){
                
                indx_pow=1;
                indx=0;
                for (j=Max_lvl-1 ; j>=lvl ; j--){			
                    if (i%indx_pow==0) { 
                        binary[j]=(binary[j]+1)%2; /* Not binary[] */
                    }
                    indx += binary[j]*indx_pow;
                    // if (rank==prank) fprintf(stderr,COLOR_RED"Rank=%d send_cnt=%d mask=%d\t indx=%d i=%d binary=%d%d%d"COLOR_RESET"\n",rank,send_cnt,mask,indx,i,binary[0],binary[1],binary[2]);
                    indx_pow *= 2;
                }

                indx += offset;
                ////////////////////////////////////////
                if (rank==prank) fprintf(stderr,"[%d]recvbuf[%d]:(%0.f)  %d , %d   (%d/%d)\n",rank,send_cnt,(float *) recvbuf +indx*seg_size*extent ,indx*seg_size,indx,i,Max_pow/lvl_pow);
                ////////////////////////////////////////
                // if (rank==prank) fprintf(stderr,COLOR_YELLOW"Rank=%d send_cnt=%d mask=%d\t indx=%d lvl=%d"COLOR_RESET"\n",rank,send_cnt,mask,indx,lvl);
                
                if (lvl>1){
                    
                    jump=0;
                    for (k=2; k<=lvl; k++){		
                        if (rank==prank) fprintf(stderr,COLOR_YELLOW"Rank=%d send_cnt=%d mask=%d\t tmp_buf_index=%d recvbuf_index=%d jump=%d k=%d"COLOR_RESET"\n",rank,send_cnt,mask,last_oprd*seg_size+jump,indx*seg_size,jump,k);
                        
                        //  MPIR_Reduce_local_impl(*inbuf, *inoutbuf,count, MPI_Datatype datatype, MPI_Op op);
                    
                        // mpi_errno = MPIR_Reduce_local_impl((char *) (tmp_buf+(last_oprd*seg_size+jump)*extent),(char *) (recvbuf+indx*seg_size*extent),seg_size,datatype,op);	

                        /* tmp_buf contains data received in this step.
                        * recvbuf contains data accumulated so far */

                        if (rank==prank) fprintf(stderr,COLOR_CYAN"[%d] Bfr Reduce send_cnt=%d  mask=%d\t tmp= %.0f recv= %.0f indx=%d lvl=%d"COLOR_RESET"\n",rank,send_cnt,mask, (float *)(tmp_buf+(last_oprd*seg_size+jump+1)*extent), (float *)recvbuf+(indx*seg_size+1)*extent,indx,lvl);
                        mpi_errno = MPIR_Reduce_local((char *) (tmp_buf+(last_oprd*seg_size+jump)*extent),(char *) (recvbuf+indx*seg_size*extent),seg_size,datatype,op);	
                        if (rank==prank) fprintf(stderr,COLOR_MAGENTA"[%d] Aft Reduce send_cnt=%d  mask=%d\t tmp= %.0f recv= %.0f indx=%d lvl=%d"COLOR_RESET"\n",rank,send_cnt,mask, (float *)(tmp_buf+(last_oprd*seg_size+jump+1)*extent), (float *)recvbuf+(indx*seg_size+1)*extent,indx,lvl);
                

                        if (k==2) base_jump=seg_size*(comm_size/4);	else base_jump = base_jump/2;
                        jump += base_jump;
                    }
                    
                    last_oprd++;
                }
                //printf("\n%c , %d", arr[indx],indx);
                
                    /* Send data from recvbuf. Recv into tmp_buf */ 
                
                // if (rank==prank) fprintf(stderr,COLOR_RED"Bfr Reduce Rank=%d send_cnt=%d tmp= %.1f recv= %.1f mask=%d indx=%d lvl=%d"COLOR_RESET"\n",rank,send_cnt,*(int *)tmp_buf+next_free*seg_size*extent,*(int *)recvbuf+indx*seg_size*extent,mask,indx,lvl);

                // if (rank==prank) fprintf(stderr,COLOR_RED"[%d] Bfr Reduce send_cnt=%d\t tmp= %.0f recv= %.0f mask=%d indx=%d lvl=%d"COLOR_RESET"\n",rank,send_cnt, (float *)tmp_buf+next_free*seg_size*extent, (float *)recvbuf+indx*seg_size*extent,mask,indx,lvl);


                /* Send data from recvbuf. Recv into tmp_buf */
                mpi_errno = MPIC_Sendrecv((char *) recvbuf +
                                        indx*seg_size*extent,
                                        seg_size, datatype,
                                        dst, MPIR_ALLREDUCE_TAG,
                                        (char *) tmp_buf +
                                        next_free*seg_size*extent,
                                        seg_size, datatype, dst,
                                        MPIR_ALLREDUCE_TAG, comm_ptr, MPI_STATUS_IGNORE, errflag);
                if (mpi_errno) {
                    /* for communication errors, just record the error but continue */
                    *errflag =
                        MPIX_ERR_PROC_FAILED ==
                        MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
                    MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                }

                // if (rank==prank) fprintf(stderr,COLOR_MAGENTA"[%d] Afr Reduce send_cnt=%d\t tmp= %.0f mask= %d indx=%d lvl=%d"COLOR_RESET"\n",rank,send_cnt,(float *)tmp_buf+next_free*seg_size*extent,mask,indx,lvl);
                            
                /*if (rank==prank) fprintf(stderr,"After  [%d] = {%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d}, nprocs=%d seg_size=%d \n", rank, 
                                *((int *) (recvbuf+MyIndex[0]*extent)) ,*((int *) (recvbuf+MyIndex[1]*extent)) ,
                                *((int *) (recvbuf+MyIndex[2]*extent)) ,*((int *) (recvbuf+MyIndex[3]*extent)) ,
                                *((int *) (recvbuf+MyIndex[4]*extent)) ,*((int *) (recvbuf+MyIndex[5]*extent)) ,
                                *((int *) (recvbuf+MyIndex[6]*extent)) ,*((int *) (recvbuf+MyIndex[7]*extent)) , 
                                *((int *) (recvbuf+MyIndex[8]*extent)) ,*((int *) (recvbuf+MyIndex[9]*extent)) ,
                                *((int *) (recvbuf+MyIndex[10]*extent)),*((int *) (recvbuf+MyIndex[11]*extent)) ,
                                *((int *) (recvbuf+MyIndex[12]*extent)),*((int *) (recvbuf+MyIndex[13]*extent)) ,
                                *((int *) (recvbuf+MyIndex[14]*extent)),*((int *) (recvbuf+MyIndex[15]*extent)) , 
                                nprocs, seg_size );
            */
            
                next_free++;
						
            }
            /* tmp_buf contains data received in this step.
             * recvbuf contains data accumulated so far */

            /* This algorithm is used only for predefined ops
             * and predefined ops are always commutative. */
            /* mpi_errno = MPIR_Reduce_local(((char *) tmp_buf + disps[recv_idx] * extent),
                                          ((char *) recvbuf + disps[recv_idx] * extent),
                                          recv_cnt, datatype, op);
            MPIR_ERR_CHECK(mpi_errno); */

            /* update send_idx for next iteration */
            lvl++;
            send_idx = recv_idx;
            mask <<= 1;

            /* update last_idx, but not in last iteration
             * because the value is needed in the allgather
             * step below. */
            if (mask < pof2)
                last_idx = recv_idx + pof2 / mask;
        }


        if (rank==prank) fprintf(stderr,"\n***********  Last Operation **********\n");

/*
        offset_index = ((binary[lvl-1]+1)%2);  

        lvl_pow = 1;		            
        for (i=1 ; i<=lvl ; i++){
            lvl_pow *= 2;
            int index = lvl-i-1;
            if (index>=0) 
                offset_index += binary[index] * lvl_pow;
        }
        //if (rank==prank) fprintf(stderr,"binary3=[%d][%d][%d]\n",binary[0],binary[1],binary[2]);
        offset = offset_index * (Max_pow/lvl_pow);
        
        if (rank==prank) fprintf(stderr,COLOR_YELLOW"LAST:[%d] send_cnt=%d mask=%d\t offset_index=%d offset=%d comm_size=%d Max_pow=%d lvl_pow=%d Max_lvl=%d lvl=%d binary=%d%d%d"COLOR_RESET"\n",rank,send_cnt,mask,offset_index, offset,comm_size,Max_pow,lvl_pow,Max_lvl,lvl,binary[0],binary[1],binary[2]);
        
        indx_pow=1;
        indx=0;
        i=0;
        for (j=Max_lvl-1 ; j>=lvl ; j--){			
            if (i%indx_pow==0) { 
                binary[j]=(binary[j]+1)%2; 
            }
            indx += binary[j]*indx_pow;
            // if (rank==prank) fprintf(stderr,COLOR_RED"Rank=%d send_cnt=%d mask=%d\t indx=%d i=%d binary=%d%d%d"COLOR_RESET"\n",rank,send_cnt,mask,indx,i,binary[0],binary[1],binary[2]);
            indx_pow *= 2;
        }

        indx += offset;

*/        
        //if (rank==prank) fprintf(stderr,COLOR_GREEN"LAST:[%d] send_cnt=%d mask=%d\t Max_lvl=%d lvl=%d binary=%d%d%d%d"COLOR_RESET"\n",rank,send_cnt,mask,Max_lvl,lvl,binary[3],binary[2],binary[1],binary[0]);
        
        indx_pow=1;
        indx=0;
        for (i=Max_lvl-1 ; i>=0 ; i--){
            indx += binary[i]*indx_pow;
            if (rank==prank) fprintf(stderr,COLOR_GREEN"LAST:[%d] indx=%d Max_lvl=%d binary[%d]=%d"COLOR_RESET"\n",rank,Max_lvl,i,binary[i]);
            indx_pow *= 2;            
        }


        jump=0;
        for (k=2; k<=lvl; k++){		
            if (rank==prank) fprintf(stderr,COLOR_YELLOW"LAST:[%d] indx=%d send_cnt=%d mask=%d\t tmp_buf_index=%d recvbuf_index=%d jump=%d k=%d"COLOR_RESET"\n",rank,indx,send_cnt,mask,last_oprd*seg_size+jump,indx*seg_size,jump,k);            
        
            
            if (rank==prank) fprintf(stderr,COLOR_CYAN"LAST:[%d] Bfr Reduce send_cnt=%d  mask=%d\t tmp= %.0f recv= %.0f indx=%d lvl=%d"COLOR_RESET"\n",rank,send_cnt,mask, (float *)(tmp_buf+(last_oprd*seg_size+jump+1)*extent), (float *)recvbuf+(indx*seg_size+1)*extent,indx,lvl);
            mpi_errno = MPIR_Reduce_local((char *) (tmp_buf+(last_oprd*seg_size+jump)*extent),(char *) (recvbuf+indx*seg_size*extent),seg_size,datatype,op);	
            if (rank==prank) fprintf(stderr,COLOR_MAGENTA"LAST:[%d] Aft Reduce send_cnt=%d  mask=%d\t tmp= %.0f recv= %.0f indx=%d lvl=%d"COLOR_RESET"\n",rank,send_cnt,mask, (float *)(tmp_buf+(last_oprd*seg_size+jump+1)*extent), (float *)recvbuf+(indx*seg_size+1)*extent,indx,lvl);
            if (k==2) base_jump=seg_size*(comm_size/4);	else base_jump = base_jump/2;
            jump += base_jump;
        }
        
     // if (rank==prank) fprintf(stderr,COLOR_CYAN"[%d] Bfr Reduce send_cnt=%d mask=%d tmp=%.0f recv=%.0f\t indx=%d lvl=%d"COLOR_RESET"\n",rank,send_cnt,mask,*(int *)recvbuf+next_free*seg_size*extent,recvbuf+indx*seg_size*extent,indx,lvl);
        
        
        /*        
        mpi_errno = MPIC_Sendrecv((char *) recvbuf + indx*seg_size*extent,
                                    //disps[send_idx]*extent,                                                 
                                    seg_size, datatype,  
                                    dst, MPIR_ALLREDUCE_TAG, 
                                    (char *) tmp_buf + next_free*seg_size*extent,
                                    //////disps[recv_idx]*extent,
                                    seg_size, datatype, dst,
                                    MPIR_ALLREDUCE_TAG, comm_ptr,
                                    MPI_STATUS_IGNORE, errflag);						
        
        // if (rank==prank) fprintf(stderr,COLOR_MAGENTA"After Reduce Rank=%d send_cnt=%d tmp=%d mask=%d\t indx=%d lvl=%d"COLOR_RESET"\n",rank,send_cnt,mask,*(int *)recvbuf+next_free*seg_size*extent,indx,lvl);
        lvl++;
         jump=0;
        for (k=2; k<=lvl; k++){		
            if (rank==prank) fprintf(stderr,COLOR_MAGENTA"Rank=%d send_cnt=%d mask=%d\t tmp_buf_index=%d recvbuf_index=%d jump=%d k=%d"COLOR_RESET"\n",rank,send_cnt,mask,last_oprd*seg_size+jump,indx*seg_size,jump,k);            
        
            
            if (rank==prank) fprintf(stderr,COLOR_CYAN"[%d] Bfr Reduce send_cnt=%d  mask=%d\t tmp= %.0f recv= %.0f indx=%d lvl=%d"COLOR_RESET"\n",rank,send_cnt,mask, (float *)(tmp_buf+(last_oprd*seg_size+jump)*extent), (float *)recvbuf+indx*seg_size*extent,indx,lvl);
            mpi_errno = MPIR_Reduce_local((char *) (tmp_buf+(last_oprd*seg_size+jump)*extent),(char *) (recvbuf+indx*seg_size*extent),seg_size,datatype,op);	
            if (rank==prank) fprintf(stderr,COLOR_MAGENTA"[%d] Aft Reduce send_cnt=%d  mask=%d\t tmp= %.0f recv= %.0f indx=%d lvl=%d"COLOR_RESET"\n",rank,send_cnt,mask, (float *)(tmp_buf+(last_oprd*seg_size+jump)*extent), (float *)recvbuf+indx*seg_size*extent,indx,lvl);
            if (k==2) jump=seg_size*(comm_size/4);	else jump += jump/2;
        }
        */

        if (rank==prank) {
            fprintf(stderr,"F-recvbuf = {%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f} [%d] nprocs=%d seg_size=%d \n", 
                *((float *) (recvbuf+MyIndex[0]*extent)) ,*((float *) (recvbuf+MyIndex[1]*extent)) ,
                *((float *) (recvbuf+MyIndex[2]*extent)) ,*((float *) (recvbuf+MyIndex[3]*extent)) ,
                *((float *) (recvbuf+MyIndex[4]*extent)) ,*((float *) (recvbuf+MyIndex[5]*extent)) ,
                *((float *) (recvbuf+MyIndex[6]*extent)) ,*((float *) (recvbuf+MyIndex[7]*extent)) , 
                *((float *) (recvbuf+MyIndex[8]*extent)) ,*((float *) (recvbuf+MyIndex[9]*extent)) ,
                *((float *) (recvbuf+MyIndex[10]*extent)),*((float *) (recvbuf+MyIndex[11]*extent)) ,
                *((float *) (recvbuf+MyIndex[12]*extent)),*((float *) (recvbuf+MyIndex[13]*extent)) ,
                *((float *) (recvbuf+MyIndex[14]*extent)),*((float *) (recvbuf+MyIndex[15]*extent)) , 
                rank, nprocs, seg_size );

            fprintf(stderr,"F-Tempbuf = {%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f} [%d] nprocs=%d seg_size=%d \n", 
                *((float *) (tmp_buf+MyIndex[0]*extent)) ,*((float *) (tmp_buf+MyIndex[1]*extent)) ,
                *((float *) (tmp_buf+MyIndex[2]*extent)) ,*((float *) (tmp_buf+MyIndex[3]*extent)) ,
                *((float *) (tmp_buf+MyIndex[4]*extent)) ,*((float *) (tmp_buf+MyIndex[5]*extent)) ,
                *((float *) (tmp_buf+MyIndex[6]*extent)) ,*((float *) (tmp_buf+MyIndex[7]*extent)) , 
                *((float *) (tmp_buf+MyIndex[8]*extent)) ,*((float *) (tmp_buf+MyIndex[9]*extent)) ,
                *((float *) (tmp_buf+MyIndex[10]*extent)),*((float *) (tmp_buf+MyIndex[11]*extent)) ,
                *((float *) (tmp_buf+MyIndex[12]*extent)),*((float *) (tmp_buf+MyIndex[13]*extent)) ,
                *((float *) (tmp_buf+MyIndex[14]*extent)),*((float *) (tmp_buf+MyIndex[15]*extent)) , 
                rank, nprocs, seg_size );    

            }

        /* now do the allgather */

        mask >>= 1;
        while (mask > 0) {
            newdst = newrank ^ mask;
            /* find real rank of dest */
            dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

            send_cnt = recv_cnt = 0;
            if (newrank < newdst) {
                /* update last_idx except on first iteration */
                if (mask != pof2 / 2)
                    last_idx = last_idx + pof2 / (mask * 2);

                recv_idx = send_idx + pof2 / (mask * 2);
                for (i = send_idx; i < recv_idx; i++)
                    send_cnt += cnts[i];
                for (i = recv_idx; i < last_idx; i++)
                    recv_cnt += cnts[i];
            } else {
                recv_idx = send_idx - pof2 / (mask * 2);
                for (i = send_idx; i < last_idx; i++)
                    send_cnt += cnts[i];
                for (i = recv_idx; i < send_idx; i++)
                    recv_cnt += cnts[i];
            }

            mpi_errno = MPIC_Sendrecv((char *) recvbuf +
                                      disps[send_idx] * extent,
                                      send_cnt, datatype,
                                      dst, MPIR_ALLREDUCE_TAG,
                                      (char *) recvbuf +
                                      disps[recv_idx] * extent,
                                      recv_cnt, datatype, dst,
                                      MPIR_ALLREDUCE_TAG, comm_ptr, MPI_STATUS_IGNORE, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag =
                    MPIX_ERR_PROC_FAILED ==
                    MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
                MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }

            if (newrank > newdst)
                send_idx = recv_idx;

            mask >>= 1;
        }
    }
    /* In the non-power-of-two case, all odd-numbered
     * processes of rank < 2*rem send the result to
     * (rank-1), the ranks who didn't participate above. */
    if (rank < 2 * rem) {
        if (rank % 2)   /* odd */
            mpi_errno = MPIC_Send(recvbuf, count,
                                  datatype, rank - 1, MPIR_ALLREDUCE_TAG, comm_ptr, errflag);
        else    /* even */
            mpi_errno = MPIC_Recv(recvbuf, count,
                                  datatype, rank + 1,
                                  MPIR_ALLREDUCE_TAG, comm_ptr, MPI_STATUS_IGNORE, errflag);
        if (mpi_errno) {
            /* for communication errors, just record the error but continue */
            *errflag =
                MPIX_ERR_PROC_FAILED ==
                MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
            MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
    }
  fn_exit:
    MPIR_CHKLMEM_FREEALL();
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}



#endif