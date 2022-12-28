#include <cstring>
#include <iostream>
#include "preprocess.h"
/*
points: 原始点数据，xyzr
points_size: 点个数
voxels: 每个voxel的点的数据
mask: 每个voxel的点的个数
*/
void generateVoxels(float *points, size_t points_size, const Params& params, float* voxels, unsigned int* mask)
{
    for (size_t point_idx = 0; point_idx < points_size; point_idx++)
    {
        float4 point(*(points+4*point_idx),*(points+4*point_idx+1),*(points+4*point_idx+2),*(points+4*point_idx+3));
        if (point.x<params.min_x_range || point.x>params.max_x_range
            || point.y<params.min_y_range || point.y>params.max_y_range
            || point.z<params.min_z_range || point.z>params.max_z_range)
        {
            continue;
        }
        int voxel_idx = (point.x - params.min_x_range)/params.pillar_x_size;
        int voxel_idy = (point.y - params.min_y_range)/params.pillar_y_size;
        unsigned int voxel_index = voxel_idy * params.grid_x_size + voxel_idx; //voxel索引

        unsigned int point_id = mask[voxel_index];
        if( point_id >= params.max_num_points_per_pillar)
            continue;
        mask[voxel_index]++;
        
        float *address = voxels + (voxel_index*params.max_num_points_per_pillar + point_id)*4;//voxel_index处的pillar中的point_id点
        *address=point.x;
        *(address+1)=point.y;
        *(address+2)=point.z;
        *(address+3)=point.r;
        // std::cout<<*(address)<<std::endl;
    }
}

/*
pillar_num：非空pillar数量
voxel_features：特征
voxel_idxs：非空pillar坐标，00yx
voxel_num: 非空pillar内点数
*/
void generateBaseFeatures(float* voxels, unsigned int* mask, const Params& params, unsigned int *pillar_num, float* voxel_features, unsigned int* voxel_idxs, unsigned int* voxel_num)
{
    for (size_t voxel_idy = 0; voxel_idy < params.grid_y_size; voxel_idy++)
    {
        for(size_t voxel_idx = 0; voxel_idx < params.grid_x_size; voxel_idx++)
        {
            unsigned int voxel_index = voxel_idy * params.grid_x_size + voxel_idx;
            // std::cout<<voxel_index<<std::endl;
            unsigned int count = mask[voxel_index]; //voxel内点个数
            if( !(count>0) ) continue;
            count = count<params.max_num_points_per_pillar?count:params.max_num_points_per_pillar;
            unsigned int current_pillarId = *pillar_num;
            (*pillar_num)++;
            voxel_num[current_pillarId] = count;
            voxel_idxs[4*current_pillarId] = 0; 
            voxel_idxs[4*current_pillarId+1] = 0;
            voxel_idxs[4*current_pillarId+2] = voxel_idy;
            voxel_idxs[4*current_pillarId+3] = voxel_idx;
            for (int i=0; i<count; i++)
            {
                int inIndex = voxel_index*params.max_num_points_per_pillar + i;
                int outIndex = current_pillarId*params.max_num_points_per_pillar + i;
                // std::cout<<*(voxels+4*inIndex)<<std::endl;
                memcpy(voxel_features+4*outIndex, voxels+4*inIndex,4*sizeof(float));//voxels数据复制给voxel_features
            }
            memset(mask+voxel_index, 0, sizeof(unsigned int));
        }
    }
    
}

