#ifndef PREPROCESS_H
#define PREPROCESS_H
#include <stdio.h>
#include "params.h"

struct float4
{
    float x;
    float y;
    float z;
    float r;
    float4(float _x, float _y, float _z, float _r):x(_x),y(_y),z(_z),r(_r){};
};

void generateVoxels(float *points, size_t points_size, const Params& params, float* voxels, unsigned int* mask);
void generateBaseFeatures(float* voxels, unsigned int* mask, const Params& params, unsigned int *pillar_num, float* voxel_features, unsigned int* voxel_idxs, unsigned int* voxel_num);

#endif