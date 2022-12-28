#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <memory>
#include <cstring>
#include "preprocess.h"
#include "postprocess.h"

int loadData(const char *file, void **data, unsigned int *length)
{
  std::fstream dataFile(file, std::ifstream::in);

  if (!dataFile.is_open())
  {
	  std::cout << "Can't open files: "<< file<<std::endl;
	  return -1;
  }

  //get length of file:
  unsigned int len = 0;
  dataFile.seekg (0, dataFile.end);
  len = dataFile.tellg();
  dataFile.seekg (0, dataFile.beg);

  //allocate memory:
  char *buffer = new char[len];
  if(buffer==NULL) {
	  std::cout << "Can't malloc buffer."<<std::endl;
    dataFile.close();
	  exit(-1);
  }

  //read data as a block:
  dataFile.read(buffer, len);
  dataFile.close();

  *data = (void*)buffer;
  *length = len;
  return 0;  
}

void readAnchors(const char* file, float* anchors)
{
  std::ifstream in(file);
  std::string line;
  int i=0;
  while(std::getline(in,line))
  {
    std::stringstream ss(line);
    ss >> anchors[i];
    i++;
  }
}

int main(int argc, char** argv)
{
    std::string dataFile = "../../data/000000.bin";

    std::cout << "load file: "<< dataFile <<std::endl;

    //load points cloud
    unsigned int length = 0;
    void *data = NULL;
    std::shared_ptr<char> buffer((char *)data, std::default_delete<char[]>());
    loadData(dataFile.data(), &data, &length);
    buffer.reset((char *)data);

    float* points = (float*)buffer.get();
    size_t points_size = length/sizeof(float)/4; //点的数量，每个点(x,y,z,r)

    std::cout << "find points num: "<< points_size <<std::endl;

    Params params_;

    unsigned int anchors_size = params_.feature_x_size*params_.feature_y_size*6*7*sizeof(float);
    float* anchors = (float*)malloc(anchors_size);
    memset(anchors,0,anchors_size);
    std::string anchorFile = "../anchors.txt";
    readAnchors(anchorFile.data(), anchors);
    // for (size_t i = 0; i < 10; i++)
    // {
    //   std::cout<<anchors[i]<<std::endl;
    // }

    unsigned int voxel_count_data_size = MAX_VOXELS * params_.max_num_points_per_pillar * sizeof(unsigned int);
    unsigned int* voxel_count_data = (unsigned int*)malloc(voxel_count_data_size);
    memset(voxel_count_data,0,voxel_count_data_size);
    for (size_t i = 0; i < MAX_VOXELS; i++)
    {
      for(int j=0;j<params_.max_num_points_per_pillar;j++)
      {
        voxel_count_data[i*params_.max_num_points_per_pillar+j]=j;
      }
    }
    
    

    unsigned int voxels_size = params_.grid_z_size
          * params_.grid_y_size
          * params_.grid_x_size
          * params_.max_num_points_per_pillar //32
          * params_.num_point_values //4
          * sizeof(float); //27426816*4
    
    // unsigned int voxel_idx_size = params_.grid_z_size
    //           * params_.grid_y_size
    //           * params_.grid_x_size
    //           * 4
    //           * sizeof(unsigned int);//857088*4
    
    unsigned int mask_size = params_.grid_z_size
              * params_.grid_y_size
              * params_.grid_x_size
              * sizeof(unsigned int); //214272*4
    
    float* voxels = (float*)malloc(voxels_size);
    unsigned int* mask =(unsigned int*)malloc(mask_size);

    memset(voxels,0,voxels_size);
    memset(mask,0,mask_size);

    generateVoxels(points,points_size,params_,voxels,mask);

    //
    unsigned int voxel_features_size =  MAX_VOXELS * params_.max_num_points_per_pillar * 4 * sizeof(float);
    unsigned int voxel_idx_size = MAX_VOXELS * 4 * sizeof(unsigned int);
    unsigned int voxel_num_size = MAX_VOXELS * sizeof(unsigned int);

    float* voxel_features = (float*)malloc(voxel_features_size);
    unsigned int* voxel_idxs=(unsigned int*)malloc(voxel_idx_size);
    unsigned int* voxel_num=(unsigned int*)malloc(voxel_num_size);
    
    memset(voxel_features,0,voxel_features_size);
    memset(voxel_idxs,0,voxel_idx_size);
    memset(voxel_num,0,voxel_num_size);
    
    unsigned int pillar_num=0;
    generateBaseFeatures(voxels,mask,params_,&pillar_num,voxel_features,voxel_idxs,voxel_num);

    // for (size_t i = 0; i < 10000; i+=4)
    // {
    //   // std::cout<<voxel_features[i]<<std::endl;
    //   std::cout<<voxel_idxs[i]<<" "<<voxel_idxs[i+1]<<" "<<voxel_idxs[i+2]<<" "<<voxel_idxs[i+3]<<std::endl;
    //   std::cout<<voxel_num[i]<<std::endl;
    // }
    
    std::vector<Bndbox> preds;
    //postprocess(cls_preds, box_preds,params,preds);
    std::vector<Bndbox> nms_pred;
    nms_cpu(preds, params_.nms_thresh, nms_pred);


    free(voxel_features);
    free(voxel_idxs);
    free(voxel_num);
    free(voxels);
    free(mask);
    free(anchors);
    free(voxel_count_data);
    return 0;
}