#ifndef POSTPROCESS_H_
#define POSTPROCESS_H_
#include <vector>
#include "params.h"

struct float2
{
    float x;
    float y;
    float2(){};
    float2(float _x, float _y):x(_x),y(_y){};
};

struct Bndbox {
    float x;
    float y;
    float z;
    float w;
    float l;
    float h;
    float rt; //direction
    int id; //class id
    float score;
    Bndbox(){};
    Bndbox(float x_, float y_, float z_, float w_, float l_, float h_, float rt_, int id_, float score_)
        : x(x_), y(y_), z(z_), w(w_), l(l_), h(h_), rt(rt_), id(id_), score(score_) {}
};

void postprocess(float* cls_preds, float* box_preds,const Params& params,std::vector<Bndbox>& preds);
int nms_cpu(std::vector<Bndbox> bndboxes, const float nms_thresh, std::vector<Bndbox> &nms_pred);

#endif