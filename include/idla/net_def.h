#ifndef IDLA__NET_DEF_H_
#define IDLA__NET_DEF_H_

#include "input_pair.h"
#include "multiclass_less.h"
#include "reinterpret.h"
#include "xnbhd_diff.h"
#include <dlib/dnn.h>

template <
    long num_filters,
    long nr,
    long nc,
    int stride_y,
    int stride_x,
    typename SUBNET
    >
using con_nopad = dlib::add_layer<dlib::con_<num_filters,nr,nc,stride_y,stride_x,0,0>, SUBNET>;

template <long N, template <typename> class BN, long shape, long stride, typename SUBNET>
using idla_block = dlib::relu<BN<con_nopad<N, shape, shape, stride, stride, SUBNET>>>;

template <template <typename> class BN_CON, template <typename> class BN_FC>
using modified_idla = dlib::fc<2,
                      dlib::relu<BN_FC<dlib::fc<500,reinterpret<2,
                      dlib::max_pool<2,2,2,2,idla_block<25,BN_CON,3,1,
                      idla_block<25,BN_CON,5,5, // patch summary
                      dlib::relu<xnbhd_diff<5,5,
                      dlib::max_pool<2,2,2,2,idla_block<25,BN_CON,3,1,idla_block<25,BN_CON,3,1,
                      dlib::max_pool<2,2,2,2,idla_block<20,BN_CON,3,1,idla_block<20,BN_CON,3,1,
                      input_rgb_image_pair
                      >>>>>>>>>>>>>>>>;

using train_net_type = loss_multiclass_log_lr<modified_idla<dlib::bn_con, dlib::bn_fc>>;
using infer_net_type = dlib::softmax<modified_idla<dlib::affine, dlib::affine>>;

#endif // IDLA__NET_DEF_H_
