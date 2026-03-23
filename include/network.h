#ifndef IDLA__NETWORK_H_
#define IDLA__NETWORK_H_

#include <dlib/dnn.h>
#include "difference.h"
#include "input.h"
#include "multiclass_loss.h"
#include "reinterpret.h"

// ---------------------------------------------------------------------------

template <long N, template <typename> class BN, long shape, long stride, typename SUBNET>
using block = dlib::relu<BN<dlib::con<N, shape, shape, stride, stride, SUBNET>>>;

template <template <typename> class BN_CON, template <typename> class BN_FC>
using mod_idla = loss_multiclass_log_lr<dlib::fc<2,
                                           BN_FC<dlib::fc<500,reinterpret<2,
                                           dlib::max_pool<2,2,2,2,block<25,BN_CON,3,1,
                                           block<25,BN_CON,5,5, // Patch Summary Features
                                           cross_neighborhood_differences<5,5,
                                           dlib::max_pool<2,2,2,2,block<25,BN_CON,3,1,
                                           block<25,BN_CON,3,1,
                                           dlib::max_pool<2,2,2,2,block<20,BN_CON,3,1,
                                           block<20,BN_CON,3,1,
                                           input_rgb_image_pair>
                                           >>>>>>>>>>>>>>;

using net_type = mod_idla<dlib::bn_con, dlib::bn_fc>;    // Training Net
using anet_type = mod_idla<dlib::affine, dlib::affine>;  // Testing Net

#endif // IDLA__NETWORK_H_
