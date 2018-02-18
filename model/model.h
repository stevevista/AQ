#pragma once

#include "dnn/layers.h"
#include "dnn/core.h"


using namespace dlib;

namespace alphago {

template <int N, typename SUBNET> 
using conv2d = relu<con_bias<N,3,3,1,1,SUBNET>>;

template <typename SUBNET>
using res =     add_prev1<
                conv2d<192,
                conv2d<192,
                conv2d<192,
                tag1<SUBNET>
                >>>>;

using net_type =    softmax<
                    fc<361,
                    con<1,1,1,1,1,
                    repeat<7, res,
                    con_bias<192,5,5,1,1, 
                    input<matrix<unsigned char>>
                    >>>>>;


using vnet_type =   htan<fc<1,
                    relu<fc<256,
                    avg_pool<5,5,5,5,2,2,
                    add_prev3<
                    conv2d<936,
                    conv2d<808,
                    conv2d<680,
                    tag3<
                    avg_pool<2,2,2,2,0,0,
                    add_prev2<
                    conv2d<552,
                    conv2d<480,
                    tag2<
                    add_prev1<
                    conv2d<408,
                    conv2d<336,
                    conv2d<264,
                    tag1<
                    avg_pool<2,2,2,2,1,1,
                    repeat<3, res,
                    con_bias<192,5,5,1,1, 
                    input<matrix<unsigned char>>
                    >>>>>>>>>>>>>>>>>>>>>>>;
}

