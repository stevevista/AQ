#pragma once

#include <bitset>
#include <array>


#include "dnn/layers.h"
#include "dnn/core.h"
/*
// For windows support
#if defined(_MSC_VER) && !defined(__CUDACC__)
#ifdef gomodel_EXPORTS1
#define LMODEL_API __declspec(dllexport)
#else
#define LMODEL_API __declspec(dllimport)
#endif
#else
#define LMODEL_API
#endif
*/
#define LMODEL_API

namespace lightmodel {

using namespace dlib;


constexpr int board_size = 19;
constexpr int board_count = board_size*board_size;
constexpr int board_moves = board_count + 1;


template <int N, long kernel, typename SUBNET> using bn_conv2d = affine<con<N,kernel,kernel,1,1,SUBNET>>;

template <int N, typename SUBNET> 
using block  = bn_conv2d<N,3,relu<bn_conv2d<N,3,SUBNET>>>;

template <template <int,typename> class block, int N, typename SUBNET>
using residual = add_prev1<block<N,tag1<SUBNET>>>;

template <int classes, typename SUBNET>
using policy_head = softmax<fc<classes, relu<bn_conv2d<2, 1, SUBNET>>>>;

template <typename SUBNET>
using value_head = htan<fc<1, fc<256, relu<bn_conv2d<1, 1, SUBNET>>>>>;


namespace zero {

constexpr int RESIDUAL_FILTERS = 256;
constexpr int RESIDUAL_BLOCKS = 19;

template <typename SUBNET> 
using ares  = relu<residual<block, RESIDUAL_FILTERS, SUBNET>>;

using net_type = 
                            value_head<
                            skip1<
                            policy_head<board_moves,
                            tag1<
                            repeat<RESIDUAL_BLOCKS, ares,
                            relu<bn_conv2d<RESIDUAL_FILTERS,3,
                            input<matrix<unsigned char>>
                            >>>>>>>;

}

using zero_net_type = zero::net_type;

// leela model
namespace leela {
    
template <typename SUBNET> 
using ares  = relu<residual<block, 128, SUBNET>>;   
    
using net_type = 
                            value_head<
                            skip1<
                            policy_head<board_moves,
                            tag1<
                            repeat<6, ares,
                            relu<bn_conv2d<128,3,
                            input<matrix<unsigned char>>
                            >>>>>>>;
}

using leela_net_type = leela::net_type;

  
template <typename SUBNET> 
using convs  = relu<bn_conv2d<256,3,SUBNET>>;
                                
using dark_net_type =       fc_no_bias<1, // fake value head
                            softmax<
                            fc_no_bias<board_moves,
                            con_bias<1,1,1,1,1,
                            repeat<12, convs,
                            convs<
                            input<matrix<unsigned char>>
                            >>>>>>;
                                
                                


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




}
