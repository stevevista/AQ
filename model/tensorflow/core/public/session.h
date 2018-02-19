#pragma once

#include "../../../model.h"
#include "../../../graph.pb.h"
#include <iostream>
#include <sstream>

#ifdef _WIN32
#include <windows.h>
#endif


namespace tensorflow {

using namespace dlib;

// stubs

using TensorShape = std::vector<int>;

class Tensor {
public:
    Tensor(Tensor&& other): x(std::move(other.x)), shape(std::move(other.shape)) {}
    Tensor(const Tensor& other) : x(other.x), shape(other.shape) {}
    Tensor(int, const TensorShape&);

    Tensor& operator=(Tensor&& other) {
        x = std::move(other.x);
        shape = std::move(other.shape);
        return *this;
    }

    Tensor& operator=(const Tensor& other) {
        x = other.x;
        shape = other.shape;
        return *this;
    }

    template<typename dtype>
    class data {
        Tensor* owner;
    public:
        data(Tensor* t) : owner(t) {}
        dtype& operator()() { return owner->x[0]; }
        dtype& operator()(int i, int j, int k) { 
            auto idx = i*owner->shape[1]*owner->shape[2]*owner->shape[3] + k*owner->shape[2]*owner->shape[3] + j;
            return owner->x[idx]; 
        }

        const dtype& operator()(int i) const { 
            return owner->x[i]; 
        }

        const dtype& operator()(int i, int j) const { 
            auto idx = i*owner->shape[1]*owner->shape[2]*owner->shape[3] + j;
            return owner->x[idx]; 
        }
    };

    template<typename dtype, int dim>
    data<dtype> tensor() {
        return data<dtype>(this);
    }

    template<typename dtype>
    data<dtype> scalar() {
        return data<dtype>(this);
    }

    template<typename dtype>
    data<dtype> matrix() {
        return data<dtype>(this);
    }

    std::vector<float> x;
    std::vector<int> shape;
};


struct Session {
    void Create(GraphDef&);
    const dlib::tensor& run(const dlib::tensor&, double tmp = 1);

    void Run(const std::vector<std::pair<std::string, Tensor>>& inputs, 
                const std::vector<std::string>& outnodes, 
                const std::vector<std::string>&, 
                std::vector<Tensor>* outputs);

    std::shared_ptr<::alphago::net_type> policy_net;
    std::shared_ptr<::alphago::vnet_type> value_net;

    resizable_tensor x;
    int device_id;
};

inline void* SessionOptions() { return nullptr; }
Session* NewSession(void*);

namespace Env {
	inline void* Default() { return nullptr; }
}

void ReadBinaryProto(void*, const std::string& path, GraphDef*);

namespace graph {
    void SetDefaultDevice(const std::string& device_name, GraphDef*);
}




}

