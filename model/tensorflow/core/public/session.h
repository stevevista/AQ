#pragma once

#include "../../../model.h"
#include "../../../graph.pb.h"
#include <iostream>
#include <sstream>


namespace tensorflow {

using namespace dlib;

// stubs
struct Session {
    void Create(GraphDef&);
    const dlib::tensor& run(const dlib::tensor&, double tmp = 1);
    std::shared_ptr<::alphago::net_type> policy_net;
    std::shared_ptr<::alphago::vnet_type> value_net;
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

