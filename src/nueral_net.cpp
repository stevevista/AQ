#include "nueral_net.h"

using namespace tensorflow;
using std::string;
using std::cerr;
using std::endl;

int cfg_sym_idx = 0;

#undef USE_52FEATURE

#ifdef USE_52FEATURE
	constexpr int feature_cnt = 52;
#else
	constexpr int feature_cnt = 49;
#endif


/**
 *  ����m����\������l�b�g���[�N
 *  Calculate probability distribution with the Policy Network.
 */
 void PolicyNet(Session* sess, std::vector<FeedTensor>& ft_list,
	std::vector<std::array<double,EBVCNT>>& prob_list,
	double temp, int sym_idx)
{

	prob_list.clear();
	int ft_cnt = (int)ft_list.size();
	resizable_tensor x;

	x.set_size(ft_cnt, feature_cnt, BSIZE, BSIZE);
	auto x_eigen = x.host_write_only();

	std::vector<int> sym_idxs;
	if(sym_idx > 7){
		for(int i=0;i<ft_cnt;++i){
			sym_idxs.push_back(mt_int8(mt_32));
		}
	}

	for(int i=0;i<ft_cnt;++i){
		for(int j=0;j<BVCNT;++j){
			for(int k=0;k<feature_cnt;++k){

				if(sym_idx == 0) x_eigen[i*feature_cnt*BVCNT + k*BVCNT + j] = ft_list[i].feature[j][k];
				else if(sym_idx > 7) x_eigen[i*feature_cnt*BVCNT + k*BVCNT + j] = ft_list[i].feature[sv.rv[sym_idxs[i]][j][0]][k];
				else x_eigen[i*feature_cnt*BVCNT + k*BVCNT + j] = ft_list[i].feature[sv.rv[sym_idx][j][0]][k];

			}
		}
	}

	auto& outputs = sess->run(x, temp);

	auto policy = outputs.host();

	for(int i=0;i<ft_cnt;++i){
		std::array<double, EBVCNT> prob;
		prob.fill(0.0);

		for(int j=0;j<BVCNT;++j){
			int v = rtoe[j];

			if(sym_idx == 0) prob[v] = (double)policy[i*BVCNT+j];
			else if(sym_idx > 7) prob[v] = (double)policy[i*BVCNT+sv.rv[sym_idxs[i]][j][1]];
			else prob[v] = (double)policy[i*BVCNT+sv.rv[sym_idx][j][1]];

			// 3����蒆�����̃V�`���E�𓦂����̊m�������
			// Reduce probability of moves escaping from Ladder.
			if(ft_list[i].feature[j][LADDERESC] != 0 && DistEdge(v) > 2) prob[v] *= 0.001;
		}
		prob_list.push_back(prob);
	}

}


/**
 *  �ǖʕ]���l��\������l�b�g���[�N
 *  Calculate value of the board with the Value Network.
 */
 void ValueNet(Session* sess, std::vector<FeedTensor>& ft_list,
	std::vector<float>& eval_list, int sym_idx)
{

	eval_list.clear();
	int ft_cnt = (int)ft_list.size();
	const int nk = feature_cnt+1;
	resizable_tensor x;

#if 1
	x.set_size(ft_cnt, nk, BSIZE, BSIZE);
	auto x_eigen = x.host_write_only();

	int sym_idx_rand = mt_int8(mt_32);

	for(int i=0;i<ft_cnt;++i){
		for(int j=0;j<BVCNT;++j){
			for(int k=0;k<feature_cnt;++k){
				if(sym_idx == 0) x_eigen[i*nk*BVCNT + k*BVCNT + j] = ft_list[i].feature[j][k];
				else if(sym_idx > 7) x_eigen[i*nk*BVCNT + k*BVCNT + j] = ft_list[i].feature[sv.rv[sym_idx_rand][j][0]][k];
				else x_eigen[i*nk*BVCNT + k*BVCNT + j] = ft_list[i].feature[sv.rv[sym_idx][j][0]][k];
			}
			x_eigen[i*nk*BVCNT + feature_cnt*BVCNT + j] = (float)ft_list[i].color;
		}
	}

	auto& outputs = sess->run(x);
	auto value = outputs.host();
	for(int i=0;i<ft_cnt;++i){
		eval_list.push_back((float)value[i]);
	}

#else 
	x.set_size(1, nk, BSIZE, BSIZE);
	auto x_eigen = x.host_write_only();

	int sym_idx_rand = mt_int8(mt_32);

	for(auto& ft : ft_list) {
		for(int j=0;j<BVCNT;++j){
			for(int k=0;k<feature_cnt;++k){
				if(sym_idx == 0) x_eigen[k*BVCNT + j] = ft.feature[j][k];
				else if(sym_idx > 7) x_eigen[k*BVCNT + j] = ft.feature[sv.rv[sym_idx_rand][j][0]][k];
				else x_eigen[k*BVCNT + j] = ft.feature[sv.rv[sym_idx][j][0]][k];
			}
			x_eigen[feature_cnt*BVCNT + j] = (float)ft.color;
		}

		auto& outputs = sess->run(x);
		auto value = outputs.host();

		eval_list.push_back((float)value[0]);
	}

#endif

}
