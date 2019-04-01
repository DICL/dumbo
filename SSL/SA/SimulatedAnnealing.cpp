#include<iostream>
#include<fstream>
#include<sstream>
#include<iomanip>
#include<string>
#include<vector>
#include<cstdlib>
#include<ctime>
#include<cmath>
#include<cassert>
#include<unistd.h>

#define NA 8
#define NC 10
#define NF 24

using namespace std;

class CONF {
public:
	int n_type, n_app, core_vm, n_vm, total_core;
	int *core_t, *node_t, *idlecore_t, **host;
	void init(CONF conf) {
		n_type = conf.n_type;
		n_app = conf.n_app;
		core_vm = conf.core_vm;
		n_vm = conf.n_vm;
		total_core = conf.total_core;
		core_t = new int[n_type];
		node_t = new int[n_type];
		idlecore_t = new int[n_type];
		host = new int*[n_type];

		for(int i=0; i<n_type; i++) {
			core_t[i] = conf.core_t[i];
			node_t[i] = conf.node_t[i];
			idlecore_t[i] = conf.idlecore_t[i];
			host[i] = new int[node_t[i]];
			for(int j=0; j<node_t[i]; j++) host[i][j] = 0;
		}
	}
};

CONF System;
vector<CONF> VC;
int ***mapping;
double factors[NA][NC][NF], srt[NA][NC];
int VCrank[NA][NC], Trank[NA][4];
int *Appidx;
double cf = 0.8;
string Appname[NA] = { "grepspark", "wcspark", "grephadoop", "teragen", "cg", "132", "lammps_5dhfr", "namd_5dhfr" };

// Generate VM placement for whole applications
bool InitialRandomSelect(vector<CONF> list, int* addr, int* x, int idx, int prev);
bool InitialSimpleSelect(vector<CONF> list, int* addr, int* x, int idx, int prev);
bool InitialFullSelect(vector<CONF> list, int* addr, int* x, int idx, int prev);
bool InitialQuasarSelect(vector<CONF> list, int* addr, int* x, int idx, int prev);
void ChangeSetting_old(int* addr);
bool AllocateVM(int* addr, int n);
void GenerateSample(int *addr, int n);

double GetResult();
void GetInput(char* setup, char* app);
void InitMapping();
void KeepBestSample();

void PrintList(vector<CONF> list);
void PrintElem(CONF elem);
void PrintMapping();

int main(int argc, char **argv) {
	int i, j, k;
	double temperature, threshold, thres;
	double metric, prev_m, delta, bc = 1.3806503e-23, rr;

	if(argc < 5) {
		cout << "Usage: " << argv[0] << " Initial_Temperature Threshold Cooling_factor Setupdata Appdata\n";
		return -1;
	}

	thres = atof(argv[2]);
	cf = atof(argv[3]);
	srand(time(NULL));
	GetInput(argv[4], argv[5]);
//	PrintList(VC);

	int *addr, *idlecore_t;
	addr = new int[System.n_app];
	idlecore_t = new int[System.n_type];
	for(i=0; i<System.n_app; i++) addr[i] = -1;
	for(i=0; i<System.n_type; i++) idlecore_t[i] = System.idlecore_t[i];

	for(j=1; j<=System.n_app; j++) {
//		temperature = atof(argv[1]) * (j + System.n_app) / System.n_app;
		temperature = atof(argv[1]) / (double)j;
		threshold = thres / (double)j;
/* For using SA */
		InitialRandomSelect(VC, addr, idlecore_t, j - 1, -1);
/* For using Simple or Static */
//		InitialSimpleSelect(VC, addr, idlecore_t, j - 1, -1);
//		InitialQuasarSelect(VC, addr, idlecore_t, j - 1, -1);
//		temperature = threshold;


//	PrintMapping();

		GenerateSample(addr, j);
		KeepBestSample();
		prev_m = GetResult();
		metric = prev_m;

		for(i=0; i<j; i++) {
		        PrintElem(VC[addr[i]]);
		        cout << "| ";
		}
		cout << "Initial, Geomean : ";
		cout << metric << endl;
//	PrintMapping();

		if(j == System.n_app) break;

//return 1;
		int *addr_new = new int[System.n_app];

//		for(k=0; k<VC.size(); k++) {
		while(temperature > threshold) {
			for(i=0; i<System.n_app; i++) addr_new[i] = addr[i];
			InitialRandomSelect(VC, addr_new, idlecore_t, j - 1, addr_new[j-1]);
//			if(InitialFullSelect(VC, addr_new, idlecore_t, j - 1, k) == false) continue;
			GenerateSample(addr_new, j);

//		if(AllocateVM(addr) == false) cout << "ALLOCATION FAILED!!\n";
			metric = GetResult();

			delta = metric - prev_m;
			rr = rand() / (double)RAND_MAX;
//cout << delta << " " << exp(delta / temperature) << " " << rr << endl;
//			if(exp(delta / (temperature * bc)) > rr) {
			if(exp(delta / temperature) > rr) {
				prev_m = metric;

		        	for(i=0; i<j; i++) {
					if(addr[i] != addr_new[i]) cout << "* ";
					PrintElem(VC[addr_new[i]]);
					if(addr[i] != addr_new[i]) cout << "* ";
					cout << "| ";
		        	}
				cout << "T " << temperature << ", Geomean : ";
				cout << metric << endl;
//			PrintMapping();

				for(i=0; i<System.n_app; i++) addr[i] = addr_new[i];
				KeepBestSample();
			}
			else {
		        	for(i=0; i<j; i++) {
					PrintElem(VC[addr[i]]);
					cout << "| ";
		        	}
				cout << "T " << temperature << ", Geomean : ";
				cout << prev_m << endl;
//			AllocateVM(addr);
//			PrintMapping();
			}
			
			temperature *= cf;
		}

		for(i=0; i<System.n_type; i++) idlecore_t[i] -= VC[addr[j-1]].idlecore_t[i];
	}
//	cout << "FINISHED\n";

	return 0;
}

bool AllocateVM(int* addr, int n) {
	int i, j, k, l, m, s, f;
	int *idx_t = new int[System.n_type], *keep_t = new int[System.n_type];
	int nvc[6];

	for(i=0; i<6; i++) nvc[i] = 0;
	for(i=0; i<n; i++)
		nvc[addr[i]]++;

	if(nvc[5] && nvc[4]) return false;
	if(n == System.n_app) {
		if(nvc[5] % 2) return false;
		if(nvc[4] % 2) return false;
	}

	for(i=0; i<System.n_type; i++) idx_t[i] = keep_t[i] = 0;
	InitMapping();

// I7 2222, NUMA 44
	for(i=0; i<n; i++) {
		if(addr[i] == 1) {
			if(idx_t[0] + 4 > System.node_t[0]) {
                	        if(idx_t[0] == keep_t[0]) return false;
                	        idx_t[0] = keep_t[0];
                	}
			for(j=0; j<4; j++) {
				for(k=0; k<System.core_t[0]; k++) mapping[0][idx_t[0]][k] = i;
				idx_t[0]++; keep_t[0]++;
			}
		} else if(addr[i] == 3) {
			if(idx_t[1] + 2 > System.node_t[1]) {
                	        if(idx_t[1] == keep_t[1]) return false;
                	        idx_t[1] = keep_t[1];
                	}
			for(j=0; j<2; j++) {
				for(k=0; k<System.core_t[1]; k++) mapping[1][idx_t[1]][k] = i;
				idx_t[1]++; keep_t[1]++;
			}
		}
	}

// NUMA 2222
	for(i=0; i<n; i++) {
		if(addr[i] != 2) continue;
		if(idx_t[1] + 4 > System.node_t[1]) {
			if(idx_t[1] == keep_t[1]) return false;
			idx_t[1] = keep_t[1];
		}

		for(j=0; j<4; j++) {
			for(s=0; s<System.core_t[1]; s++) if(mapping[1][idx_t[1]][s] == -1) break;
			f = s + 2;
			for(k=s; k<f; k++) mapping[1][idx_t[1]][k] = i;
			if(f == System.core_t[1]) keep_t[1]++;
			idx_t[1]++;
		}
	}

	idx_t[1] = keep_t[1];

// I7 22 NUMA 22
	for(i=0; i<n; i++) {
		if(addr[i] != 5) continue;
		if(idx_t[0] + 2 > System.node_t[0]) {
                        if(idx_t[0] == keep_t[0]) return false;
                        idx_t[0] = keep_t[0];
                }
		if(idx_t[1] + 2 > System.node_t[1]) {
                        if(idx_t[1] == keep_t[1]) return false;
                        idx_t[1] = keep_t[1];
                }

		for(j=0; j<2; j++) {
			for(k=0; k<System.core_t[0]; k++) mapping[0][idx_t[0]][k] = i;
			idx_t[0]++; keep_t[0]++;
		}
		for(j=0; j<2; j++) {
			for(s=0; s<System.core_t[1]; s++) if(mapping[1][idx_t[1]][s] == -1) break;
			f = s + 2;
			for(k=s; k<f; k++) mapping[1][idx_t[1]][k] = i;
			if(f == System.core_t[1]) keep_t[1]++;
			idx_t[1]++;
		}
	}

// I7 11111111
	for(i=0; i<n; i++) {
		if(addr[i] != 0) continue;
		if(idx_t[0] + 8 > System.node_t[0]) {
			if(idx_t[0] == keep_t[0]) return false;
			idx_t[0] = keep_t[0];
		}

		for(j=0; j<8; j++) {
			for(s=0; s<System.core_t[0]; s++) if(mapping[0][idx_t[0]][s] == -1) break;
			f = s + 1;
			for(k=s; k<f; k++) mapping[0][idx_t[0]][k] = i;
			if(f == System.core_t[0]) keep_t[0]++;
			idx_t[0]++;
		}
	}

	for(i=0; i<System.n_type; i++) idx_t[i] = keep_t[i];

// I7 1111 NUMA 1111
	for(i=0; i<n; i++) {
		if(addr[i] != 4) continue;
		if(idx_t[0] + 4 > System.node_t[0]) {
			if(idx_t[0] == keep_t[0]) return false;
			idx_t[0] = keep_t[0];
		}
		if(idx_t[1] + 4 > System.node_t[1]) {
			if(idx_t[1] == keep_t[1]) return false;
			idx_t[1] = keep_t[1];
		}

		for(j=0; j<4; j++) {
			for(s=0; s<System.core_t[0]; s++) if(mapping[0][idx_t[0]][s] == -1) break;
			f = s + 1;
			for(k=s; k<f; k++) mapping[0][idx_t[0]][k] = i;
			if(f == System.core_t[0]) keep_t[0]++;
			idx_t[0]++;
		}

		for(j=0; j<4; j++) {
			for(s=0; s<System.core_t[1]; s++) if(mapping[1][idx_t[1]][s] == -1) break;
			f = s + 1;
			for(k=s; k<f; k++) mapping[1][idx_t[1]][k] = i;
			if(f == System.core_t[1]) keep_t[1]++;
			idx_t[1]++;
		}
		idx_t[0] = keep_t[0];
	}

	for(i=0; i<n; i++) {
		int crs[3], ncr = 0;
		for(j=0; j<System.n_type; j++) {
			for(k=0; k<System.node_t[j]; k++) {
				bool here = true;
				for(l=0; l<System.core_t[j]; l++)
					if(mapping[j][k][l] == i) { here = false; break; }
				if(here) continue;

				for(l=0; l<System.core_t[j]; l++) 
					if(mapping[j][k][l] != i && mapping[j][k][l] != -1) {
						for(m=0; m<ncr; m++) if(crs[m] == mapping[j][k][l]) break;
						if(m == ncr) {
							if(ncr == 3) return false;
							crs[ncr++] = mapping[j][k][l];
						}
					}
			}
		}
	}

	return true;
}

bool InitialQuasarSelect(vector<CONF> list, int* addr, int* idlecore_t, int idx, int prev) {
	int i ,j;
	bool res;
	vector<int> list2, list3;

	for(i=0; i<list.size(); i++) 
		list2.push_back(VCrank[Appidx[idx]][i]);

	i = j = 0;
	while(list2.empty() == false) {
		if(list[list2[i]].idlecore_t[Trank[Appidx[idx]][j]] > 0) {
			list3.push_back(list2[i]);
			list2.erase(list2.begin() + i);
		}
		else i++;
		if(i == list2.size()) {
			j++;
			i = 0;
		}
	}

//	for(i=0; i<list.size(); i++) cout << list3[i] << " ";
//	cout << endl;

	while(1) {
		for(j=0; j<System.n_type; j++) {
			if(idlecore_t[j] < list[list3[0]].idlecore_t[j]) break;
		}
		if(j < System.n_type) { list3.erase(list3.begin()); continue; }
//cout << idx << " " << i << " " << idlecore_t[0] << " " << idlecore_t[1] << endl;
//PrintMapping();
		addr[idx] = list3[0];
		if(list3[0] != prev) res = AllocateVM(addr, idx + 1);
		if(res) return true;
		addr[idx] = -1;
		list3.erase(list3.begin());
	}

	return false;
}

bool InitialSimpleSelect(vector<CONF> list, int* addr, int* idlecore_t, int idx, int prev) {
	int i ,j;
	bool res;
	vector<int> list2;

	for(i=0; i<list.size(); i++) 
/* For using Static */
//		if(VCrank[Appidx[idx]][i] == 1 || VCrank[Appidx[idx]][i] == 3)
		list2.push_back(VCrank[Appidx[idx]][i]);

	while(1) {
		for(j=0; j<System.n_type; j++) {
			if(idlecore_t[j] < list[list2[0]].idlecore_t[j]) break;
		}
		if(j < System.n_type) { list2.erase(list2.begin()); continue; }
//cout << idx << " " << i << " " << idlecore_t[0] << " " << idlecore_t[1] << endl;
//PrintMapping();
		addr[idx] = list2[0];
		if(list2[0] != prev) res = AllocateVM(addr, idx + 1);
		if(res) return true;
		addr[idx] = -1;
		list2.erase(list2.begin());
	}

	return false;
}

bool InitialFullSelect(vector<CONF> list, int* addr, int* idlecore_t, int idx, int choose) {
	int i ,j;
	bool res;

	while(1) {
		i = choose;
		for(j=0; j<System.n_type; j++) {
			if(idlecore_t[j] < list[i].idlecore_t[j]) break;
		}
		if(j < System.n_type) break; 
		addr[idx] = i;
		res = AllocateVM(addr, idx + 1);
		if(res) return true;
		addr[idx] = -1;
		break;
	}

	return false;
}

bool InitialRandomSelect(vector<CONF> list, int* addr, int* idlecore_t, int idx, int prev) {
	int i ,j;
	bool res;
	vector<int> list2;
	for(i=0; i<list.size(); i++) list2.push_back(i);

	while(1) {
		i = rand() % list2.size();
		for(j=0; j<System.n_type; j++) {
			if(idlecore_t[j] < list[list2[i]].idlecore_t[j]) break;
		}
		if(j < System.n_type) { list2.erase(list2.begin() + i); continue; }
//cout << idx << " " << i << " " << idlecore_t[0] << " " << idlecore_t[1] << endl;
//PrintMapping();
		addr[idx] = list2[i];
		if(list2[i] != prev) res = AllocateVM(addr, idx + 1);
		if(res) return true;
		addr[idx] = -1;
		list2.erase(list2.begin() + i);
	}

	return false;
}

void GenerateSample(int *addr, int n) {
	int i, j, k, l, m;
	int value[VC.size()];
	ofstream out, out2;
	string filename;

	for(i=0; i<VC.size(); i++) value[i] = 0;
	for(i=0; i<n; i++) value[addr[i]]++;

	out2.open("Corunners");

	for(i=0; i<n; i++) {
		stringstream ss;
		ss << "App" << i << "_";
		ss >> filename;
		filename = filename + Appname[Appidx[i]] + ".sample";

		out.open(filename.c_str());

		out << VC[addr[i]].idlecore_t[0] << ",";
		for(j=0; j<System.n_type; j++) {
			int node = 0;
			for(k=0; k<System.node_t[j]; k++) if(VC[addr[i]].host[j][k]) node++;
			out << node << ",";
		}

		int crs[3], vcr[3], ncr = 0;
		int blk = 0, nvm = 0, weight[2][3], bapp[2][3], changed = false;
		for(j=0; j<System.n_type; j++) {
			for(k=0; k<System.node_t[j]; k++) {
				bool here = true;
				for(l=0; l<System.core_t[j]; l++)
					if(mapping[j][k][l] == i) { here = false; break; }
				if(here) continue;

				for(l=0; l<System.core_t[j]; l++) 
					if(mapping[j][k][l] != i && mapping[j][k][l] != -1) {
						for(m=0; m<ncr; m++) if(crs[m] == mapping[j][k][l]) break;
						if(m == ncr) {
							assert(ncr < 3);
							crs[ncr++] = mapping[j][k][l];
						}
					}
			}
		}

		out2 << Appname[Appidx[i]] << " " << addr[i];
		for(j=0; j<ncr; j++) out2 << " " << Appname[Appidx[crs[j]]] << " " << addr[crs[j]];
		out2 << endl;

		for(j=0; j<3; j++) vcr[j] = 0;
		for(j=0; j<System.n_type; j++) {
			int node = 0;
			for(k=0; k<System.node_t[j]; k++) {
				bool here = true;
				for(l=0; l<System.core_t[j]; l++) {
					if(mapping[j][k][l] == i) { nvm++; here = false; changed = true;}
				}
				if(here) continue;
				node++;

				for(l=0; l<System.core_t[j]; l++) {
					if(mapping[j][k][l] != i && mapping[j][k][l] != -1) {
						for(m=0; m<ncr; m++) if(crs[m] == mapping[j][k][l]) break;
						if(m == ncr) continue;
						vcr[m]++;
					}
				}
				if(changed && nvm % 4 == 0) {
					assert(blk < 2);
					for(l=0; l<3; l++) {
						weight[blk][l] = vcr[l] / node;
						bapp[blk][l] = crs[l];
						vcr[l] = 0;
					}
					blk++;
					changed = false;
					node = 0;
				}
			}
			assert(nvm % 4 == 0);
		}

		for(j=0; j<2; j++) 
			for(k=0; k<3; k++)
				for(l=k+1; l<3; l++) 
					if(weight[j][k] < weight[j][l]) {
						m = weight[j][k]; weight[j][k] = weight[j][l]; weight[j][l] = m;
						m = bapp[j][k]; bapp[j][k] = bapp[j][l]; bapp[j][l] = m;
					}

		if(value[5] == 2 && value[2] == 1 && value[1] == 1 && addr[i] == 2)
			for(j=0; j<3; j++) {
				k = weight[0][j]; weight[0][j] = weight[1][j]; weight[1][j] = k;
				k = bapp[0][j]; bapp[0][j] = bapp[1][j]; bapp[1][j] = k;
			}

		for(j=0; j<2; j++)
			for(k=0; k<3; k++) out << weight[j][k] << ",";

		if(addr[i] == 3) k = 4;
		else if(addr[i] == 4 || addr[i] == 0) k = 1;
		else k = 2;
		for(j=0; j<NF; j++)
			out << factors[Appidx[i]][addr[i]][j] * k << ",";

		for(j=0; j<2; j++)
			for(k=0; k<3; k++) {
				if(weight[j][k] == 0) {
					for(l=0; l<NF; l++) out << "0,";
					continue;
				}
				int idx = addr[bapp[j][k]];
				if(idx >= 4) idx = (idx - 3) * 2 + 4 + j;
				for(l=0; l<NF; l++)
					out << factors[Appidx[bapp[j][k]]][idx][l] * weight[j][k] << ",";
			}
				
		out << srt[Appidx[i]][addr[i]] << ",0,0" << endl;

		out.close();
	}

	out2.close();
}

void GetInput(char* setup, char* app) {
	int size_vc;
	int i, j, k, temp;
	ifstream in;
	string filename;

	in.open(setup);

	in >> System.n_type >> System.n_app >> System.core_vm;
	System.core_t = new int[System.n_type];
	System.node_t = new int[System.n_type];
	System.idlecore_t = new int[System.n_type];
	System.host = new int*[System.n_type];
	System.total_core = 0;

	for(i=0; i<System.n_type; i++) {
		in >> System.node_t[i] >> System.core_t[i];
		System.host[i] = new int[System.node_t[i]];
		System.core_t[i] /= System.core_vm;
		System.idlecore_t[i] = System.node_t[i] * System.core_t[i];
		System.total_core += System.node_t[i] * System.core_t[i];
	}

	System.n_vm = System.total_core / System.n_app;

	in >> size_vc;
        for(i=0; i<size_vc; i++) {
                CONF vc;
                vc.init(System);
                for(j=0; j<vc.n_type; j++) in >> vc.idlecore_t[j];;
//			vc.idlecore_t[j] -= temp;

                for(j=0; j<vc.n_type; j++) {
                        int temp, temp2;
                        in >> temp;
                        if(temp == 0) continue;
//                        temp2 = (vc.core_t[j] * vc.node_t[j] - vc.idlecore_t[j]) / temp;
                        temp2 = vc.idlecore_t[j] / temp;
                        for(k=0; k<temp; k++) vc.host[j][k] = temp2;
                }
                VC.push_back(vc);
        }

	in.close();
	in.open(app);

	Appidx = new int[System.n_app];
	for(i=0; i<System.n_app; i++)
		in >> Appidx[i];

	in.close();

	mapping = new int**[System.n_type];
	for(i=0; i<System.n_type; i++) {
		mapping[i] = new int*[System.node_t[i]];
		for(j=0; j<System.node_t[i]; j++) {
			mapping[i][j] = new int[System.core_t[i]];
			for(k=0; k<System.core_t[i]; k++) mapping[i][j][k] = -1;
		}
	}

	for(i=0; i<NA; i++) {
		filename = Appname[i] + ".dat";
		in.open(filename.c_str());

		for(j=0; j<NC; j++) {
			in >> srt[i][j];
			for(k=0; k<NF; k++) in >> factors[i][j][k];
		}
		in.close();

		for(j=0; j<VC.size(); j++) VCrank[i][j] = j;
		for(j=0; j<VC.size(); j++) for(k=j+1; k<VC.size(); k++) 
			if(srt[i][VCrank[i][j]] > srt[i][VCrank[i][k]])
				{ temp = VCrank[i][j]; VCrank[i][j] = VCrank[i][k]; VCrank[i][k] = temp; }

		for(j=0; j<System.n_type; j++) Trank[i][j] = 0;
		for(j=0; j<VC.size(); j++) 
			for(k=0; k<System.n_type; k++) if(VC[j].idlecore_t[k] == 8)
				{ Trank[i][k] += srt[i][j]; break; }

		int rank[4] = {0};
		for(j=0; j<System.n_type; j++) {
			int min = 1000000000, idx = -1;
			for(k=0; k<System.n_type; k++) if(min > Trank[i][k])
				{ min = Trank[i][k]; idx = k; }
			rank[j] = idx;
		}

		for(j=0; j<System.n_type; j++) Trank[i][j] = rank[j];
	}
}

double GetResult() {
	int pid;
	double res = 0;
	ifstream in;
	stringstream ss;
	string temp;
	ss << System.n_app;
	ss >> temp;

	pid = fork();
	assert(pid != -1);
	switch(pid) {
case 0:
		execl("./SAhelper.sh", "./SAhelper.sh", temp.c_str(), NULL);
//		execl("/home/gorae/NewVersion_170206/HelpST/SAhelper.sh", "/home/gorae/NewVersion_170206/HelpST/SAhelper.sh", temp.c_str(), NULL);
		exit(-1);
default:
		sleep(1);
		wait(pid);
//		while(1) {
//			in.open("Model_result", ios::in);
//			if(in.is_open()) break;
//		}
		in.open("Model_result", ios::in);
		in >> res;

		in.close();
//		remove("Model_result");
	}

	return res;
}

void KeepBestSample() {
	int i;
	ifstream in;
	ofstream out;
	string filename, dest, line;

	for(i=0; i<System.n_app; i++) {
		stringstream ss;
		ss << "App" << i << "_";
		ss >> filename;
		filename = filename + Appname[Appidx[i]] + ".sample";
		dest = "bestsample/" + filename;

		in.open(filename.c_str());
		out.open(dest.c_str());
		in >> line;
		out << line << endl;
		in.close();
		out.close();
	}

	in.open("Corunners");
	out.open("bestsample/Corunners");
	while(1) {
		getline(in, line);
		if(in.eof()) break;
		out << line << endl;
	}
	in.close();
	out.close();
}
void PrintList(vector<CONF> list) {
	int i, j, k;
	cout << "List size : " << list.size() << endl;
	for(i=0; i<list.size(); i++) {
		PrintElem(list[i]);
		cout << endl;
	}
}

void PrintElem(CONF elem) {
	int j, k;
	for(j=0; j<System.n_type; j++) {
		if(elem.host[j][0] == 0) continue;
		cout << "T" << j+1 << " ";
		for(k=0; k<System.node_t[j]; k++) {
			if(elem.host[j][k] == 0) break;
			cout << elem.host[j][k];
		}
		cout << " ";
//		cout << " / ";
	}
//	for(j=0; j<System.t; j++) cout << elem.x[j] << " ";
}

void InitMapping() {
	int i, j, k;
	for(i=0; i<System.n_type; i++)
		for(j=0; j<System.node_t[i]; j++)
			for(k=0; k<System.core_t[i]; k++)
				mapping[i][j][k] = -1;
}

void PrintMapping() {
	int i, j, k;
	for(i=0; i<System.n_type; i++) {
		for(j=0; j<System.core_t[i]; j++) {
			for(k=0; k<System.node_t[i]; k++) cout << setw(3) << mapping[i][k][j];
			cout << endl;
		}
		cout << endl;
	}
}
