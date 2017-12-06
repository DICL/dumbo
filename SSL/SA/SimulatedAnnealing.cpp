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
int *Appidx;
double cf = 0.8;
string Appname[NA] = { "grepspark", "wcspark", "grephadoop", "teragen", "cg", "132", "lammps_5dhfr", "namd_5dhfr" };

// Generate VM placement for whole applications
bool InitialRandomSelect(vector<CONF> list, int* addr, int* x, int idx);
bool CheckIfCombinationIsPossible(vector<CONF> list, int* addr, int t);
bool CheckIfCombinationIsPossible2(vector<CONF> list, int* addr, int* host, int t, int idx);
void PlaceVMBinPacking(vector<int*> allcase, int* host, int* conf, int* idx, int count, int t);

void ChangeSetting(int* addr);
bool AllocateVM(int* addr);
void GenerateSample(int *addr);

double GetResult();
void GetInput();
void InitMapping();
void KeepBestSample();

void PrintList(vector<CONF> list);
void PrintElem(CONF elem);
void PrintMapping();

int main(int argc, char **argv) {
	int i;
	double temperature, threshold;
	double metric, prev_m, delta, bc = 1.3806503e-23, rr;

	if(argc < 3) {
		cout << "Usage: " << argv[0] << " Initial_Temperature Threshold Cooling_factor\n";
		return -1;
	}

	temperature = atof(argv[1]);
	threshold = atof(argv[2]);
	cf = atof(argv[3]);
	srand(time(NULL));
	GetInput();
//	PrintList(VC);

	int *addr, *idlecore_t;
	addr = new int[System.n_app];
	idlecore_t = new int[System.n_type];
	for(i=0; i<System.n_app; i++) addr[i] = -1;
	for(i=0; i<System.n_type; i++) idlecore_t[i] = System.idlecore_t[i];

	InitialRandomSelect(VC, addr, idlecore_t, 0);

/*
	int test[][4] = {{0,0,2,2}, {0,0,3,3}, {1,1,2,2}, {1,1,3,3}, {0,4,4,2}, {1,4,4,2}, {1,5,5,2}, {1,5,5,3}, {4,4,4,4}, {5,5,5,5}};

	for(i=0; i<10; i++)
		if(AllocateVM(test[i])) { cout << "ALLOC " << i+1 << " PASSED\n"; PrintMapping(); }
*/

//	addr[0] = 0; addr[1] = 4; addr[2] = 2; addr[3] = 4;
//	addr[0] = 2; addr[1] = 5; addr[2] = 1; addr[3] = 5;
	AllocateVM(addr);
//	PrintMapping();
/*
	for(i=0; i<System.n_app; i++) addr[i] = 0;
	for(i=10; i<System.n_app; i++) addr[i] = 0;
	for(i=20; i<System.n_app; i++) addr[i] = 3;
	for(i=30; i<System.n_app; i++) addr[i] = 3;
	AllocateVM(addr);
*/
        for(i=0; i<System.n_app; i++) {
                PrintElem(VC[addr[i]]);
                cout << " | ";
        }
//	if(AllocateVM(addr) == false) cout << "ALLOCATION FAILED!!\n";
	GenerateSample(addr);
	KeepBestSample();
	prev_m = GetResult();
	metric = prev_m;
	cout << "Initial, Geomean : ";
	cout << metric << endl;
//	PrintMapping();

//return 1;
	int *addr_new = new int[System.n_app];

	while(temperature > threshold) {
		for(i=0; i<System.n_app; i++) addr_new[i] = addr[i];
		ChangeSetting(addr_new);
		GenerateSample(addr_new);

//		if(AllocateVM(addr) == false) cout << "ALLOCATION FAILED!!\n";
		metric = GetResult();

		delta = metric - prev_m;
		rr = rand() / (double)RAND_MAX;
//cout << delta << " " << exp(delta / temperature) << " " << rr << endl;
//		if(exp(delta / (temperature * bc)) > rr) {
		if(exp(delta / temperature) > rr) {
			prev_m = metric;

	        	for(i=0; i<System.n_app; i++) {
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
	        	for(i=0; i<System.n_app; i++) {
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

//	cout << "FINISHED\n";

	return 0;
}

bool AllocateVM(int* addr) {
	int i, j, k, l, m, s, f;
	int *idx_t = new int[System.n_type], *keep_t = new int[System.n_type];
	int nvc[6];

	for(i=0; i<6; i++) nvc[i] = 0;
	for(i=0; i<System.n_app; i++)
		nvc[addr[i]]++;

	if(nvc[5] % 2) return false;
	if(nvc[4] % 2) return false;
	if(nvc[5] && nvc[4]) return false;

	for(i=0; i<System.n_type; i++) idx_t[i] = keep_t[i] = 0;
	InitMapping();

// I7 2222, NUMA 44
	for(i=0; i<System.n_app; i++) {
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
	for(i=0; i<System.n_app; i++) {
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
	for(i=0; i<System.n_app; i++) {
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
	for(i=0; i<System.n_app; i++) {
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
	for(i=0; i<System.n_app; i++) {
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

	for(i=0; i<System.n_app; i++) {
		int crs[3], ncr = 0;
		for(j=0; j<System.n_type; j++) {
			for(k=0; k<System.node_t[j]; k++) {
				bool here = true;
				for(l=0; l<System.core_t[j]; l++)
					if(mapping[j][k][l] == i) { here = false; break; }
				if(here) continue;

				for(l=0; l<System.core_t[j]; l++) 
					if(mapping[j][k][l] != i) {
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

void ChangeSetting(int* addr) {
	int i, j, r1, r2, r3, save1, save2, temp;
	int *vm_t = new int[System.n_type];
	bool twozero = true;

	vector<int> target;
	for(i=0; i<System.n_app; i++) target.push_back(i);

	while(target.size() > 0) {
		r1 = rand() % target.size();
		temp = target[r1];
		target.erase(target.begin() + r1);
		r1 = temp;

		vector<int> conf_t;
		for(i=0; i<VC.size(); i++) {
			if(i == addr[r1]) continue;
			conf_t.push_back(i);
		}

		while(conf_t.size() > 0) {
			r2 = rand() % conf_t.size();
			temp = conf_t[r2];
			conf_t.erase(conf_t.begin() + r2);
			r2 = temp;

			twozero = true;
			for(i=0; i<System.n_type; i++) {
				vm_t[i] = VC[addr[r1]].idlecore_t[i] - VC[r2].idlecore_t[i];
				if(vm_t[i] != 0) twozero = false;
			}

			save1 = addr[r1];
			addr[r1] = r2;
			if(twozero && AllocateVM(addr)) return;

			vector<int> pos;
			for(i=0; i<System.n_app; i++) {
				if(i == r1) continue;
				for(j=0; j<System.n_type; j++) {
					int tmp = vm_t[j] + VC[addr[i]].idlecore_t[j];
					if(tmp > 8 || tmp < 0) break;
					if(twozero && tmp != VC[addr[r1]].idlecore_t[j]) break;
				}
				if(j == System.n_type) pos.push_back(i);
			}

			while(pos.size() > 0) {
				r3 = rand() % pos.size();
				temp = pos[r3];
				pos.erase(pos.begin() + r3);
				r3 = temp;

				save2 = addr[r3];

				vector<int> entry;
				for(i=0; i<VC.size(); i++) entry.push_back(i);

				for(i=0; i<VC.size(); i++) {
					int idx = rand() % entry.size();
					addr[r3] = entry[idx];

					if(Appidx[r3] == Appidx[r1] && addr[r3] == save1 && addr[r1] == save2) {
						entry.erase(entry.begin() + idx);
						continue;
					}
//			for(j=0; j<System.n_type; j++)
//				if(!CheckIfCombinationIsPossible(VC, addr, j)) break;
//			if(j == System.n_type) break;
					if(AllocateVM(addr)) return;

					entry.erase(entry.begin() + idx);
				}
	
				addr[r3] = save2;
			}
			addr[r1] = save1;
		}
	}
}

void PlaceVMBinPacking(vector<int*> *allcase, int* host, int* conf, int* idx, int count, int t) {
	int i, j, history = -1;

	if(count == System.node_t[t]) {
		int* newconf;
		int temp;
		newconf = new int[System.node_t[t]];
		for(i=0; i<System.node_t[t]; i++) newconf[i] = conf[idx[i]];
		for(i=0; i<allcase->size(); i++) {
			for(j=0; j<System.node_t[t]; j++) if(newconf[j] != allcase->at(i)[j]) break;
			if(j == System.node_t[t]) break;
		}
		if(i < allcase->size()) {
			delete newconf;
			return;
		}
//for(i=0; i<System.node_t[t]; i++) cout << newconf[i]; cout << endl;
		allcase->push_back(newconf);
	}

	for(i=0; i<System.node_t[t]; i++) {
		for(j=0; j<count; j++) if(idx[j] == i) break;
		if(j < count) continue;

		if(conf[i] + host[count] > System.core_t[t]) continue;
		if(history == conf[i]) continue;
		idx[count] = i;
		history = conf[i];
		host[count] += conf[i];
		PlaceVMBinPacking(allcase, host, conf, idx, count + 1, t);
		host[count] -= conf[i];
	}
}

bool CheckIfCombinationIsPossible2(vector<CONF> list, int* addr, int* host, int t, int idx) {
	int i, j, k, *conf, *loc, tmp, *orihost;
	bool res;
	vector<int*> allcase;

	if(idx == System.n_app) return true;

	orihost = new int[System.node_t[t]];
	conf = new int[System.node_t[t]];
	loc = new int[System.node_t[t]];

	for(i=0; i<System.node_t[t]; i++) {
		conf[i] = list[addr[idx]].host[t][i];
		loc[i] = 0;
	}
/*
for(i=0; i<System.n[t]; i++) cout << conf[i];
cout << endl;
for(i=0; i<System.n[t]; i++) cout << host[i];
cout << endl;
*/
	PlaceVMBinPacking(&allcase, host, conf, loc, 0, t);
//cout << allcase.size() << endl;
	for(i=0; i<allcase.size(); i++) {
//for(j=0; j<System.n[t]; j++) cout << allcase[i][j]; cout << endl;
		for(j=0; j<System.node_t[t]; j++) orihost[j] = host[j];
		for(j=0; j<System.node_t[t]; j++) host[j] += allcase[i][j];
// Sorting Part
		for(j=0; j<System.node_t[t]; j++) for(k=j+1; k<System.node_t[t]; k++)
			if(host[j] < host[k]) {tmp = host[j]; host[j] = host[k]; host[k] = tmp;}
		res = CheckIfCombinationIsPossible2(list, addr, host, t, idx + 1);
		if(res) {
			delete conf;
			delete loc;
			return true;
		}
		for(j=0; j<System.node_t[t]; j++) host[j] = orihost[j];
//		for(j=0; j<System.node_t[t]; j++) host[j] -= allcase[i][j];
	}

	for(i=allcase.size()-1 ; i>=0; i--) {
		int* temp = allcase[i];
		delete temp;
		allcase.pop_back();
	}

	delete orihost;	
	delete conf;
	delete loc;

	return false;
}

bool CheckIfCombinationIsPossible(vector<CONF> list, int* addr, int t) {
	int i, j, *host, *addr_val, *addr_new, temp;
	bool res;
	host = new int[System.node_t[t]];
//Sorting Part
	addr_val = new int[System.n_app];
	addr_new = new int[System.n_app];

	for(i=0; i<System.n_app; i++) { addr_val[i] = 0; addr_new[i] = addr[i]; }

	for(i=0; i<System.node_t[t]; i++) 
		for(j=0; j<System.n_app; j++) 
			addr_val[j] = addr_val[j] * System.core_t[t] + list[addr[j]].host[t][i];
	for(i=0; i<System.n_app; i++) for(j=i+1; j<System.n_app; j++) 
		if(addr_val[i] < addr_val[j]) {
			temp = addr_val[i]; addr_val[i] = addr_val[j]; addr_val[j] = temp;
			temp = addr_new[i]; addr_new[i] = addr_new[j]; addr_new[j] = temp;
		}

	for(i=0; i<System.node_t[t]; i++) host[i] = list[addr_new[0]].host[t][i];
	res = CheckIfCombinationIsPossible2(list, addr_new, host, t, 1);

//	for(i=0; i<System.node_t[t]; i++) host[i] = list[addr[0]].host[t][i];
//	res = CheckIfCombinationIsPossible2(list, addr, host, t, 1);

	delete addr_val;
	delete addr_new;
	delete host;
	return res;
}

bool InitialRandomSelect(vector<CONF> list, int* addr, int* idlecore_t, int idx) {
	int i ,j, c = 0;
	if(idx == System.n_app) {
//		for(i=0; i<System.n_type; i++) 
//			if(!CheckIfCombinationIsPossible(list, addr, i)) break;
//		if(i < System.n_type) return false;

		return AllocateVM(addr);
//		if(AllocateVM(addr) == false) { cout << "ALLOCATION FAILED!!\n"; return false; }
//		return true;
	}

	while(1) {
		i = (rand() + c) % list.size();
		for(j=0; j<System.n_type; j++) {
			if(idlecore_t[j] < list[i].idlecore_t[j]) break;
		}
		if(j < System.n_type) continue;

//cout << idx << " " << i << " " << c << endl;

		addr[idx] = i;
		for(j=0; j<System.n_type; j++) idlecore_t[j] -= list[i].idlecore_t[j];
		if(InitialRandomSelect(list, addr, idlecore_t, idx + 1)) return true;
		addr[idx] = -1;
		for(j=0; j<System.n_type; j++) idlecore_t[j] += list[i].idlecore_t[j];
		c++;
		if(c == list.size()) return false;
	}

	return false;
}

void GenerateSample(int *addr) {
	int i, j, k, l, m;
	int value[VC.size()];
	ofstream out, out2;
	string filename;

	for(i=0; i<VC.size(); i++) value[i] = 0;
	for(i=0; i<System.n_app; i++) value[addr[i]]++;

	out2.open("Corunners");

	for(i=0; i<System.n_app; i++) {
//cout << endl << "App" << i << endl;
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
//cout << "Type" << j << endl;
			for(k=0; k<System.node_t[j]; k++) {
				bool here = true;
				for(l=0; l<System.core_t[j]; l++)
					if(mapping[j][k][l] == i) { here = false; break; }
				if(here) continue;

				for(l=0; l<System.core_t[j]; l++) 
					if(mapping[j][k][l] != i) {
						for(m=0; m<ncr; m++) if(crs[m] == mapping[j][k][l]) break;
						if(m == ncr) {
							assert(ncr < 3);
							crs[ncr++] = mapping[j][k][l];
//cout << crs[ncr-1] << endl;
						}
					}
			}
		}

		for(j=0; j<ncr; j++) for(k=j+1; k<ncr; k++) if(crs[j] > crs[k]) { l = crs[j]; crs[j] = crs[k]; crs[k] = l; }

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
					if(mapping[j][k][l] != i) {
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

void GetInput() {
	int size_vc;
	int i, j, k;
	ifstream in;
	string filename;

	cin >> System.n_type >> System.n_app >> System.core_vm;
	System.core_t = new int[System.n_type];
	System.node_t = new int[System.n_type];
	System.idlecore_t = new int[System.n_type];
	System.host = new int*[System.n_type];
	System.total_core = 0;

	for(i=0; i<System.n_type; i++) {
		cin >> System.node_t[i] >> System.core_t[i];
		System.host[i] = new int[System.node_t[i]];
		System.core_t[i] /= System.core_vm;
		System.idlecore_t[i] = System.node_t[i] * System.core_t[i];
		System.total_core += System.node_t[i] * System.core_t[i];
	}

	System.n_vm = System.total_core / System.n_app;

	Appidx = new int[System.n_app];
	for(i=0; i<System.n_app; i++)
		cin >> Appidx[i];

	cin >> size_vc;
        for(i=0; i<size_vc; i++) {
                CONF vc;
                vc.init(System);
                for(j=0; j<vc.n_type; j++) {
			int temp;
                        cin >> temp;
//			vc.idlecore_t[j] -= temp;
			vc.idlecore_t[j] = temp;
		}

                for(j=0; j<vc.n_type; j++) {
                        int temp, temp2;
                        cin >> temp;
                        if(temp == 0) continue;
//                        temp2 = (vc.core_t[j] * vc.node_t[j] - vc.idlecore_t[j]) / temp;
                        temp2 = vc.idlecore_t[j] / temp;
                        for(k=0; k<temp; k++) vc.host[j][k] = temp2;
                }
                VC.push_back(vc);
        }

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
		execl("/home/gorae/NewVersion_170206/SAhelper.sh", "/home/gorae/NewVersion_170206/SAhelper.sh", temp.c_str(), NULL);
//		execl("/home/gorae/NewVersion_170206/HelpST/SAhelper.sh", "/home/gorae/NewVersion_170206/HelpST/SAhelper.sh", temp.c_str(), NULL);
		exit(0);
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
