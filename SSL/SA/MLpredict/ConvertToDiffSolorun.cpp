#include<iostream>
#include<fstream>
#include<sstream>
#include<string>

#define N 4
#define M 10
#define O 25

using namespace std;

string apps[] = {"grepspark", "wcspark", "grephadoop", "teragen", "cg", "lammps", "namd", "132"};

int main(int argc, char** argv) {
	double original_value[N][M][O], new_value[N][M][O];
	double cfg[3], factor[6], temp, temp2;
	int i, j, k, ii, x, y, x2, y2;
	char ch;
	string filename, line;
	ifstream in;

	if(argc < 3) {
		cout << "Usage: " << argv[0] << " original_datapath new_datapath [opt]\n";
		return -1;
	}

	for(i=0; i<N; i++) {
		filename = argv[1];
		filename.append("/" + apps[i] + ".dat");
//		cout << "Open " << filename << endl;
		in.open(filename.c_str());
		for(j=0; j<M; j++) for(k=0; k<O; k++) in >> original_value[i][j][k];
		in.close();
	}

	for(i=0; i<N; i++) {
		filename = argv[2];
		filename.append("/" + apps[i] + ".dat");
//		cout << "Open " << filename << endl;
		in.open(filename.c_str());
		for(j=0; j<M; j++) for(k=0; k<O; k++) in >> new_value[i][j][k];
		in.close();
	}

	if(argc > 3) getline(cin, line);
	while(1) {
		getline(cin, line);
		if(cin.eof()) break;
		stringstream ss;
		ss << line;

		for(i=0; i<3; i++) {
			ss >> cfg[i] >> ch;
			cout << cfg[i] << ",";
		}

		for(i=0; i<6; i++) {
			ss >> factor[i] >> ch;
			cout << factor[i] << ",";
		}

		ss >> temp;
//		temp2 = temp / (8 / (cfg[1] > cfg[2] ? cfg[1] : cfg[2]));
		temp2 = temp / (8 / (cfg[1] + cfg[2]));
		
		for(x=0; x<N; x++) {
			for(y=0; y<M; y++)
				if(temp2 - original_value[x][y][1] < 0.00001 && temp2 - original_value[x][y][1] > -0.00001) break;
			if(y != M) break;
		}

		if(x == N) {
			cout << temp;
			for(k=2; k<O; k++) {
				ss >> ch >> temp;
				cout << "," << temp;
			}
		}
		else {
			temp2 = 8 / (cfg[1] > cfg[2] ? cfg[1] : cfg[2]);
			cout << new_value[x][y][1] * temp2;
			for(k=2; k<O; k++) {
				ss >> ch >> temp;
				cout << "," << new_value[x][y][k] * temp2;
			}
		}

		x2 = x; y2 = y;

		for(ii=0; ii<6; ii++) {
			ss >> ch >> temp;
			if(factor[ii] > 0) temp2 = temp / factor[ii];
			else temp2 = 0;
			for(x=0; x<N; x++) {
				for(y=0; y<M; y++)
					if(temp2 - original_value[x][y][1] < 0.00001 && temp2 - original_value[x][y][1] > -0.00001) break;
				if(y != M) break;
			}

			if(x == N) {
				cout << "," << temp;
				for(k=2; k<O; k++) {
					ss >> ch >> temp;
					cout << "," << temp;
				}
			}
			else {
				cout << "," << new_value[x][y][1] * factor[ii];
				for(k=2; k<O; k++) {
					ss >> ch >> temp;
					cout << "," << new_value[x][y][k] * factor[ii];
				}
			}
		}

		ss >> ch >> temp;
//		if(x2 == N) cout << "," << temp;
//		else cout << "," << new_value[x2][y2][0];
		cout << "," << new_value[x2][y2][0];

		for(i=0; i<2; i++) {
			ss >> ch >> temp;
			cout << "," << temp;
		}

		cout << endl;
	}
}
