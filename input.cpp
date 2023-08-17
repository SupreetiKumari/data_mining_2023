#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>

using namespace std;

vector<vector<int>> transactions;
unordered_map<int, int> frequency_map;
unordered_set<int> duplicate_checker;

bool freq_compare(int a, int b) {
    return (frequency_map[a] > frequency_map[b]);
}

int main() {
    ifstream in("in.dat");
    in >> noskipws;

    while (!in.eof()) {
        vector<int> new_trans;
        char ch;
        int num;
        int curr;
        while (true) {            
            num = 0;
            bool eol = false;

            while (true) { //for reading the number
                in >> ch;
                if (ch == '\n') {
                    eol = true;
                    break;
                }
                if (ch == ' ') break;
                curr = ch - '0';
                num = (num*10)+curr;
            }

            if (num <= 0) {
                if (eol) break;
                else continue;
            }

            if (duplicate_checker.find(num) == duplicate_checker.end()) {
                duplicate_checker.insert(num);
                new_trans.push_back(num);
                if (frequency_map.find(num) == frequency_map.end()) {
                    frequency_map[num] = 0;
                }
                frequency_map[num]++;
            }
            if (eol) break;
        }
        if (new_trans.size() > 0) {
            transactions.push_back(new_trans);
            duplicate_checker.clear();
        }
    }

    for (int i = 0; i < transactions.size(); i++) {
        sort(transactions[i].begin(), transactions[i].end(), frequency_compare);
    }

    return 0;
}
