#include <iostream>
#include <unordered_set>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <vector>
#include <queue>
#include <map>
#include <algorithm>
#include <set>
#include <string>
#include <sstream>
using namespace std;
int main(){
 ifstream inputFile("compressed_dataset.dat"); // Replace with your file name
   

    int numTransactions;
    inputFile >> numTransactions;
    inputFile.ignore(); // Consume the newline after the number

    std::vector<std::vector<int>> transactions;

    for (int i = 0; i < numTransactions; ++i) {
        std::string line;
        std::getline(inputFile, line);
        
        std::istringstream iss(line);
        std::vector<int> transaction;
        
        int value;
        while (iss >> value) {
            transaction.push_back(value);
        }
        
        transactions.push_back(transaction);
    }

    // Printing the transactions
    for (const auto& transaction : transactions) {
        for (int value : transaction) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }
 int numMapElements;
    inputFile >> numMapElements;
    inputFile.ignore(); // Consume the newline after the number

    std::map< int,string> keyValueMap;

    for (int i = 0; i < numMapElements; ++i) {
        int key;
        string value;
        inputFile >> value>>key ;
        
        keyValueMap[key] = value;
    }
     for (const auto& entry : keyValueMap) {
        std::cout << "Key: " << entry.first << ", Value: " << entry.second << std::endl;
    }

    inputFile.close();


    vector <vector<int>> final_dataset(transactions.size());

    // iterate over the compressed dataset and for each transaction, replace the compressed items with the original items
    for(int i=0;i<transactions.size();i++){
        for(int j=0;j<transactions[i].size();j++){
            
            // check if the item is present in the reverse compression map
            if(keyValueMap.find(transactions[i][j]) != keyValueMap.end()){
                string item = keyValueMap[transactions[i][j]];
                
                
                stringstream ss(item);
                string item1;
                
                // split the string by "_" and add the items to the final dataset
                while(getline(ss, item1, '_')){
                    // if the string is "copy" then ignore it
                    if(item1 == "copy"){
                        continue;
                    }
                    final_dataset[i].push_back(stoi(item1));
                }
                

            }
            else{
                // add the item to the final dataset
                final_dataset[i].push_back((transactions[i][j]));
            }
            

        }
        
    }
 for(int i=0;i<final_dataset.size();i++){
        sort(final_dataset[i].begin(), final_dataset[i].end());
    }

    // now sort the final dataset vector    
    sort(final_dataset.begin(), final_dataset.end());
      ofstream out("reconstructed.dat");

    
    for(int i=0;i<final_dataset.size();i++){
        for(int j=0;j<final_dataset[i].size();j++){
            out << final_dataset[i][j] << " ";
        }
        out << endl;
    }


}
