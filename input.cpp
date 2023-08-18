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

vector<vector<int>> transactions;
unordered_map<int, int> frequency_map;
unordered_set<int> duplicate_checker;

bool freq_compare(int a, int b) {
    return (frequency_map[a] > frequency_map[b]);
}





///// -----------------------------------

struct FPNode {
    string item;
    int frequency;
    FPNode* parent;
    map<string, FPNode*> children;

    FPNode(const string& item, FPNode* parent) : item(item), frequency(0), parent(parent) {}
};

class FPTree {
private:
    FPNode* root;
    
    

public:
    FPTree() : root(new FPNode("", nullptr)) {}

    void insert(const vector<string>& transaction) {
        FPNode* current = root;
        for (const string& item : transaction) {
            if (current->children.find(item) == current->children.end()) {
                current->children[item] = new FPNode(item, current);
            }
            current = current->children[item];
            current->frequency++; 


        }
    }

   

    void merge(){
        FPNode* prev = NULL ;
        FPNode* curr = new FPNode("", nullptr);
        curr->children = root->children;
        
        // while(prev!=curr){
        while(true){

            // map to store the frequency of each edge
        map <string, int> edge_frequency;
        
        // create a copy of the root node for deletion
        // curr = new FPNode("", nullptr);
        // curr->children = root->children;
        
        int lev = 0;

        queue <FPNode*> q;
        q.push(curr) ;

        int max_freq = 0 ;
        string max_edge = "";



        // iterate over the trie and maintain a frequency map for each edge of items
        while(!q.empty()) {
            
            int n = q.size();

            for(int i=0;i<n;i++){
                FPNode* temp = q.front();
                q.pop();

                for(auto it = temp->children.begin(); it != temp->children.end(); ++it){
                    q.push(it->second);
                    if(lev>0){
                                    FPNode* node = it->second;
                        // calculate the frequency of the edge formed by the node with its parent
                        string edge = node->parent->item + " " + node->item;
                        int freq = node->frequency;
                        if(edge_frequency.find(edge) != edge_frequency.end()){
                            freq += edge_frequency[edge];
                        }
                        edge_frequency[edge] = freq;

                        // update the max_freq and max_edge
                        if(freq > max_freq){
                            max_freq = freq;
                            max_edge = edge;
                        }
                    }
         
                }
            }
            lev++ ;
        }

        
        // cout << "edge frequency map:" << endl ;

        // now select the edge with maximum frequency and merge the two items
    //    int max_freq = 0 ;
    //     string max_edge = "";
        // for(auto it = edge_frequency.begin(); it != edge_frequency.end(); ++it){
        //     // print the edge and its frequency
        //     // cout << it->first << " " << it->second << endl;
        //     if(it->second > max_freq){
        //         max_freq = it->second;
        //         max_edge = it->first;
        //     }
        // }

        



        if(max_freq<3){
            // terminate = true ;
            break ;
        }

        cout << "max freq edge: " << max_edge << " has frequency: "  << max_freq << endl;

        

        // now merge the two items in the edge
        stringstream ss(max_edge);
        string item1, item2;
        ss >> item1;
        ss >> item2;

        

        

        // cout << item1 << " " << item2 << endl;

        
        

        // now merge the two items everywhere in the tree
        // iterate over the entire tree to find all the edges with item1 and item2
        
        FPNode *temp1 = new FPNode("", nullptr);
        temp1->children = root->children;

        
        
        q.push(temp1) ;
        
        
        // q.push(root) ;


        while(!q.empty()) {
            
            
            // vector<FPNode*> level;
            // for (auto it = temp->children.begin(); it != temp->children.end(); ++it) {
            //     level.push_back(it->second);
            // }
            // cout << "temp children: " << temp1->children.size() << endl;
            int n = q.size();
            for(int i=0;i<n;i++){
                FPNode* temp = q.front();
                q.pop();

                // cout << "start processing node " << temp->item << " " << temp->frequency << " " << endl ;

                // store all the children of temp in a vector
                vector<FPNode*> children;
                for(auto it = temp->children.begin(); it != temp->children.end(); ++it){
                    children.push_back(it->second);
                }


                
                for(auto it = children.begin(); it != children.end(); ++it){
                    
                    FPNode* node = *it;
                    // cout << "processing node " << node->item << " " << node->frequency << " " << endl ;
                    if(node->item == item1){

                    // if found then merge this node with its child node with item2
                    // first find the nodes corresponding to the two items
                    FPNode* node1 = node;
                    // check if node1 has a child with item2
                    if(node1->children.find(item2) == node1->children.end()){
                        q.push(node);
                        continue;
                    }

                    // node 2 is a child of node 1
                    FPNode* node2 = node1->children[item2];

                    

                    // check if frequency of node2 is less than node1
                    if(node2->frequency < node1->frequency){
                        // add a copy of node 1 for the other children of node1 except node 2
                        FPNode* node1_copy = new FPNode(item1 + "copy", node1->parent);
                        node1_copy->frequency = node1->frequency - node2->frequency;
                        
                        // the children of node 1 except node 2 are the children of temp
                        
                        node1_copy->children = node1->children;
                        // node1_copy->children.erase(item2);

                        // update the parent pointers of the children of temp
                        for(auto it = node1_copy->children.begin(); it != node1_copy->children.end(); ++it){
                            it->second->parent = node1_copy;
                        }

                        // now add temp to the children of node1's parent
                        node1->parent->children[item1+"copy"] = node1_copy;



                        // now merge the two nodes
                        node1->item = item1 + "_" + item2;
                        //frequency is that of the node 2
                        node1->frequency = node2->frequency;
                        node1->children = node2->children;

                        // update the children map of the parent of node 1
                        node1->parent->children.erase(item1);
                        node1->parent->children[item1 + "_" + item2] = node1;

                        // now remove the node2 from the tree
                        node2->parent->children.erase(item2);

                        // now update the parent pointers of the children of node2
                        for(auto it = node2->children.begin(); it != node2->children.end(); ++it){
                            it->second->parent = node1;
                        }

                        // remove the node2 from the children of node1_copy
                        node1_copy->children.erase(item2);
                        

                        // remove the string "copy" from the item of node1_copy
                        // node1_copy->item = item1;


                        // update the queue
                        
                        q.push(node1_copy);
                        q.push(node1) ;

                            
                        
                    }
                    else{ // just merge the nodes
                        // now merge the two nodes

                        // cout << "merging nodes " << node1->item << " " << node1->frequency << " " << node2->item << " " << node2->frequency << endl;
                    node1->item = item1 + "_" + item2;
                    //frequency is that of the node 2
                    node1->frequency = node2->frequency;
                    node1->children = node2->children;

                    // update the children map of the parent of node 1
                    node1->parent->children.erase(item1);
                    node1->parent->children[item1 + "_" + item2] = node1;

                    // now remove the node2 from the tree
                    node2->parent->children.erase(item2);

                    // now update the parent pointers of the children of node2
                    for(auto it = node2->children.begin(); it != node2->children.end(); ++it){
                        it->second->parent = node1;
                    }


                    // update the queue
                    q.push(node1) ;

                    // cout << "merged nodes " << node1->item << " " << node1->frequency << endl;

                    // cout << "queue: " << endl ;
                    // // print the elements of queue without popping
                    // int n = q.size();
                    // for(int i=0;i<n;i++){
                    //     FPNode* node = q.front();
                    //     q.pop();
                    //     cout << node->item <<  " ";
                    //     q.push(node);
                    // }
                    // cout << endl;




                



                    }

                    


                }
                else{
                    q.push(node);
                }

                // cout << "end processing node " << node->item << " " << node->frequency << " " << endl ;

                    
                }

                

                
            }




        }

        // clear space for the edge_frequency map
        // edge_frequency.clear();
        // delete the copy of the root node
        // delete temp;


        // cout << "end of iteration " << endl ;
        

        }

        


    }

    // print the fp tree
    void print() {
        cout << "fptree:" << endl ; 
        
        // create a copy of the root node for deletion
        FPNode* curr = new FPNode("", nullptr);
        curr->children = root->children;

        queue <FPNode*> q;
        q.push(curr) ;
        

        // print the entire fp tree in level order
        while(!q.empty()) {
            // vector<FPNode*> level;
            // for (auto it = curr->children.begin(); it != curr->children.end(); ++it) {
            //     level.push_back(it->second);
            // }

            int n = q.size();

            for(int i=0;i<n;i++){
                FPNode* node = q.front();
                q.pop();
                
                
                    // cout << node->item << " " << node->frequency << " " << node->parent->item << " " << node->children.size() << " ";
                for(auto it = node->children.begin(); it != node->children.end(); ++it){
                    q.push(it->second);
                    cout << it->second->item << " " << it->second->frequency << " ";
                }
                


                
            }
            cout << endl;
        

        }

    }
};


/////-----------------------------------



int main() {
    ifstream in("D_medium.dat");
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
        sort(transactions[i].begin(), transactions[i].end(), freq_compare);
    }

    // // print the new_trans
    // for (int i = 0; i < transactions.size(); i++) {
    //     for (int j = 0; j < transactions[i].size(); j++) {
    //         cout << transactions[i][j] << " ";
    //     }
    //     cout << endl;
    // }

    // convert vectors of vectors of int to vectors of vectors of string
    vector<vector<string>> transactions_str(transactions.size());

    for (int i = 0; i < transactions.size(); i++) {
        for (int j = 0; j < transactions[i].size(); j++) {
            transactions_str[i].push_back(to_string(transactions[i][j]));
            
        }
        
    }

    // print the elements of the transactions_str
    for (int i = 0; i < transactions_str.size(); i++) {
        for (int j = 0; j < transactions_str[i].size(); j++) {
            cout << transactions_str[i][j] << " ";
        }
        cout << endl;
    }
    

    vector<vector<string> > dataset =  transactions_str; 
    
  
    // int min_support = 10;

    FPTree fp_tree;

    for (const auto& transaction : dataset) {
        fp_tree.insert(transaction);
        // fp_tree.update_support(transaction);
    }

    // print the fp tree
    // fp_tree.print();

    // now merge the fp tree
    fp_tree.merge();

    // print the fp tree
    // fp_tree.print();

    
    

    return 0;
}
