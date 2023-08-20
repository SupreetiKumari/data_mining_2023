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



int threshold = 3;

int trans_size = 0 ;

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

    // header map to store the nodes corresponding to each item
        map <string, vector <FPNode*>> header_map;

    void insert(const vector<string>& transaction) {
        FPNode* current = root;
        for (const string& item : transaction) {
            if (current->children.find(item) == current->children.end()) {
                current->children[item] = new FPNode(item, current);
                // add the node to the header map
                if (header_map.find(item) == header_map.end()) {
                    vector <FPNode*> v;
                    header_map[item] = v;
                }
                header_map[item].push_back(current->children[item]);
                // header_map[item].push_back(current->children[item]);
            }
            current = current->children[item];
            current->frequency++; 


        }
    }

   

    void merge(){
        FPNode* prev = NULL ;
        FPNode* curr = new FPNode("", nullptr);
        curr->children = root->children;
        

        queue <FPNode*> q;
        

        
        // now iterate over the fptree to merge all consecutive nodes with same frequency
        
        q.push(curr) ;

        while(!q.empty()){
            

            int n = q.size();
            for(int i=0;i<n;i++){
                FPNode* temp = q.front();
                q.pop();
            

            // store all the children of temp in a vector
            vector<FPNode*> children;
            for(auto it = temp->children.begin(); it != temp->children.end(); ++it){
                children.push_back(it->second);
            }

            for(auto it = children.begin(); it != children.end(); ++it){
                FPNode* node = *it;

                if(node->children.size() == 0){
                    continue;
                }

                // skip this node if if it has more than 1 child
                if(node->children.size() > 1){
                    q.push(node);
                    continue;
                }

                if(node->frequency<10){
                    continue ;
                }
                if(node->frequency!=node->children.begin()->second->frequency){
                    q.push(node);
                    continue ;
                }

                
                // keep iterating to find the last node with the same frequency
                FPNode *iter = node->children.begin()->second;
                FPNode *last = node ;
                string new_item = node->item;
                string item1 = node->item;
                
                // add code for not merging just two elements
                
                while(iter!=NULL && iter->frequency == node->frequency){
                    new_item += "_" + iter->item ;
                    last = iter;
                    iter = iter->children.begin()->second;       
                }   
                // new_item += "_" + iter->item ;

        


                // now merge the nodes
                // node->item = node->item + "_" + last->item;
                node->item = new_item;
                //frequency is that of the last node
                node->frequency = last->frequency;
                node->children = last->children;

                // update the children map of the parent of node
                node->parent->children.erase(item1);
                node->parent->children[node->item] = node;

                // now remove the last node from the tree
                last->parent->children.erase(last->item);

                // now update the parent pointers of the children of last
                for(auto it = last->children.begin(); it != last->children.end(); ++it){
                    it->second->parent = node;
                }


                // update the queue
                q.push(node);





            }


        }
        }



        

        

        int it = 0 ;

        // while(prev!=curr){
        while(true){

            // map to store the frequency of each edge
        map <string, int> edge_frequency;

        
        
        // create a copy of the root node for deletion
        // curr = new FPNode("", nullptr);
        // curr->children = root->children;
        
        int lev = 0;

        
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
        
        if(it==0){
            if(max_freq<100000 && trans_size<100000){
                threshold = 3;
            }
            else{
                if(trans_size>=100000 && max_freq<100000){
                    threshold = 100;
                }
                else if(trans_size>=100000 && max_freq>=100000 && max_freq<1000000){
                    threshold = 5000;
                }
                else{
                    threshold = max_freq/100 ;
                }
            }
        }

        // now iterate over the fptree to remove the nodes such that the frequency of the edge formed by the node with its parent is less than threshold
        while(it == 0 && !q.empty()) {
            
            int n = q.size();

            for(int i=0;i<n;i++){
                FPNode* temp = q.front();
                q.pop();

                vector<FPNode*> children;
                for(auto it = temp->children.begin(); it != temp->children.end(); ++it){
                    children.push_back(it->second);
                }

                for(auto it = children.begin(); it != children.end(); ++it){
                    FPNode* node = *it;
                    if(lev>0){
                        
                        // calculate the frequency of the edge formed by the node with its parent
                        string edge = node->parent->item + " " + node->item;
                        int f1 = edge_frequency[edge];
                        
                        //iterate over the children of node to find the max frequency of the edge formed by the node with its children
                        
                        int flag = 0;
                        for(auto it1 = node->children.begin(); it1 != node->children.end(); ++it1){
                            string edge1 = node->item + " " + it1->second->item;
                            if(edge_frequency[edge1] > threshold){
                                flag = 1;
                                break ;
                            }
                        }
                        if(flag == 1){
                            q.push(node);
                            continue;
                        }

                        // else remove this node
                        node->parent->children.erase(node->item);
                        // update the edge frequency map
                        edge_frequency.erase(edge);
                        

                        // // now update the header map
                        // header_map[node->item].erase(header_map[node->item].begin(), header_map[node->item].end());

                        

                        
                        // now update the parent pointers of the children of node
                        for(auto it1 = node->children.begin(); it1 != node->children.end(); ++it1){
                            //update the edge frequency map by removing the edge formed by the node and its child
                            string edge1 = node->item + " " + it1->second->item;
                            edge_frequency.erase(edge1);

                            it1->second->parent = node->parent;

                            // update the edge frequency map by adding the edge formed by the node's parent and its child
                            string edge2 = node->parent->item + " " + it1->second->item;
                            edge_frequency[edge2] = it1->second->frequency;

                            // update the children map of the parent of node
                            node->parent->children[it1->second->item] = it1->second;

                            // update the header map
                            // header_map[it1->second->item].push_back(it1->second);

                            // update the queue
                            q.push(it1->second);


                        }

                        // now delete the node
                        delete node;

                        
                        

                    }
                    else{
                        q.push(node);
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

        



        if(max_freq<threshold){
            // terminate = true ;
            break ;
        }

        // cout << "max freq edge: " << max_edge << " has frequency: "  << max_freq << endl;

        

        // now merge the two items in the edge
        stringstream ss(max_edge);
        string item1, item2;
        ss >> item1;
        ss >> item2;

        

        

        // cout << item1 << " " << item2 << endl;

        


        // // iterate over the header map to find the nodes corresponding to the two items
        

        // // 
        
        // vector <int> ind_remove(header_map[item1].size(), 0);
        // for(int i =0 ;i<header_map[item1].size();i++){  
        //     // cout << i->item << " " << i->frequency << endl;
        //     FPNode *node1 = header_map[item1][i];
            
        //     // check if node1 has a child with item2
        //     if(node1->children.find(item2) == node1->children.end()){
        //         continue;
        //     }
            
        //     ind_remove[i] = 1;


        //     // node 2 is a child of node 1
        //     FPNode* node2 = node1->children[item2];

        //     // check if frequency of node2 is less than node1
        //     if(node2->frequency < node1->frequency){
        //         // add a copy of node 1 for the other children of node1 except node 2
        //         FPNode* node1_copy = new FPNode(item1 + "_copy", node1->parent);
        //         node1_copy->frequency = node1->frequency - node2->frequency;
                
        //         // the children of node 1 except node 2 are the children of temp
                
        //         node1_copy->children = node1->children;
        //         node1_copy->children.erase(item2);

        //         // // update the parent pointers of the children of temp
        //         // for(auto it = node1_copy->children.begin(); it != node1_copy->children.end(); ++it){
        //         //     it->second->parent = node1_copy;
        //         // }

        //         // now add temp to the children of node1's parent
        //         node1->parent->children[item1+"_copy"] = node1_copy;

        //         // now merge the two nodes

        //         // merge 
        //         node1->item = item1 + "_" + item2;
        //         //frequency is that of the node 2
        //         node1->frequency = node2->frequency;
        //         node1->children = node2->children;

        //         // update the children map of the parent of node 1
        //         node1->parent->children.erase(item1);
        //         node1->parent->children[item1 + "_" + item2] = node1;

        //         // now remove the node2 from the tree
        //         node2->parent->children.erase(item2);

        //         // now update the parent pointers of the children of node2
        //         for(auto it = node2->children.begin(); it != node2->children.end(); ++it){
        //             it->second->parent = node1;
        //         }


        //         // update the parent pointers of the children of node1_copy
        //         for(auto it = node1_copy->children.begin(); it != node1_copy->children.end(); ++it){
        //             it->second->parent = node1_copy;
        //         }


        //         // update the header map with node1_copy
        //         header_map[node1_copy->item].push_back(node1_copy);
        //         header_map[node1->item].push_back(node1);

        //         // remove node2 from the header map
        //         // header_map[item2].erase(header_map[item2].begin(), header_map[item2].end());



                
        //     }
        //     // merge the consecutive nodes with max frequency



        //     else{ // just merge the nodes
                
                
        //     // // keep iterating over the children to find the last node with the same frequency
        //     // FPNode *iter = node1->children[item2];
        //     // while(iter!=NULL && iter->frequency == node1->frequency){
        //     //     iter = iter->children[item2];
        //     // }
        //     //     iter = 
                
                
        //         // now merge the two nodes
        //         node1->item = item1 + "_" + item2;
        //         //frequency is that of the node 2
        //         node1->frequency = node2->frequency;
        //         node1->children = node2->children;

        //         // update the children map of the parent of node 1
        //         node1->parent->children.erase(item1);
        //         node1->parent->children[item1 + "_" + item2] = node1;

        //         // now remove the node2 from the tree
        //         node2->parent->children.erase(item2);

        //         // now update the parent pointers of the children of node2
        //         for(auto it = node2->children.begin(); it != node2->children.end(); ++it){
        //             it->second->parent = node1;
        //         }

        //         // update the header map with node1
        //         header_map[node1->item].push_back(node1);

        //         // remove node2 from the header map
        //         // header_map[item2].erase(header_map[item2].begin(), header_map[item2].end());
        //     }




        // }

        // // create a new vector to store the nodes corresponding to item1
        // vector <FPNode*> v;
        // for(int i =0 ;i<header_map[item1].size();i++){
        //     if(ind_remove[i] ==0){
        //         v.push_back(header_map[item1][i]);
        //     }
        // }

        // // update the header map
        // header_map[item1] = v;

      
        


        

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

                    // cout << "FOUND THE NODES" << endl ;

                    // node 2 is a child of node 1
                    FPNode* node2 = node1->children[item2];

                    

                    // check if frequency of node2 is less than node1
                    if(node2->frequency < node1->frequency){
                        // add a copy of node 1 for the other children of node1 except node 2
                        FPNode* node1_copy = new FPNode(item1 + "_copy", node1->parent);
                        node1_copy->frequency = node1->frequency - node2->frequency;
                        
                        // the children of node 1 except node 2 are the children of temp
                        
                        node1_copy->children = node1->children;
                        node1_copy->children.erase(item2);

                        // // update the parent pointers of the children of temp
                        // for(auto it = node1_copy->children.begin(); it != node1_copy->children.end(); ++it){
                        //     it->second->parent = node1_copy;
                        // }

                        // now add temp to the children of node1's parent
                        node1->parent->children[item1+"_copy"] = node1_copy;



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
                        // node1_copy->children.erase(item2);

                        // update the parent pointers of the children of node1_copy
                        for(auto it = node1_copy->children.begin(); it != node1_copy->children.end(); ++it){
                            it->second->parent = node1_copy;
                        }
                        

                        // remove the string "copy" from the item of node1_copy
                        // node1_copy->item = item1;


                        // update the queue
                        
                        q.push(node1_copy);
                        q.push(node1) ;

                            
                        
                    }
                    else{ // just merge the nodes
                        // now merge the two nodes

                        // cout << "HI I am in else" <<endl ;

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
        it++ ;

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

    void compress(vector<vector<string>> &compressed_dataset, map <string, string> &compression_map){
        //now iterate over the fp tree and for each node having "_" in its item, print the item and its frequency
        queue <FPNode*> q;
        q.push(root) ;

        // map <string, string> compression_map ;
        // char ch = 'A';
        int sub = -1 ;

        while(!q.empty()) {
            FPNode* node = q.front();
            q.pop();

            if(node->item.find("_") != string::npos){
                // check if node is already present in the compression map
                if(compression_map.find(node->item) == compression_map.end()){
                    // add the node to the compression map
                    // compression_map[node->item] = ch;
                    compression_map[node->item] = to_string(sub);
                    // ch++;
                    sub--;
                }
                
            }

            for(auto it = node->children.begin(); it != node->children.end(); ++it){
                q.push(it->second);
            }
        }

        // print the compression map
        
        
        // cout << "compression map:" << endl;
        // for(auto it = compression_map.begin(); it != compression_map.end(); ++it){
        //     cout << it->first << " " << it->second << endl;
        // }

        // cout << "compression map size :" << compression_map.size() << endl;

        // // using the compression map replace the items in the fp tree
        // queue <FPNode*> q1;
        // q1.push(root) ;

        // while(!q1.empty()) {
        //     FPNode* node = q1.front();
        //     q1.pop();

        //     if(node->item.find("_") != string::npos){
        //         // replace the item of the node with the corresponding value in the compression map
        //         node->item = compression_map[node->item];
        //     }

        //     for(auto it = node->children.begin(); it != node->children.end(); ++it){
        //         q1.push(it->second);
        //     }
        // }

        


        // iterate over the fptree and for each root to leaf path, construct the compressed transaction
        q.push(root) ;
        while(!q.empty()) {
            FPNode* node = q.front();
            q.pop();
            // cout <<  "HI GUYS" << endl ;

            // check if node is a leaf node
            if(node->children.size() == 0){
                // construct the compressed transaction
                vector<string> compressed_transaction;
                FPNode* temp = node;
                while(temp != root){
                    // cout << "processing node " << temp->item << " " << temp->frequency << " " << temp->parent->item << " " << temp->children.size() << " ";
                    
                    // check if the item is present in the compression map
                    if(compression_map.find(temp->item) != compression_map.end()){
                        compressed_transaction.push_back(compression_map[temp->item]);
                    }
                    else{
                        compressed_transaction.push_back(temp->item);
                    }
                    // compressed_transaction.push_back(temp->item);
                    temp = temp->parent;
                }

                // add the compressed transaction to the compressed dataset the number of times equal to the frequency of the node
                for(int i=0;i<node->frequency;i++){
                    compressed_dataset.push_back(compressed_transaction);
                }
            }

            else{
                // check if node's frequency is greater than sum of frequencies of its children
                int sum = 0;
                for(auto it = node->children.begin(); it != node->children.end(); ++it){
                    sum += it->second->frequency;
                }
                if (node->frequency > sum){
                    // construct the compressed transaction
                    vector<string> compressed_transaction;
                    FPNode* temp = node;
                    while(temp!= root){
                        // check if the item is present in the compression map
                        if(compression_map.find(temp->item) != compression_map.end()){
                            compressed_transaction.push_back(compression_map[temp->item]);
                        }
                        else{
                            compressed_transaction.push_back(temp->item);
                        }
                        temp = temp->parent;
                    }
                
                // add the root to leaf path to this node to the compressed dataset the number of times equal to the difference of the frequency of the node and the sum of frequencies of its children
                for(int i=0;i<node->frequency-sum;i++){
                    // add the compressed transaction to the compressed dataset
                    compressed_dataset.push_back(compressed_transaction);
                }
                }
                

            }

            for(auto it = node->children.begin(); it != node->children.end(); ++it){
                q.push(it->second);
            }
        }




        // // print the compressed dataset
        // // calc the space of the compressed dataset
        // double space = 0;
        // cout << "compressed dataset:" << endl;
        // for(int i=0;i<compressed_dataset.size();i++){
        //     for(int j=0;j<compressed_dataset[i].size();j++){
        //         space += compressed_dataset[i][j].length();
        //         cout << compressed_dataset[i][j] << " ";
        //     }
        //     cout << endl;
        // }

        // cout << "space: " << space << endl;
    



    }


    // func to calc space of the dataset formed by the fptree
    int calc_space(){
        // iterate over the fptree and calculate the size of transactions in the dataset formed corresponding to this fptree
        queue <FPNode*> q;
        q.push(root) ;
        int space = 0;

        while(!q.empty()){
            FPNode* node = q.front();
            q.pop();

            for(auto it = node->children.begin(); it != node->children.end(); ++it){
                // for each child node, calculate the space of the transaction formed by the root to leaf path to this node
                // space = freq * (size of item)
                space += it->second->frequency * (it->second->item.length());
                q.push(it->second);
            }

        }

        cout << "space: " << space << endl;

        return space;


    }
    

};


void check_correctness(map<string, string> compression_map, vector<vector<string>> compressed_dataset, vector<vector<string>> transactions_str){

    cout << "size of compression map" << compression_map.size() << endl;
    // construct the reverse compression map
    map <string, string> reverse_compression_map;
    for(auto it = compression_map.begin(); it != compression_map.end(); ++it){
        reverse_compression_map[it->second] = it->first;
    }

    


    //check the correctness of the compressed dataset by reconstructing the original dataset

    vector <vector<string>> final_dataset(compressed_dataset.size());

    // iterate over the compressed dataset and for each transaction, replace the compressed items with the original items
    for(int i=0;i<compressed_dataset.size();i++){
        for(int j=0;j<compressed_dataset[i].size();j++){
            
            // check if the item is present in the reverse compression map
            if(reverse_compression_map.find(compressed_dataset[i][j]) != reverse_compression_map.end()){
                string item = reverse_compression_map[compressed_dataset[i][j]];
                
                
                stringstream ss(item);
                string item1;
                
                // split the string by "_" and add the items to the final dataset
                while(getline(ss, item1, '_')){
                    // if the string is "copy" then ignore it
                    if(item1 == "copy"){
                        continue;
                    }
                    final_dataset[i].push_back(item1);
                }
                

            }
            else{
                // add the item to the final dataset
                final_dataset[i].push_back(compressed_dataset[i][j]);
            }
            

        }
        
    }

    // // print the final dataset
    // cout << "final dataset:" << endl;
    // for(int i=0;i<final_dataset.size();i++){
    //     for(int j=0;j<final_dataset[i].size();j++){
    //         cout << final_dataset[i][j] << " ";
    //     }
    //     cout << endl;
    // }

    // compare the final dataset with the original dataset where the order of transactions and items in each transaction is not important
    // sort the transactions in the final dataset
    for(int i=0;i<final_dataset.size();i++){
        sort(final_dataset[i].begin(), final_dataset[i].end());
    }

    // now sort the final dataset vector    
    sort(final_dataset.begin(), final_dataset.end());

    // sort the transactions in the original dataset
    for(int i=0;i<transactions_str.size();i++){
        sort(transactions_str[i].begin(), transactions_str[i].end());
    }

    // now sort the transactions_str vector
    sort(transactions_str.begin(), transactions_str.end());

    // now compare the two vectors
    bool flag = true;
    for(int i=0;i<final_dataset.size();i++){
        if(final_dataset[i] != transactions_str[i]){
            flag = false;
            break;
        }
    }

    if(flag){
        cout << "correctness of compression: true" << endl;
    }
    else{
        cout << "correctness of compression: false" << endl;
    }


}


/////-----------------------------------



int main() {
    // start time 
    clock_t start, end;
    start = clock();
    
    ifstream in("D_medium.dat");
    in >> noskipws;

    // create a frequency map for each pair of items
    map <pair<int,int>, int> edge_frequency_map;



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

        // // construct the frequency map for each pair of items
        // for(int i=0;i<transactions.size();i++){
        //     for(int j=0;j<transactions[i].size();j++){
        //         for(int k=j+1;k<transactions[i].size();k++){
        //             pair <int, int> p;
        //             p.first = transactions[i][j];
        //             p.second = transactions[i][k];
        //             if(edge_frequency_map.find(p) == edge_frequency_map.end()){
        //                 edge_frequency_map[p] = 0;
        //             }
        //             edge_frequency_map[p]++;
        //         }
        //     }
        // }

    // // sort each transaction in the dataset in decreasing order of edge frequency
    // for(int i=0;i<transactions.size();i++){
    //     // sort each transaction in the dataset in decreasing order of edge frequency
    //     sort(transactions[i].begin(), transactions[i].end(), [&](int a, int b){
    //         pair <int, int> p;
    //         p.first = a;
    //         p.second = b;
    //         return edge_frequency_map[p] > edge_frequency_map[p];
    //     });        
    // }
    
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

    // calc space of the dataset
    double orig_space = 0; 

    for (int i = 0; i < transactions.size(); i++) {
        orig_space += transactions[i].size();
        for (int j = 0; j < transactions[i].size(); j++) {
            transactions_str[i].push_back(to_string(transactions[i][j]));
            
        }
        
    }

    // double orig_space = sizeof(transactions_str);
    

    // cout << "orig space: " << orig_space << endl;

    


    // // print the elements of the transactions_str
    // for (int i = 0; i < transactions_str.size(); i++) {
    //     for (int j = 0; j < transactions_str[i].size(); j++) {
    //         cout << transactions_str[i][j] << " ";
    //     }
    //     cout << endl;
    // }
    
    // end time
    end = clock();

    // print the time taken
    // cout << "Time taken in reading the file: " << (double)(end - start)/CLOCKS_PER_SEC  << "seconds" << endl;


    // start time
    start = clock();
    vector<vector<string> > dataset =  transactions_str; 
    
  
    // int min_support = 10;

    FPTree fp_tree;

    for (const auto& transaction : dataset) {
        fp_tree.insert(transaction);
        // fp_tree.update_support(transaction);
    }

    // print the fp tree
    // fp_tree.print();

    // int orig_space = fp_tree.calc_space();

    // now merge the fp tree
    fp_tree.merge();

    // end time
    end = clock();

    // print the time taken
    // cout << "Time taken in merging the fp tree: " << (double)(end - start)/CLOCKS_PER_SEC << "seconds" << endl;


    // print the fp tree
    // fp_tree.print();

    vector<vector<string>> compressed_dataset;
    map <string, string> compression_map ;

    // compress the fp tree
    fp_tree.compress(compressed_dataset, compression_map);

    // write to a file the compressed dataset and the compression map
    ofstream out("compressed_dataset.txt");
    for(int i=0;i<compressed_dataset.size();i++){
        for(int j=0;j<compressed_dataset[i].size();j++){
            out << compressed_dataset[i][j] << " ";
        }
        out << endl;
    }

    // write the compression map to a file
    ofstream out1("compression_map.txt");
    for(auto it = compression_map.begin(); it != compression_map.end(); ++it){
        out1 << it->first << " " << it->second << endl;
    }

    out.close();
    out1.close();
    

    // double compressed_space = sizeof(compressed_dataset);
    // calc the space of the compressed dataset
    int compressed_space = 0;
    for(int i=0;i<compressed_dataset.size();i++){
        compressed_space += compressed_dataset[i].size();
    }


    // add the size of compression map to the compressed space
    for(auto it = compression_map.begin(); it != compression_map.end(); ++it){
        // calc the number of "_" separated elements in the item
        stringstream ss(it->first);
        string item;
        int count = 0;
        while(getline(ss, item, '_')){
            // check if item is "copy"
            if (item == "copy"){
                continue;
            }
            count++;
        }
        compressed_space += count+1;
        
    }

    // cout << "compressed space: " << compressed_space << endl;

    // print the fp tree
    // fp_tree.print();

    // int compressed_space = fp_tree.calc_space();

    // print the compression ratio
    // cout << "compression ratio: " << (double)orig_space/compressed_space << endl;
    // print the percentage of compressionn
    // cout << "percentage of compression: " << 100 * (double)(orig_space-compressed_space)/orig_space << endl;


    // check the correctness of the compression
    // check_correctness(compression_map, compressed_dataset, transactions_str);    



    return 0;
}
