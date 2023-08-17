#include <iostream>
#include <vector>
#include <queue>
#include <map>
#include <algorithm>
#include <set>
#include <string>
#include <sstream>

using namespace std;

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
        prev = curr ;
        // create a copy of the root node for deletion
        curr = new FPNode("", nullptr);
        curr->children = root->children;
        int lev = 0;

        // iterate over the trie and maintain a frequency map for each edge of items
        while(curr->children.size() > 0) {
            
            vector<FPNode*> level;
            for (auto it = curr->children.begin(); it != curr->children.end(); ++it) {
                level.push_back(it->second);
            }

            // skip the first level because it has the edges from the null node to the first level
            if(lev==0){
                curr->children.clear();
                for (FPNode* node : level) {
                    for (auto it = node->children.begin(); it != node->children.end(); ++it) {
                        curr->children[it->first] = it->second;
                    }
                }

                lev++ ;
                continue;
            }

            // iterate over the nodes in the level to calculate the frequency of each edge
            for (FPNode* node : level) {
                
                
                FPNode* temp = node;
                // calculate the frequency of the edge formed by the node with its parent
                string edge = node->parent->item + " " + node->item;
                int freq = node->frequency;
                if(edge_frequency.find(edge) != edge_frequency.end()){
                    freq += edge_frequency[edge];
                }
                edge_frequency[edge] = freq;


                // print the edge and its frequency
                // cout << node->item << " " << node->parent->item << " " << node->frequency << endl;                
            }
            // cout << endl;
            curr->children.clear();
            for (FPNode* node : level) {
                for (auto it = node->children.begin(); it != node->children.end(); ++it) {
                    curr->children[it->first] = it->second;
                }
            }

            lev++ ;
        }

        
        cout << "edge frequency map:" << endl ;

        // now select the edge with maximum frequency and merge the two items
       int max_freq = 0 ;
        string max_edge = "";
        for(auto it = edge_frequency.begin(); it != edge_frequency.end(); ++it){
            // print the edge and its frequency
            cout << it->first << " " << it->second << endl;
            if(it->second > max_freq){
                max_freq = it->second;
                max_edge = it->first;
            }
        }

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
        FPNode *temp = new FPNode("", nullptr);
        temp->children = root->children;
        while(temp->children.size() > 0) {
            vector<FPNode*> level;
            for (auto it = temp->children.begin(); it != temp->children.end(); ++it) {
                level.push_back(it->second);
            }
            

            // iterate over the nodes in the level to find node1
            for (FPNode* node : level) {
                if(node->item == item1){
                    // if found then merge this node with its child node with item2
                    // first find the nodes corresponding to the two items
                    FPNode* node1 = node;
                    // check if node1 has a child with item2
                    if(node1->children.find(item2) == node1->children.end()){
                        continue;
                    }

                    // node 2 is a child of node 1
                    FPNode* node2 = node1->children[item2];

                    // cout << "Hi Guys" << endl ;
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


                }
            }

            temp->children.clear();
            for (FPNode* node : level) {
                for (auto it = node->children.begin(); it != node->children.end(); ++it) {
                    temp->children[it->first] = it->second;
                }
            }



        }


        // // first find the nodes corresponding to the two items
        // FPNode* node1 = root->children[item1];
        // // node 2 is a child of node 1
        // FPNode* node2 = node1->children[item2];


        // // now merge the two nodes
        // node1->item = item1 + "_" + item2;
        // node1->frequency = max_freq;
        // node1->children = node2->children;

        // // update the children map of the parent of node 1
        // node1->parent->children.erase(item1);
        // node1->parent->children[item1 + "_" + item2] = node1;



        // // cout << "HI GUYS" << endl;

        // // now remove the node2 from the tree
        // node2->parent->children.erase(item2);

        // // now update the parent pointers of the children of node2
        // for(auto it = node2->children.begin(); it != node2->children.end(); ++it){
        //     it->second->parent = node1;
        // }

        curr = root ;

        }

        


    }

    // print the fp tree
    void print() {
        cout << "fptree:" << endl ; 
        
        // create a copy of the root node for deletion
        FPNode* curr = new FPNode("", nullptr);
        curr->children = root->children;


        // print the entire fp tree in level order
        while(curr->children.size() > 0) {
            vector<FPNode*> level;
            for (auto it = curr->children.begin(); it != curr->children.end(); ++it) {
                level.push_back(it->second);
            }
            for (FPNode* node : level) {
                cout << node->item << " " << node->frequency << " ";
            }
            cout << endl;
            curr->children.clear();
            for (FPNode* node : level) {
                for (auto it = node->children.begin(); it != node->children.end(); ++it) {
                    curr->children[it->first] = it->second;
                }
            }
        }

    }
};

int main() {
    vector<vector<string> > dataset = {
        {"A", "B", "C", "D", "E"},
        {"A", "B", "C", "D", "F"},
        {"A", "B", "C", "D", "E", "G", "A", "B"},
        {"A", "B", "C", "D", "E", "F", "G"}
    };

    // int min_support = 10;

    FPTree fp_tree;

    for (const auto& transaction : dataset) {
        fp_tree.insert(transaction);
        // fp_tree.update_support(transaction);
    }

    // print the fp tree
    fp_tree.print();

    // now merge the fp tree
    fp_tree.merge();

    // print the fp tree
    fp_tree.print();

    // vector<vector<string>> frequent_itemsets;
    // fp_tree.get_frequent_itemsets(min_support, frequent_itemsets);

    // map<set<string>, string> mapping;
    // for (size_t i = 0; i < frequent_itemsets.size(); ++i) {
    //     set<string> itemset(frequent_itemsets[i].begin(), frequent_itemsets[i].end());
    //     mapping[itemset] = "X" + to_string(i);
    // }

    // vector<vector<string>> compressed_dataset;
    // for (const auto& transaction : dataset) {
    //     vector<string> compressed_transaction;
    //     for (const string& item : transaction) {
    //         set<string> itemset = {item};
    //         auto it = mapping.find(itemset);
    //         if (it != mapping.end()) {
    //             compressed_transaction.push_back(it->second);
    //         } else {
    //             compressed_transaction.push_back(item);
    //         }
    //     }
    //     compressed_dataset.push_back(compressed_transaction);
    // }

    

    // cout << "Mapping:" << endl;
    // for (const auto& mapping_entry : mapping) {
    //     cout << mapping_entry.second << ": ";
    //     for (const string& item : mapping_entry.first) {
    //         cout << item << " ";
    //     }
    //     cout << endl;
    // }

    // cout << "Compressed Dataset:" << endl;
    // for (const auto& transaction : compressed_dataset) {
    //     for (const string& item : transaction) {
    //         cout << item << " ";
    //     }
    //     cout << endl;
    // }

    return 0;
}