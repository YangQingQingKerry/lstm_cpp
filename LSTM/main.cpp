#include <iostream>
#include "lstm.h"



using namespace std;

int main(){
    fetchdata df;
    lstm ins(df.dataMatrix);
    ins.train();
    
     
    return 0;
}


/*
 1. normalize the dataset
 2. increase the length of the input
 */
