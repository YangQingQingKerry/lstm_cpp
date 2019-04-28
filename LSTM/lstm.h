#ifndef lstm_h
#define lstm_h

#include <iostream>
#include <vector>
#include "stdio.h"
#include "stdlib.h"
#define path "..//data//SP500.csv"



//全局变量 config
#define innode 5
#define hidenode 6
#define outnode 1
#define rollingwindow 20
#define Epoch 1000
#define alpha 0.01


using namespace std;
 



//-=====================================================
//Load csv file from path
class fetchdata{
public:
    vector<vector<double>> dataMatrix;
    fetchdata();
    void Finit(vector<vector<double>> &D);
};
    


//[lstm cell]===========================================
class lstm{
public:
    lstm(vector<vector<double>> D);
    void winit(double w[], int n);
    double sigmoid(double x);
    double dsigmoid(double x);
    double tanh(double x);
    double dtanh(double x);
    int min(int x, int y);
    void train();
    
    
public:
    vector<vector<double>>* pt;
    int len;
    double W_i[hidenode][hidenode];
    double U_i[hidenode][innode];
    double W_f[hidenode][hidenode];
    double U_f[hidenode][innode];
    double W_c[hidenode][hidenode];
    double U_c[hidenode][innode];
    double W_o[hidenode][hidenode];
    double U_o[hidenode][innode];
    double V[outnode][hidenode];
    double b_i[hidenode];
    double b_f[hidenode];
    double b_c[hidenode];
    double b_o[hidenode];
    double b_v[outnode];
};




//[激活函数]==============================================
//[Sigmoid]
class SigmoidActivator{
private:
    double weighted_input;
    double output;
public:
    SigmoidActivator(double x);
    double forward();
    double backward();
};
//[Tanh]
class TanhActivator{
private:
    double weighted_input;
    double output;
public:
    TanhActivator(double x);
    double forward();
    double backward();
};

 
#endif /* lstm_h */
