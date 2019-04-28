#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "assert.h"
#include "time.h"
#include "lstm.h"

#define uniform_def ((double)(2.0*rand())/((double)RAND_MAX+1.0)-1.0)

using namespace std;

// ============================================================================
// [read csv file]
fetchdata::fetchdata(){
    Finit(dataMatrix);
}
//成员函数Finit对成员变量进行赋值： 传递成员变量地址，以便对其进行更改
void  fetchdata::Finit(vector<vector<double>> &D){
    //load csv文件: [open, high, low, close, adj_close, column]
    ifstream inFile(path, ios::in);
    if(!inFile){
        cout << "Open Filed！" << endl;
        exit(1);
    }
    
    //存成二维表结构
    int row=0;
    string line;
    while(getline(inFile, line)){
        stringstream ss(line);
        string str;
        if(row>0){ //跳过第一行
            int i=0;
            vector<double> lineArray;
            while(getline(ss, str, ',')){
                if((i!=0) & (i!=3))//跳过第一列和close列
                    lineArray.push_back((double)atof( str.c_str() ));
                i++;
            }
            D.push_back(lineArray);
        }
        row++;
    }
}



//[lstm cell] =======================================================================
lstm::lstm(vector<vector<double>> D){
    pt =&D;
    winit((double*) W_i, hidenode*hidenode);
    winit((double*) U_i, hidenode*innode);
    winit((double*) W_f, hidenode*hidenode);
    winit((double*) U_f, hidenode*innode);
    winit((double*) W_c, hidenode*hidenode);
    winit((double*) U_c, hidenode*innode);
    winit((double*) W_o, hidenode*hidenode);
    winit((double*) U_o, hidenode*innode);
    winit((double*) V, outnode*hidenode);
    
    winit((double*) b_i, hidenode);
    winit((double*) b_f, hidenode);
    winit((double*) b_c, hidenode);
    winit((double*) b_o, hidenode);
    winit((double*) b_v, outnode);
    len= (int) D.size();
}

//[winit]
void lstm::winit(double w[], int n){
    for(int i=0; i<n;i++)
        w[i]=uniform_def;
}

int lstm::min(int x, int y){
    if(x<y)
        return x;
    else
        return y;
}

 

//[train]
void lstm::train(){
    for(int epoch=0; epoch<min(20,floor(len/rollingwindow)); epoch++){
        vector<double*> f_vector; //遗忘门
        vector<double*> i_vector; //输入门
        vector<double*> c_vector; //新信息
        vector<double*> o_vector; //遗忘门
        vector<double*> s_vector; //状态值
        vector<double*> h_vector; //输出值
        vector<double*> x_vector; //输出值
        vector<double> y_delta;
        double error=0.0;
        
        //在0时刻是没有之前的隐藏层的，所以初始化一个全为零的
        double *s=new double[hidenode];
        double *h=new double[hidenode];
        for(int i=0; i<hidenode;i++){
            s[i]=0;
            h[i]=0;
        }
        s_vector.push_back(s);
        h_vector.push_back(h);
        
        
        //正向传播
        for(int time=0; time<rollingwindow; time++){
            int index=epoch*rollingwindow+time; //当前的时间戳
            double *x=new double[innode]; //当前时刻的输入
            double *y=new double[outnode]; //当前时刻的输出
            double t;  //实际label
        
            for(int i=0; i<innode; i++)
                x[i]=(*pt)[index][i];
            x_vector.push_back(x);
            t=(*pt)[index+1][innode-2]; //close列
            
            double* Gate_i =new double[hidenode];
            double* Gate_o=new double[hidenode];
            double* Gate_f=new double[hidenode];
            double* Gate_c=new double[hidenode];
            double* state=new double[hidenode];
            double* h=new double[hidenode];
        
            
            for(int i=0; i< hidenode; i++){
                double igate=0;
                double ogate=0;
                double fgate=0;
                double cgate=0;
                
                for(int j=0; j<innode; j++){
                    igate+=U_i[i][j]*x[j];
                    ogate+=U_o[i][j]*x[j];
                    fgate+=U_f[i][j]*x[j];
                    cgate+=U_c[i][j]*x[j];
                }
                double* h_pre=h_vector.back();
                double* state_pre=s_vector.back();
                for(int j=0; j<hidenode; j++){
                    igate+=W_i[i][j]*h_pre[j]+b_i[j];
                    ogate+=W_o[i][j]*h_pre[j]+b_o[j];
                    fgate+=W_f[i][j]*h_pre[j]+b_f[j];
                    cgate+=W_c[i][j]*h_pre[j]+b_c[j];
                }
                
                

                Gate_i[i]=sigmoid(igate);
                Gate_o[i]=sigmoid(ogate);
                Gate_f[i]=sigmoid(fgate);
                Gate_c[i]=tanh(cgate);
                
                double s_pre=state_pre[i];
                state[i]=Gate_f[i]*s_pre+Gate_i[i]*Gate_c[i];
                h[i]=Gate_o[i]*tanh(state[i]);
            }
            
            for(int i=0; i<outnode; i++){
                //隐藏层传播到输出层
                double out=0.0;
                for(int j=0; j<hidenode; j++)
                    out+=V[i][j]*h[j]+b_v[j];
                y[i]=out; //线性单元
            }
            
            //保存隐藏层，以便下次计算
            i_vector.push_back(Gate_i);
            f_vector.push_back(Gate_f);
            o_vector.push_back(Gate_o);
            c_vector.push_back(Gate_c);
            s_vector.push_back(state);
            h_vector.push_back(h);
            y_delta.push_back(y[0]-t);
            error+=0.5*(y[0]-t)*(y[0]-t);
            
        }
        
        cout << error << endl;
        
        
        
        //误差反向传播
        //[当前」
        double h_delta[hidenode];
        double *o_delta=new double[hidenode];
        double *i_delta=new double[hidenode];
        double *f_delta=new double[hidenode];
        double *c_delta=new double[hidenode];
        double *state_delta=new double[hidenode];
        //[之后]
        double *o_fut_delta=new double[hidenode];
        double *i_fut_delta=new double[hidenode];
        double *f_fut_delta=new double[hidenode];
        double *c_fut_delta=new double[hidenode];
        double *state_fut_delta=new double[hidenode];
        double *f_gate_fut=new double[hidenode];
        for(int i=0; i<hidenode; i++){
            o_fut_delta[i]=0;
            i_fut_delta[i]=0;
            f_fut_delta[i]=0;
            c_fut_delta[i]=0;
            state_fut_delta[i]=0;
            f_gate_fut[i]=0;
        }
        
        for(int time=rollingwindow-1; time>=0; time--){
            double* x_in=x_vector[time];  //当前输入
            
            
            //当前隐藏层
            double * i_gate=i_vector[time];
            double * f_gate=f_vector[time];
            double * c_gate=c_vector[time];
            double * o_gate=o_vector[time];
            double * state=s_vector[time+1];
            double * h=h_vector[time+1];
            
            //前一个隐藏层
            double * state_pre=s_vector[time];
            double * h_pre=h_vector[time];
            
            for(int i=0; i<outnode; i++){ //更新输出层连接权重
                b_v[i]-=alpha*y_delta[time];
                for(int j=0; j<hidenode; j++)
                    V[i][j]-=alpha*y_delta[time]*h[i];
            }
            
            //更新隐藏层连接权重
            for(int i=0; i<hidenode; i++){
                h_delta[i]=0.0;
                for(int k=0; k<outnode;k++)
                    h_delta[i]+=y_delta[time]*V[k][i];
                for(int k=0; k<hidenode; k++){
                    h_delta[i]+=i_fut_delta[k]*W_i[k][i];
                    h_delta[i]+=f_fut_delta[k]*W_f[k][i];
                    h_delta[i]+=c_fut_delta[k]*W_c[k][i];
                    h_delta[i]+=o_fut_delta[k]*W_o[k][i];
                }
                
                
                
                //隐含层的矫正误差
                state_delta[i]=h_delta[i]*o_gate[i]*dtanh(state[i]);
                o_delta[i]=h_delta[i]*(tanh(state[i]))*dsigmoid(o_gate[i]);
                f_delta[i]=state_delta[i]*state_pre[i]*dsigmoid(f_gate[i]);
                i_delta[i]=state_delta[i]*c_gate[i]*dsigmoid(i_gate[i]);
                c_delta[i]=state_delta[i]*i_gate[i]*dtanh(c_gate[i]);
                
                //更新前一个隐藏层和现在隐藏层之间的权重
                for(int k=0; k<hidenode; k++){
                    W_i[i][k]-=alpha*i_delta[i]*h_pre[k];
                    W_o[i][k]-=alpha*o_delta[i]*h_pre[k];
                    W_c[i][k]-=alpha*c_delta[i]*h_pre[k];
                    W_f[i][k]-=alpha*f_delta[i]*h_pre[k];
                }
                
                
                //更新output-hideput权重矩阵
                for(int k=0; k<innode; k++){
                    U_i[i][k]-=alpha*i_delta[i]*x_in[k];
                    U_o[i][k]-=alpha*o_delta[i]*x_in[k];
                    U_c[i][k]-=alpha*c_delta[i]*x_in[k];
                    U_f[i][k]-=alpha*f_delta[i]*x_in[k];
                }
                
                //更新偏移项
                b_i[i]-=alpha*i_delta[i];
                b_o[i]-=alpha*o_delta[i];
                b_c[i]-=alpha*c_delta[i];
                b_f[i]-=alpha*f_delta[i];
                

                
                
            }
            
            
        
            o_fut_delta=o_delta;
            f_fut_delta=f_delta;
            i_fut_delta=i_delta;
            c_fut_delta=c_delta;
            f_gate_fut=f_gate;
            state_fut_delta=state_delta;
            
        }
         
         
        
        
        
        
    
    }
    
    
}


//[sigmoid]
double lstm::sigmoid(double x){
    return 1.0/(1.0+exp(-x));
}
double lstm::dsigmoid(double x){
    return x*(1-x);
}
//[Tanh]
double lstm::tanh(double x){
    return 2.0/(1.0+exp(-2*x))-1;
}
double lstm::dtanh(double x){
    return 1-x*x;
}


